import os
import time
import json
import csv
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Logging setup
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Logging initialized.")

# Load user template
def load_user_template(prompt_file):
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file '{prompt_file}' does not exist.")
    with open(prompt_file, "r") as file:
        return file.read().strip()

# Generate prompt
def get_prompt(prompt_file):
    system_template = "You are an AI expert in distributed computing with deep knowledge of Apache Spark."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    user_template = load_user_template(prompt_file)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    return prompt

# FAISS index initialization
def initialize_faiss_index(index_path, documents):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [text_splitter.split_text(doc) for doc in documents]
    chunks_flattened = [item for sublist in chunks for item in sublist]
    
    if os.path.exists(index_path):
        retriever = FAISS.load_local(index_path, embeddings)
    else:
        retriever = FAISS.from_texts(chunks_flattened, embeddings)
        retriever.save_local(index_path)
    
    return retriever

# Retrieve context
def retrieve_context(query, retriever, top_k=3):
    results = retriever.similarity_search(query, k=top_k)
    context = "\n".join([result.page_content for result in results])
    return context

# Process code with RAG
def process_code_with_prompt_rag(code, prompt_file, llm, retriever):
    prompt = get_prompt(prompt_file)
    context = retrieve_context(code, retriever)

    chat_prompt_value = prompt.format_prompt(context=context, code=code)
    messages = chat_prompt_value.to_messages()
    response = llm.invoke(messages)
    response_tokens = llm.get_num_tokens(response.content)
    prompt_tokens = llm.get_num_tokens_from_messages(messages)

    return response, prompt_tokens, response_tokens

# Analyze directory
def analyze_directory(dataset_dir, cases, models, prompts_dir, output_dir, csv_file, csv_file2, retriever):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    detection_results = []

    for file_name in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, file_name)
        if not file_name.endswith(".py"):
            logging.info(f"Skipping non-Python file: {file_name}")
            continue
        
        logging.info(f"Processing file: {file_name}")
        with open(file_path, "r") as file:
            code = file.read()

        for model_name, llm in models.items():
            for i, case_type in enumerate(cases):
                prompt_file = os.path.join(prompts_dir, f"prompt{i}.txt")
                logging.info(f"Analyzing case: {case_type} with model: {model_name} using prompt: {prompt_file}")

                try:
                    start_time = time.time()
                    analysis, prompt_tokens, response_tokens = process_code_with_prompt_rag(code, prompt_file, llm, retriever)
                    end_time = time.time()
                    latency = end_time - start_time
                    logging.info(f"Analysis completed in {latency:.2f} seconds")
                except Exception as e:
                    logging.error(f"Error processing {file_name} with prompt {prompt_file}: {e}")
                    continue
                
                total_tokens = prompt_tokens + response_tokens
                results.append([model_name, case_type, file_name, prompt_tokens, response_tokens, total_tokens, latency])

                output_file_name = f"{model_name}_{case_type.replace(' ', '_')}_{file_name.replace('.py', '')}.json"
                output_file_path = os.path.join(output_dir, output_file_name)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                raw_response = analysis.content
                if raw_response.startswith("```json"):
                    raw_response = raw_response[7:]
                raw_response = raw_response.rstrip()
                if raw_response.endswith("```"):
                    raw_response = raw_response[:-3]

                try:
                    parsed_json = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON: {e}")
                    continue

                with open(output_file_path, "w") as output_file:
                    json.dump(parsed_json, output_file, indent=4)

                detection_result = parsed_json.get("detected", False)
                occurrences = parsed_json.get("occurrences", 0)
                detection_results.append([model_name, case_type, file_name, latency, detection_result, occurrences])

    with open(csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Model", "Case", "File", "Input Tokens", "Output Tokens", "Total Tokens", "Latency"])
        csv_writer.writerows(results)
        logging.info(f"All results logged to CSV: {csv_file}")

    with open(csv_file2, mode="w", newline="", encoding="utf-8") as csv_file2:
        csv_writer = csv.writer(csv_file2)
        csv_writer.writerow(["Model", "Case", "File", "Latency", "Detection Result"])
        csv_writer.writerows(detection_results)
        logging.info(f"All results logged to CSV: {csv_file2}")

# Initialize models
def init_models(models):
    llms = {}
    for model in models:
        if model.startswith("gemini"):
            try:
                llms[model] = ChatVertexAI(model=model, temperature=0, max_tokens=None, max_retries=6, stop=None)
            except Exception as e:
                logging.error(f"Error initializing model {model}: {e}")
        elif model.startswith("publishers"):
            try:
                llms[model] = ChatVertexAI(model="publishers/meta/models/llama-3.1-8b-instruct-maas", temperature=0, max_tokens=None, max_retries=6, stop=None)
            except Exception as e:
                logging.error(f"Error initializing model {model}: {e}")
        else:
            try:
                llms[model] = ChatOpenAI(model=model, temperature=0, max_tokens=None, max_retries=1, stop=None, request_timeout=180, api_key=OPENAI_API_KEY)
            except Exception as e:
                logging.error(f"Error initializing model {model}: {e}")
    return llms

# Main function with RAG
def main_with_rag():
    log_file = "analysis_with_rag.log"
    setup_logging(log_file)
    config = {
        "models": ["publishers/meta/models/llama-3.1-8b-instruct-maas"],
        "cases": ["RDD vs DataFrame", "Coalesce vs Repartition", "Map vs MapPartitions", "Serialized Data Formats", "Avoiding UDFs", "All"],
        "dataset_dir": "dataset2",
        "prompts_dir": "prompts",
        "output_dir": "output_with_rag",
        "csv_file": "tokenresults_with_rag.csv",
        "csv_file2": "detectionresults_with_rag.csv",
        "index_path": "faiss_index"
    }

    pdf_path = "../resources/LearningSpark2.0.pdf"  # Replace with the path to your PDF file
    pdf_text = extract_text_from_pdf(pdf_path)
    documents = [pdf_text]


    retriever = initialize_faiss_index(config["index_path"], documents)
    llms = init_models(config["models"])
    analyze_directory(config["dataset_dir"], config["cases"], llms, config["prompts_dir"], config["output_dir"], config["csv_file"], config["csv_file2"], retriever)

if __name__ == "__main__":
    try:
        main_with_rag()
    except Exception as e:
        logging.error(f"Fatal error occurred: {e}", exc_info=True)
