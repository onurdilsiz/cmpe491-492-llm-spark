import os
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
import csv
import json
import logging
from dotenv import load_dotenv
import time
load_dotenv()

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

def setup_logging(log_file):
    """Setup logging to a file."""
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Logging initialized.")



def load_user_template(prompt_file):
    """Load a user template from the specified file."""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file '{prompt_file}' does not exist.")
    with open(prompt_file, "r") as file:
        return file.read().strip()


def get_prompt( prompt_file):
    # Create a system message template
    system_template = "You are an AI expert in distributed computing with deep knowledge of Apache Spark."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Create a user message template, including placeholders for case_type and code
    user_template = load_user_template(prompt_file)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    # Combine system and user messages into a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    return prompt

def process_code_with_prompt(code, prompt_file, llm):
   
    prompt = get_prompt(prompt_file)
    
    chat_prompt_value = prompt.format_prompt( code=code)
    messages = chat_prompt_value.to_messages()
    response = llm.invoke(messages)
    response_tokens = llm.get_num_tokens(response.content)
    prompt_tokens= llm.get_num_tokens_from_messages(messages)

    return response, prompt_tokens, response_tokens


def analyze_directory(dataset_dir, cases, models, prompts_dir, output_dir,csv_file,csv_file2):
    """Analyze all files in the dataset directory with prompts from the prompts directory."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    detection_results = []

    for file_name in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, file_name)

        # Skip non-Python files
        if not file_name.endswith(".py"):
            logging.info(f"Skipping non-Python file: {file_name}")
            continue
        print(f"Processing file: {file_name}")
        logging.info(f"Processing file: {file_name}")
        with open(file_path, "r") as file:
            code = file.read()

        for model_name, llm in models.items():  # Iterate over initialized LLM objects
            for i, case_type in enumerate(cases):
                # Map each case to a corresponding prompt file (prompt0.txt, prompt1.txt, etc.)
                prompt_file = os.path.join(prompts_dir, f"prompt{i}.txt")
                logging.info(f"Analyzing case: {case_type} with model: {model_name} using prompt: {prompt_file}")

           
                # Process code with the model, case type, and prompt
                try:
                    # Measure start time
                    start_time = time.time()
                    analysis, prompt_tokens, response_tokens = process_code_with_prompt(code, prompt_file, llm)
                    end_time = time.time()
                    latency = end_time - start_time
                    logging.info(f"Analysis completed in {latency:.2f} seconds")
                except Exception as e:
                    logging.error(f"Error processing {file_name} with prompt {prompt_file}: {e}")
                    continue
                
                total_tokens = prompt_tokens + response_tokens
                results.append([model_name, case_type, file_name, prompt_tokens, response_tokens, total_tokens,latency])


                # Save the result to a uniquely named file
                output_file_name = f"{model_name}_{case_type.replace(' ', '_')}_{file_name.replace('.py', '')}.json"
                output_file_path = os.path.join(output_dir, output_file_name)


                # Assuming `analysis.content` contains the response with ```json markers
                raw_response = analysis.content

                # Strip out the ```json markers
                if raw_response.startswith("```json"):
                    raw_response = raw_response[7:]  # Remove the opening ```json
                raw_response = raw_response.rstrip()  # Ensures no trailing artifacts remain

                if raw_response.endswith("```"):
                    raw_response = raw_response[:-3]  # Remove the closing ```

                # Parse the cleaned JSON content
                parsed_json = {}

                try:
                    parsed_json = json.loads(raw_response)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON: {e}")
                    
                

                # Write the cleaned and parsed JSON to a file
                with open(output_file_path, "w") as output_file:
                    json.dump(parsed_json, output_file, indent=4)

                detection_result = parsed_json.get("detected", False)
                occurrences = parsed_json.get("occurrences", 0)
                detection_results.append([model_name, case_type, file_name, latency, detection_result, occurrences])

                logging.info(f"Analysis for {file_name} saved to {output_file_path}")
                output_file_path = output_file_path.replace(".json", ".txt")


                with open(output_file_path, "w") as output_file:
                    output_file.write(analysis.content)

               
                logging.info(f"Analysis for {file_name} saved to {output_file_path}")

    with open(csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Model", "Case", "File", "Input Tokens", "Output Tokens", "Total Tokens","Latency"])
        csv_writer.writerows(results)
        logging.info(f"All results logged to CSV: {csv_file}")
        print(f"All results logged to CSV: {csv_file}")

    with open(csv_file2, mode="w", newline="", encoding="utf-8") as csv_file2:
        csv_writer = csv.writer(csv_file2)
        csv_writer.writerow(["Model", "Case", "File","Latency", "Detection Result"])
        csv_writer.writerows(detection_results)
        logging.info(f"All results logged to CSV: {csv_file2}")
        print(f"All results logged to CSV: {csv_file2}")




def init_models(models):
    """Initialize the models for analysis."""
    
    llms = {}
    for model in models:
        if model.startswith("gemini") or model.startswith("chat-bison"):  
            try:        
                llms[model] = ChatVertexAI(
                    model=model,
                    temperature=0,
                    max_tokens=None,
                    max_retries=6,
                    stop=None,
                )
            except Exception as e:
                logging.error(f"Error initializing model {model}: {e}")
        
        elif model.startswith("meta"):
            try:    
                llms[model] = ChatVertexAI(    
                    model=model,
                    temperature=0,
                    max_tokens=None,
                    max_retries=6,
                    stop=None,
                    client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"},  # Ensure region matches deployment
)
            except Exception as e:
                logging.error(f"Error initializing model {model}: {e}")

        else:
            try:    
                llms[model] = ChatOpenAI(
                    model=model,  # Strip prefix for OpenAI models
                    temperature=0,
                    max_tokens=None,
                    max_retries=1,
                    stop=None,
                    request_timeout=30,  # 30 seconds timeout
                    api_key=OPENAI_API_KEY,
                )
            except Exception as e:
                logging.error(f"Error initializing model {model}: {e}")
    return llms
    

def main():
    log_file = "analysis05.01.log"
    setup_logging(log_file)
    # File paths for local input and output
    config = {
        "models": ["meta/llama-3.1-405b-instruct-maas" ], #meta/llama-3.1-8b-instruct-maas, gpt-3.5-turbo-0125, "gemini-1.0-pro-002""gemini-1.5-flash-002", "gemini-2.0-flash-exp"
        "cases": ["RDD vs DataFrame", "Coalesce vs Repartition", "Map vs MapPartitions", "Serialized Data Formats", "Avoiding UDFs","All"],
        "dataset_dir": "dataset",  # Directory containing all the dataset files
        "prompts_dir": "prompts",  # Directory containing prompt files
        "output_dir": "output05.01",   # Directory to save the analysis results
        "csv_file": "tokenresults05.01.csv",
        "csv_file2": "detectionresults05.01.csv"
    }
    
    # Initialize the models for analysis
    llms = init_models(config["models"])
    print("Analyzing directory")

    analyze_directory(
        config["dataset_dir"], 
        config["cases"], 
        llms,
        config["prompts_dir"], 
        config["output_dir"],
        config["csv_file"],
        config["csv_file2"]
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Fatal error occurred: {e}", exc_info=True)
