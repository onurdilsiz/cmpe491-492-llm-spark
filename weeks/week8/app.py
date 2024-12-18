import os
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_vertexai import ChatVertexAI

def load_user_template(prompt_file):
    """Load a user template from the specified file."""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file '{prompt_file}' does not exist.")
    with open(prompt_file, "r") as file:
        return file.read().strip()



def get_prompt(case_type, prompt_file):
    # Create a system message template
    system_template = "You are an AI expert in distributed computing with deep knowledge of Apache Spark."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Create a user message template, including placeholders for case_type and code
    user_template = load_user_template(prompt_file)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    # Combine system and user messages into a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    return prompt

def process_code_with_prompt(code, case_type,prompt_file):
    llm = ChatVertexAI(
        model="gemini-1.5-flash-002",
        temperature=0,
        max_tokens=None,
        max_retries=6,
        stop=None,
    )
    prompt = get_prompt(case_type, prompt_file)
    
    # Use format_prompt() instead of format()
    chat_prompt_value = prompt.format_prompt(case_type=case_type, code=code)
    messages = chat_prompt_value.to_messages()

    response = llm.invoke(messages)
    return response


def analyze_directory(dataset_dir, cases, models, prompts_dir, output_dir):
    """Analyze all files in the dataset directory with prompts from the prompts directory."""
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        for i, case_type in enumerate(cases):
            # Map each case to a corresponding prompt file (prompt0.txt, prompt1.txt, etc.)
            prompt_file = os.path.join(prompts_dir, f"prompt{i}.txt")
            print(f"Analyzing case: {case_type} with model: {model} using prompt: {prompt_file}")

            for file_name in os.listdir(dataset_dir):
                file_path = os.path.join(dataset_dir, file_name)

                # Skip non-Python files
                if not file_name.endswith(".py"):
                    print(f"Skipping non-Python file: {file_name}")
                    continue

                print(f"Processing file: {file_name}")
                with open(file_path, "r") as file:
                    code = file.read()

                # Process code with the model, case type, and prompt
                try:
                    analysis = process_code_with_prompt(code, case_type, model, prompt_file)
                except Exception as e:
                    print(f"Error processing {file_name} with prompt {prompt_file}: {e}")
                    continue

                # Save the result to a uniquely named file
                output_file_name = f"{model}_{case_type.replace(' ', '_')}_{file_name.replace('.py', '')}.txt"
                output_file_path = os.path.join(output_dir, output_file_name)

                with open(output_file_path, "w") as output_file:
                    output_file.write(analysis.content)

                print(f"Analysis for {file_name} saved to {output_file_path}")



def main():
    # File paths for local input and output
    config = {
        "models": ["gemini-1.0", "gemini-1.5-flash-002", "gemini-1.5-pro"],
        "cases": ["RDD vs DataFrame", "Coalesce vs Repartition", "Map vs MapPartitions"],
        "dataset_dir": "dataset",  # Directory containing all the dataset files
        "prompts_dir": "prompts",  # Directory containing prompt files
        "output_dir": "output",   # Directory to save the analysis results
    }
    

    analyze_directory(
        config["dataset_dir"], 
        config["cases"], 
        config["models"], 
        config["prompts_dir"], 
        config["output_dir"]
    )


if __name__ == "__main__":
    main()
