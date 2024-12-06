import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def get_prompt(case_type):
    # Create a system message template
    system_template = "You are an AI expert in distributed computing with deep knowledge of Apache Spark."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Create a user message template, including placeholders for case_type and code
    user_template = "Analyze the following Spark code for {case_type}: {code}"
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    # Combine system and user messages into a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    return prompt

def process_code_with_prompt(code, case_type):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")  # Use Gemini model
    prompt = get_prompt(case_type)
    
    # Use format_prompt() instead of format()
    chat_prompt_value = prompt.format_prompt(case_type=case_type, code=code)
    messages = chat_prompt_value.to_messages()

    response = llm.invoke(messages)
    return response

def main():
    # File paths for local input and output
    input_file_path = "MatchingCode.py"
    output_file_path = "output-analysis.txt"

    # Case type to analyze
    case_type = "RDD vs DataFrame"

    # Read Spark code from local file
    if not os.path.exists(input_file_path):
        print(f"Error: File '{input_file_path}' does not exist.")
        return
    
    with open(input_file_path, "r") as file:
        spark_code = file.read()

    # Process code with LangChain
    print(f"Analyzing code for case: {case_type}...")
    analysis = process_code_with_prompt(spark_code, case_type)

    # Write analysis results to local file
    with open(output_file_path, "w") as file:
        file.write(analysis)

    print(f"Analysis complete. Results saved to '{output_file_path}'.")

if __name__ == "__main__":
    main()
