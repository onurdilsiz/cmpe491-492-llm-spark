# Spark UI Analyzer

A LangGraph-powered tool for analyzing Apache Spark application performance using Spark UI data.

## Overview

This tool helps Spark developers and data engineers analyze their Spark application performance by extracting insights from Spark UI PDFs. It uses a multi-agent system powered by LangGraph and LangChain to provide detailed performance analysis and optimization recommendations.

## Features

- **Multi-agent architecture**: Specialized agents analyze different aspects of Spark performance
- **PDF text extraction**: Automatically extracts text from Spark UI PDFs
- **Interactive chat interface**: Ask questions about your Spark application in natural language
- **Specialized analysis areas**:
  - Autoscaling configuration
  - Shuffle partition optimization
  - Data skew detection
  - Memory utilization
  - OOM error diagnosis
  - Performance bottleneck identification
  - SQL query optimization

## Prerequisites

- Python 3.9+
- Required Python packages (see `requirements.txt`)
- Spark UI PDFs from your application run

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Preparing Spark UI PDFs

1. Run your Spark application
2. Download the following PDFs from Spark UI:
   - environment.pdf
   - executors.pdf
   - jobs.pdf
   - sql.pdf
   - stages.pdf
   - storage.pdf
3. Place these PDFs in a folder structure like:
   ```
   spark_analyzer_data/
   └── runs/
       └── spark_run_1/
           ├── environment.pdf
           ├── executors.pdf
           ├── jobs.pdf
           ├── sql.pdf
           ├── stages.pdf
           └── storage.pdf
   ```

### Running the Analyzer

Run the demo script:

```
python working_demo.py
```

The script will:

1. Load the PDF files
2. Initialize a chat session
3. Allow you to ask questions about your Spark application

### Example Questions

- "Are my executors properly utilized?"
- "Is there data skew in my application?"
- "What are the top 5 longest-running jobs?"
- "Should I adjust my shuffle partitions?"
- "Is my broadcast join threshold configured correctly?"

## Architecture

The system uses a LangGraph workflow with the following components:

1. **Orchestrator**: Determines which specialist agents to call based on the user query
2. **Specialist Agents**:
   - jobs_agent: Analyzes job-level metrics
   - stages_agent: Examines stage execution details
   - executors_agent: Analyzes executor utilization
   - sql_agent: Examines SQL query performance
   - storage_agent: Analyzes storage and caching
   - environment_agent: Examines Spark configuration
3. **Synthesizer**: Combines findings from specialists into a coherent analysis
4. **Formatter**: Formats the final response for the user

## Customization

You can customize the system by:

- Modifying specialist agent prompts in `create_specialist_node()`
- Adjusting the synthesizer prompt in `synthesizer_agent()`
- Adding new specialist agents for additional analysis areas

## Troubleshooting

- **PDF loading issues**: Ensure PDFs are in the correct location and format
- **Empty responses**: Check if the PDFs contain the expected text content
- **Memory errors**: Reduce the `max_text_length` parameter if processing large PDFs
