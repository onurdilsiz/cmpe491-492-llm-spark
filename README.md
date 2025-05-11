## Repository Structure

- **agents/sparky-langgraph/**: Contains the LangGraph implementation of Sparky
  - [Working Demo](agents/sparky-langgraph/working_demo.py): A fully functional implementation using LangGraph
  - [README](agents/sparky-langgraph/README.md): Detailed documentation for the LangGraph implementation

## Getting Started

The fastest way to try Sparky is to use the LangGraph implementation:

1. Navigate to the LangGraph implementation:

   ```
   cd agents/sparky-langgraph
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Prepare your Spark UI PDFs:

   - Download PDFs from your Spark UI (environment, executors, jobs, sql, stages, storage)
   - Place them in a directory structure like:
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

4. Run the demo:
   ```
   python working_demo.py
   ```
