# Spark UI Performance Analyzer

A multi-agent LLM system for analyzing Apache Spark UI data to identify performance bottlenecks and suggest optimizations.

## Overview

This system uses specialized AI agents to analyze PDF exports from various Apache Spark UI tabs:

- **Jobs Analyzer**: Examines job-level metrics and bottlenecks
- **Stages Analyzer**: Focuses on stage execution details and data skew
- **Storage Analyzer**: Reviews caching strategies and memory utilization
- **Environment Analyzer**: Evaluates configuration settings
- **Executors Analyzer**: Assesses resource utilization and executor performance
- **SQL Analyzer**: Identifies query performance issues
- **Synthesizer**: Integrates all findings into a comprehensive report

## Directory Structure

The system uses the following directory structure:

```
spark_analyzer_data/
├── runs/
│   ├── spark_run_20250417_143022/  # Timestamp format: YYYYMMDD_HHMMSS
│   │   ├── jobs.pdf
│   │   ├── stages.pdf
│   │   ├── storage.pdf
│   │   ├── environment.pdf
│   │   ├── executors.pdf
│   │   └── sql.pdf
├── outputs/
│   ├── spark_run_20250417_143022/
│   │   ├── jobs_analysis.md
│   │   ├── stages_analysis.md
│   │   ├── storage_analysis.md
│   │   ├── environment_analysis.md
│   │   ├── executors_analysis.md
│   │   ├── sql_analysis.md
│   │   └── final_report.md
└── logs/
    └── spark_run_20250417_143022_analysis.log
```

## Installation

### Prerequisites

- Python 3.8 or higher
- An OpenAI API key (for accessing GPT-4 or similar models)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/spark-ui-analyzer.git
   cd spark-ui-analyzer
   ```

2. Run the setup script:
   ```bash
   bash setup_spark_analyzer.sh
   ```

3. Edit the `.env` file to add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Activate the virtual environment:
   ```bash
   source spark_analyzer_env/bin/activate
   ```

## Usage

### Preparing Spark UI Data

1. From your running Spark application, access the Spark UI (typically on port 4040)
2. For each tab (Jobs, Stages, Storage, Environment, Executors, SQL):
   - Navigate to the tab
   - Use your browser's print function (Ctrl+P or Cmd+P)
   - Save as PDF with the appropriate name (jobs.pdf, stages.pdf, etc.)
3. Create a new run directory:
   ```bash
   mkdir -p spark_analyzer_data/runs/spark_run_$(date +%Y%m%d_%H%M%S)
   ```
4. Move your PDFs to this directory:
   ```bash
   mv *.pdf spark_analyzer_data/runs/spark_run_YYYYMMDD_HHMMSS/
   ```

### Running the Analysis

1. List available runs:
   ```bash
   python spark_analyzer_cli.py list-runs --base-dir ./spark_analyzer_data
   ```

2. Run the analysis:
   ```bash
   python spark_analyzer_cli.py analyze --run-id YYYYMMDD_HHMMSS --base-dir ./spark_analyzer_data
   ```

3. View the results:
   ```bash
   ls -la spark_analyzer_data/outputs/spark_run_YYYYMMDD_HHMMSS/
   ```

### CLI Commands

The CLI tool provides several commands:

- **setup**: Initialize the directory structure
  ```bash
  python spark_analyzer_cli.py setup --base-dir ./spark_analyzer_data
  ```

- **create-sample**: Create empty sample PDFs for testing
  ```bash
  python spark_analyzer_cli.py create-sample --run-id YYYYMMDD_HHMMSS --base-dir ./spark_analyzer_data
  ```

- **list-runs**: Show available Spark runs
  ```bash
  python spark_analyzer_cli.py list-runs --base-dir ./spark_analyzer_data
  ```

- **analyze**: Analyze a specific Spark run
  ```bash
  python spark_analyzer_cli.py analyze --run-id YYYYMMDD_HHMMSS --base-dir ./spark_analyzer_data [--model gpt-4o] [--temperature 0.0]
  ```

## Agent Descriptions

### Jobs Analyzer Agent

Analyzes the Jobs tab to identify:
- Jobs with unusually long duration
- Failed jobs and error patterns
- Jobs with disproportionate number of stages
- Sequential job execution that could be parallelized

### Stages Analyzer Agent

Focuses on stage-level performance issues:
- Data skew (task duration variance)
- Excessive shuffle operations
- High GC overhead
- Serialization bottlenecks
- Task distribution issues

### Storage Analyzer Agent

Evaluates storage and caching strategies:
- Inefficient storage levels
- Memory pressure from excessive caching
- Under-utilized cache
- Uneven partition sizes
- Spill-to-disk issues

### Environment Analyzer Agent

Reviews configuration settings for optimization:
- Memory allocation issues
- Inefficient parallelism settings
- Missing performance-critical configurations
- Suboptimal serialization/compression settings
- Outdated software versions

### Executors Analyzer Agent

Examines executor performance:
- Uneven task distribution
- Excessive GC time
- Memory pressure indicators
- Resource utilization imbalances
- Network/disk I/O bottlenecks

### SQL Analyzer Agent

Identifies SQL query performance issues:
- Expensive join operations
- Missing predicate pushdown/partition pruning
- Excessive shuffling
- Suboptimal join ordering
- Large-scale sorts
- Data skew in joins/aggregations

### Synthesizer Agent

Integrates findings across all areas:
- Identifies cross-cutting performance issues
- Prioritizes bottlenecks by severity and impact
- Formulates holistic recommendations
- Creates an implementation plan

## Customization

### Modifying Agent Prompts

Agent prompts are defined in the `agent_prompts` dictionary in `spark_analyzer.py`. You can modify these prompts to:
- Focus on specific metrics
- Add new analysis techniques
- Customize output formatting
- Target specific Spark performance issues

### Using Different LLM Models

By default, the system uses `gpt-4o`, but you can specify different models:

```bash
python spark_analyzer_cli.py analyze --run-id YYYYMMDD_HHMMSS --base-dir ./spark_analyzer_data --model gpt-3.5-turbo
```

For Claude or other models, you'll need to modify the imports and model initialization in `spark_analyzer.py`.

## Limitations

- PDF extraction can be imperfect, especially for complex tables
- Analysis quality depends on the completeness of the Spark UI data
- Large PDFs may need to be split or processed in chunks
- The system analyzes static snapshots and cannot observe dynamic behavior over time

## Troubleshooting

- **Missing PDFs**: Ensure all six required PDFs are in the run directory
- **API Key Issues**: Check that your OpenAI API key is correctly set in the `.env` file
- **PDF Extraction Problems**: For complex PDFs, consider preprocessing them using specialized tools
- **Memory Issues**: For very large PDFs, try running with increased Python memory limits

## License

[MIT License](LICENSE)
