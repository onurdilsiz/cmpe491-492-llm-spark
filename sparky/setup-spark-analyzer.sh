#!/bin/bash
# setup_spark_analyzer.sh

# Exit on error
set -e

echo "Setting up Spark UI Analyzer environment..."

# Create virtual environment
python3 -m venv spark_analyzer_env
source spark_analyzer_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install langchain langchain-openai pypdf tqdm python-dotenv

# Create a basic .env file for OpenAI API key if it doesn't exist
if [ ! -f .env ]; then
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    echo "Created .env file. Please edit it to add your OpenAI API key."
fi

# Make the CLI script executable
chmod +x spark_analyzer_cli.py

# Create base directory structure
mkdir -p spark_analyzer_data

# Initialize the directory structure
python3 spark_analyzer_cli.py setup --base-dir ./spark_analyzer_data

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source spark_analyzer_env/bin/activate"
echo ""
echo "To create a sample run with empty PDFs for testing:"
echo "  python spark_analyzer_cli.py create-sample --run-id $(date +%Y%m%d_%H%M%S) --base-dir ./spark_analyzer_data"
echo ""
echo "To list available runs:"
echo "  python spark_analyzer_cli.py list-runs --base-dir ./spark_analyzer_data"
echo ""
echo "To analyze a run:"
echo "  python spark_analyzer_cli.py analyze --run-id RUN_ID --base-dir ./spark_analyzer_data"
echo ""
echo "Don't forget to add your OpenAI API key to the .env file!"
