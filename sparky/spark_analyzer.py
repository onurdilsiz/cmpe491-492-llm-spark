# spark_analyzer.py
import os
import time
import json
import logging
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# PDF Processing
from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain components
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("spark_analyzer.log")
    ]
)
logger = logging.getLogger("spark_analyzer")

class SparkUIAnalyzerAgent:
    """Base class for Spark UI analyzer agents"""
    
    def __init__(self, agent_type: str, pdf_path: Path, output_path: Path, 
                 model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize the agent.
        
        Args:
            agent_type: Type of the agent (jobs, stages, etc.)
            pdf_path: Path to the PDF file to analyze
            output_path: Path to save the analysis results
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the LLM
        """
        self.agent_type = agent_type
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template for this agent type"""
        # For a real implementation, these should be stored in separate files
        # Here we're using the predefined templates from the agent_prompts dictionary
        return agent_prompts.get(self.agent_type, "")
        
    def _extract_pdf_content(self) -> List[Document]:
        """Extract content from the PDF file"""
        try:
            loader = PyPDFLoader(str(self.pdf_path))
            pages = loader.load()
            
            # For large PDFs, we might need to split into chunks
            if sum(len(page.page_content) for page in pages) > 100000:
                logger.info(f"PDF content is large, splitting into chunks")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=40000,
                    chunk_overlap=2000
                )
                return text_splitter.split_documents(pages)
            
            return pages
        except Exception as e:
            logger.error(f"Error extracting content from PDF: {e}")
            raise
            
    async def analyze(self) -> Path:
        """Analyze the PDF and save results"""
        logger.info(f"Starting analysis for {self.agent_type}")
        
        try:
            # Extract PDF content
            pages = self._extract_pdf_content()
            
            # Combine all pages for processing
            all_content = "\n".join([page.page_content for page in pages])
            
            # Create prompt
            prompt = PromptTemplate(
                input_variables=["pdf_content"],
                template=self.prompt_template
            )
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Run analysis
            result = chain.run(pdf_content=all_content)
            
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write output to file
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(result)
                
            logger.info(f"Completed analysis for {self.agent_type}")
            return self.output_path
        
        except Exception as e:
            logger.error(f"Error in {self.agent_type} analysis: {e}")
            # Create error report
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(f"# Error in {self.agent_type.capitalize()} Analysis\n\n")
                f.write(f"An error occurred during analysis: {str(e)}\n")
            return self.output_path


class SparkUISynthesizerAgent:
    """Agent for synthesizing analysis results from specialist agents"""
    
    def __init__(self, analysis_paths: Dict[str, Path], output_path: Path,
                 model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize the synthesizer agent.
        
        Args:
            analysis_paths: Paths to individual analysis files
            output_path: Path to save the synthesis results
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the LLM
        """
        self.analysis_paths = analysis_paths
        self.output_path = output_path
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.prompt_template = agent_prompts.get("synthesizer", "")
        
    async def synthesize(self) -> Path:
        """Synthesize individual analyses into a final report"""
        logger.info("Starting synthesis of analysis results")
        
        try:
            # Load all analysis content
            analysis_content = {}
            for agent_type, file_path in self.analysis_paths.items():
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        analysis_content[f"{agent_type}_analysis"] = f.read()
                else:
                    logger.warning(f"Analysis file for {agent_type} not found at {file_path}")
                    analysis_content[f"{agent_type}_analysis"] = f"# No {agent_type.capitalize()} Analysis Available"
            
            # Create prompt
            prompt = PromptTemplate(
                input_variables=list(analysis_content.keys()),
                template=self.prompt_template
            )
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Run synthesis
            result = chain.run(**analysis_content)
            
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write final report
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(result)
                
            logger.info(f"Completed synthesis report")
            return self.output_path
            
        except Exception as e:
            logger.error(f"Error in synthesis: {e}")
            # Create error report
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write("# Error in Synthesis\n\n")
                f.write(f"An error occurred during synthesis: {str(e)}\n")
            return self.output_path


class SparkAnalyzerOrchestrator:
    """Orchestrator for Spark UI analysis workflow"""
    
    def __init__(self, run_id: str, base_directory: str, 
                 model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize the orchestrator.
        
        Args:
            run_id: Identifier for the Spark run to analyze
            base_directory: Root directory for the analyzer system
            model_name: Name of the LLM model to use
            temperature: Temperature setting for the LLM
        """
        self.run_id = run_id
        self.base_directory = Path(base_directory)
        self.run_directory = self.base_directory / "runs" / f"spark_run_{run_id}"
        self.output_directory = self.base_directory / "outputs" / f"spark_run_{run_id}"
        self.model_name = model_name
        self.temperature = temperature
        self.log_file = self.base_directory / "logs" / f"spark_run_{run_id}_analysis.log"
        
        # Ensure directories exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file handler for this run
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
    def verify_input_files(self) -> bool:
        """Verify all required PDF files exist"""
        required_files = ["jobs.pdf", "stages.pdf", "storage.pdf", 
                         "environment.pdf", "executors.pdf", "sql.pdf"]
        
        missing_files = []
        for file in required_files:
            if not (self.run_directory / file).exists():
                missing_files.append(file)
                
        if missing_files:
            logger.error(f"Missing required PDF files: {', '.join(missing_files)}")
            raise FileNotFoundError(f"Missing required PDF files: {', '.join(missing_files)}")
        
        logger.info("All required PDF files verified")
        return True
        
    async def run_analyzer_agent(self, agent_type: str) -> Path:
        """Run a specific analyzer agent on its PDF"""
        pdf_path = self.run_directory / f"{agent_type}.pdf"
        output_path = self.output_directory / f"{agent_type}_analysis.md"
        
        agent = SparkUIAnalyzerAgent(
            agent_type=agent_type,
            pdf_path=pdf_path,
            output_path=output_path,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        return await agent.analyze()
    
    async def run_synthesizer(self, analysis_paths: Dict[str, Path]) -> Path:
        """Run the synthesizer agent to create final report"""
        output_path = self.output_directory / "final_report.md"
        
        synthesizer = SparkUISynthesizerAgent(
            analysis_paths=analysis_paths,
            output_path=output_path,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        return await synthesizer.synthesize()
        
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting full analysis for run {self.run_id}")
            
            # Verify input files
            self.verify_input_files()
            
            # Run all analyzer agents in parallel
            agent_types = ["jobs", "stages", "storage", "environment", "executors", "sql"]
            tasks = [self.run_analyzer_agent(agent_type) for agent_type in agent_types]
            analysis_results = await asyncio.gather(*tasks)
            
            # Map agent types to their output paths
            analysis_paths = {
                agent_type: output_path
                for agent_type, output_path in zip(agent_types, analysis_results)
            }
            
            # Run synthesizer
            final_report_path = await self.run_synthesizer(analysis_paths)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Create status report
            status = {
                "run_id": self.run_id,
                "agents_triggered": agent_types,
                "agents_completed": agent_types,
                "synthesizer_status": "completed",
                "final_report_path": str(final_report_path),
                "execution_time_seconds": round(execution_time, 2)
            }
            
            # Save status report
            status_path = self.output_directory / "status.json"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
                
            logger.info(f"Analysis complete in {execution_time:.2f} seconds")
            
            return status
            
        except Exception as e:
            logger.error(f"Error in full analysis: {e}")
            
            # Create error report
            error_report_path = self.output_directory / "error_report.md"
            with open(error_report_path, "w") as f:
                f.write("# Analysis Error Report\n\n")
                f.write(f"An error occurred during analysis: {str(e)}\n")
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            status = {
                "run_id": self.run_id,
                "error": str(e),
                "error_report_path": str(error_report_path),
                "execution_time_seconds": round(execution_time, 2)
            }
            
            # Save status report
            status_path = self.output_directory / "status.json"
            with open(status_path, "w") as f:
                json.dump(status, f, indent=2)
                
            return status


# Agent prompt templates
agent_prompts = {
    "jobs": """
You are the Jobs Analyzer Agent specialized in reviewing Apache Spark Jobs tab information. Your task is to extract key metrics from the Jobs tab PDF and identify performance bottlenecks.

Input:
- PDF content from the Jobs tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Total number of jobs
   - Job completion status (successful vs. failed jobs)
   - Job durations (min, max, average)
   - Jobs with exceptionally long duration
   - Jobs with multiple retry attempts
   - Distribution of stages per job
   - Timeline patterns (e.g., sequential vs. parallel job execution)
   - Any errors or warnings reported for jobs

2. Analyze for these common issues:
   - Jobs with unusually long duration compared to others
   - Failed jobs and their error patterns
   - Jobs with a disproportionately high number of stages
   - Sequential job execution that could be parallelized
   - Jobs that were resubmitted multiple times
   - Stages within jobs that account for the majority of execution time

3. For each identified issue, provide:
   - The specific job ID and description
   - Relevant metrics showing the problem
   - Potential root causes
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Key metrics about all jobs
- Critical Issues: High-priority bottlenecks demanding immediate attention
- Performance Observations: Notable patterns that might impact performance
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Only make claims that are directly supported by the data in the PDF. Avoid speculation beyond what the data shows.

PDF Content:
{pdf_content}
""",
    
    "stages": """
You are the Stages Analyzer Agent specialized in reviewing Apache Spark Stages tab information. Your task is to extract key metrics from the Stages tab PDF and identify performance bottlenecks at the stage level.

Input:
- PDF content from the Stages tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Total number of stages
   - Stage completion status (successful vs. failed stages)
   - Stage durations (min, median, max)
   - Number of tasks per stage
   - Task metrics including:
     - Task duration (min, median, max)
     - Shuffle read/write sizes
     - Executor run time
     - JVM GC time
     - Serialization/deserialization time
     - Memory spill (disk/memory)
   - Input/output data sizes

2. Calculate and analyze for data skew indicators:
   - Task duration variance within stages
   - Shuffle read/write size variance across tasks
   - Peak-to-median ratio for task durations

3. Identify these common stage-level issues:
   - Stages with significant data skew (max task duration > 2x median)
   - Stages with excessive shuffle operations
   - Stages with high GC overhead (GC time > 20% of execution time)
   - Stages with large data spills to disk
   - Stages with serialization bottlenecks
   - Stages with too few or too many tasks
   - Sequential stages that could be parallelized

4. For each identified issue, provide:
   - The specific stage ID and description
   - Relevant metrics showing the problem
   - Potential root causes
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Key metrics about all stages
- Critical Issues: High-priority bottlenecks demanding immediate attention
- Data Skew Analysis: Evidence of unbalanced data distribution
- Shuffle Operation Analysis: Assessment of data movement operations
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Only make claims that are directly supported by the data in the PDF. Focus particularly on quantifying data skew since this is a common Spark performance killer.

PDF Content:
{pdf_content}
""",
    
    "storage": """
You are the Storage Analyzer Agent specialized in reviewing Apache Spark Storage tab information. Your task is to extract key metrics from the Storage tab PDF and identify storage-related performance issues.

Input:
- PDF content from the Storage tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - RDDs/DataFrames being cached
   - Storage level for each cached item (MEMORY_ONLY, MEMORY_AND_DISK, etc.)
   - Memory usage per RDD/DataFrame
   - Disk usage per RDD/DataFrame (if applicable)
   - Partition counts for cached data
   - Cache hit/miss ratios (if available)
   - Fraction of cached data (vs. total size)
   - Replication factors

2. Analyze for these common storage-related issues:
   - Inefficient storage levels for workload patterns
   - Excessive caching that could lead to memory pressure
   - Under-utilization of cache (low cache hit ratio)
   - Uneven partition sizes in cached data
   - Large objects being spilled to disk
   - Objects that are cached but never reused
   - Missing cache opportunities for frequently reused data

3. For each identified issue, provide:
   - The specific RDD/DataFrame ID and description
   - Storage level and size metrics
   - Why this represents a potential problem
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Overview of cache usage
- Memory Usage Analysis: Breakdown of memory utilization
- Storage Efficiency Issues: Identified problems with caching strategy
- Partition Distribution: Analysis of partition sizes and counts
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Focus on whether the caching strategy is appropriate for the workload and identify opportunities to optimize memory usage while maintaining performance.

PDF Content:
{pdf_content}
""",
    
    "environment": """
You are the Environment Analyzer Agent specialized in reviewing Apache Spark Environment tab information. Your task is to extract configuration settings from the Environment tab PDF and identify potentially suboptimal configurations.

Input:
- PDF content from the Environment tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Spark version
   - Java/Scala version
   - System specifications (cores, memory)
   - All Spark configuration parameters, with special focus on:
     - spark.executor.memory
     - spark.executor.cores
     - spark.executor.instances
     - spark.driver.memory
     - spark.driver.cores
     - spark.default.parallelism
     - spark.sql.shuffle.partitions
     - spark.memory.fraction
     - spark.memory.storageFraction
     - spark.locality.wait
     - spark.speculation
     - spark.serializer
     - spark.io.compression.codec
     - spark.rdd.compress
     - spark.shuffle.compress
     - spark.sql.adaptive.enabled
     - spark.dynamicAllocation settings

2. Analyze for these common configuration issues:
   - Suboptimal memory allocation (too high/low)
   - Inefficient parallelism settings
   - Memory fraction imbalances
   - Missing performance-critical configurations
   - Inefficient serialization or compression settings
   - Disabled features that could improve performance
   - Configurations that don't align with cluster resources
   - Outdated Spark version with known performance issues

3. For each identified issue, provide:
   - The specific configuration parameter
   - Current setting and recommended setting
   - Rationale for the recommendation
   - Expected performance impact of the change

Output your analysis as Markdown with the following sections:
- Environment Summary: Key version and system information
- Critical Configuration Issues: High-priority settings that need adjustment
- Memory Configuration Analysis: Assessment of memory-related settings
- CPU/Parallelism Analysis: Assessment of CPU utilization and parallelism settings
- I/O and Serialization Analysis: Assessment of data processing efficiency settings
- Recommended Configuration Changes: Complete list of suggested modifications
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Provide concrete configuration recommendations with numerical values where appropriate, explaining the reasoning behind each suggestion.

PDF Content:
{pdf_content}
""",
    
    "executors": """
You are the Executors Analyzer Agent specialized in reviewing Apache Spark Executors tab information. Your task is to extract key metrics from the Executors tab PDF and identify executor-related performance issues.

Input:
- PDF content from the Executors tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - Total number of executors
   - Executor host distribution
   - Per-executor metrics including:
     - Active tasks
     - Failed tasks
     - Complete tasks
     - Total cores
     - Memory usage (peak and current)
     - On-heap/off-heap memory usage
     - Disk usage
     - Duration
     - GC time
     - Input/output
     - Shuffle read/write
   - Driver metrics (treated as a special executor)

2. Analyze for these common executor-related issues:
   - Uneven task distribution across executors
   - Executors with excessive GC time (GC time > 15% of executor uptime)
   - Memory pressure indicators (high memory usage, frequent GC)
   - Executors with unusually high failure rates
   - Resource utilization imbalances
   - Executors with significantly higher/lower throughput
   - Network I/O bottlenecks
   - Disk I/O bottlenecks
   - Driver as a bottleneck (overloaded driver)

3. For each identified issue, provide:
   - The specific executor ID or host
   - Relevant metrics showing the problem
   - Comparisons with other executors to highlight imbalances
   - Specific recommendations to address the issue

Output your analysis as Markdown with the following sections:
- Summary Statistics: Overview of executor fleet
- Resource Utilization Analysis: CPU, memory, and disk usage patterns
- GC Analysis: Garbage collection patterns and issues
- Task Distribution Analysis: Assessment of work distribution
- Critical Executor Issues: Specific executors with problems
- Driver Analysis: Assessment of driver performance
- Recommendations: Specific, actionable suggestions for improvement
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Pay particular attention to resource utilization patterns and identify opportunities for better work distribution or resource allocation.

PDF Content:
{pdf_content}
""",
    
    "sql": """
You are the SQL Analyzer Agent specialized in reviewing Apache Spark SQL/DataFrame tab information. Your task is to extract key metrics from the SQL tab PDF and identify query performance issues.

Input:
- PDF content from the SQL/DataFrame tab of Spark UI

Instructions:
1. Extract the following key information from the PDF:
   - List of executed SQL queries/DataFrame operations
   - Query execution times
   - Query plans showing:
     - Scan operations (table scans, file formats)
     - Join types and conditions
     - Filter operations
     - Aggregation operations
     - Sort operations
     - Exchange (shuffle) operations
     - Project operations
   - Statistics on rows processed at each stage
   - Predicate pushdown information
   - Partition pruning information

2. Analyze for these common SQL performance issues:
   - Cartesian products or expensive joins
   - Missing filter pushdown opportunities
   - Missing partition pruning opportunities
   - Excessive shuffling in query plans
   - Poorly optimized aggregations
   - Inefficient scan patterns
   - Suboptimal join ordering
   - Broadcast join opportunities being missed
   - Sort operations on large datasets
   - Skew in join or aggregation keys
   - Window functions with large partitions
   - Complex/inefficient expressions

3. For each identified issue, provide:
   - The specific query or operation ID
   - The problematic portion of the query plan
   - Why this represents a performance issue
   - SQL or API-level recommendations to improve the query

Output your analysis as Markdown with the following sections:
- Query Overview: Summary of analyzed queries
- Critical Query Performance Issues: High-priority SQL bottlenecks
- Join Operation Analysis: Assessment of join efficiency
- Data Reading Patterns: Analysis of scan operations
- Aggregation & Grouping Analysis: Assessment of data aggregation patterns
- Shuffle Analysis: Impact of data exchange operations
- Query Optimization Recommendations: Specific suggestions for each problematic query
- Limitations: Any information you couldn't extract or analyze from the PDF

Be precise and factual. Where possible, provide specific SQL rewrites or DataFrame API alternatives that would improve performance.

PDF Content:
{pdf_content}
""",
    
    "synthesizer": """
You are the Synthesizer Agent responsible for integrating the findings from all specialist analyzer agents and producing a comprehensive Spark performance analysis report.

Input:
- Analysis reports from all specialist agents (Jobs, Stages, Storage, Environment, Executors, SQL)

Instructions:
1. Read and assimilate the analyses from all specialist agents.

2. Identify cross-cutting performance issues that span multiple areas, such as:
   - Correlation between configuration settings and observed bottlenecks
   - Connection between SQL query patterns and stage/task performance
   - Relationship between storage decisions and executor memory pressure
   - Impact of join strategies on shuffle operations and network I/O
   - Connections between GC pressure and memory configuration
   - How data skew in stages relates to join/groupBy operations in SQL

3. Prioritize identified issues based on:
   - Severity (impact on overall performance)
   - Frequency of occurrence
   - Difficulty of resolution
   - Potential performance improvement

4. Formulate a holistic set of recommendations addressing:
   - Quick wins (simple configuration changes)
   - Medium-effort improvements (query/job restructuring)
   - Strategic changes (data layout, application architecture)

5. Create a comprehensive executive summary that highlights:
   - The most critical bottlenecks
   - Root causes spanning multiple areas
   - Expected performance improvements from recommendations

Output your analysis as a well-structured Markdown report with the following sections:
- Executive Summary: Brief overview of key findings
- Critical Performance Bottlenecks: Prioritized list of the most significant issues
- Cross-Cutting Analysis: Performance issues that span multiple areas
- Root Cause Analysis: Deeper investigation into underlying causes
- Optimization Recommendations:
  - Configuration Changes
  - Application Code Improvements
  - Data Management Strategies
  - Resource Allocation Adjustments
- Implementation Plan: Suggested order for implementing changes
- Expected Outcomes: Projected performance improvements
- Further Investigation: Areas where more data would be beneficial

Ensure recommendations are specific, actionable, and directly tied to observed issues in the data. Prioritize recommendations based on expected impact and implementation difficulty.

Jobs Analysis:
{jobs_analysis}

Stages Analysis:
{stages_analysis}

Storage Analysis:
{storage_analysis}

Environment Analysis:
{environment_analysis}

Executors Analysis:
{executors_analysis}

SQL Analysis:
{sql_analysis}
"""
}


async def main():
    """Main function to run the Spark UI analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spark UI Performance Analyzer")
    parser.add_argument("--run-id", required=True, help="Identifier for the Spark run to analyze")
    parser.add_argument("--base-dir", required=True, help="Base directory for the analyzer system")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature setting for the LLM")
    
    args = parser.parse_args()
    
    # Create and run the orchestrator
    orchestrator = SparkAnalyzerOrchestrator(
        run_id=args.run_id,
        base_directory=args.base_dir,
        model_name=args.model,
        temperature=args.temperature
    )
    
    status = await orchestrator.run_full_analysis()
    
    print(json.dumps(status, indent=2))
    
    if "error" in status:
        return 1
    return 0


if __name__ == "__main__":
    asyncio.run(main())
