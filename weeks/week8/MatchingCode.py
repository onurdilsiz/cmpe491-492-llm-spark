import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, max

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a SparkSession
spark = SparkSession.builder.appName("MatchingCode").getOrCreate()

path_tasks = "data/DatasetTasks.csv"
path_versions = "data/DatasetVersions.csv"

try:
    logging.info("Reading DatasetTasks DataFrame")
    datasetTasks = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(path_tasks)

    logging.info("Reading DatasetVersions DataFrame")
    datasetVersions = spark.read.csv(path_versions, header=True, inferSchema=True)

    # Add operations on datasetTasks (example aggregations)
    logging.info("Performing aggregations on datasetTasks")
    task_aggregations = datasetTasks.groupBy("datasetId").agg(
        avg("taskId").alias("avg_taskId"),
        max("taskId").alias("max_taskId")
    )
    task_aggregations.show()

    logging.info("Joining dataset versions and dataset tasks")
    joined_df = datasetVersions.join(datasetTasks, datasetVersions["datasetId"] == datasetTasks["datasetId"], how="inner")

    logging.info("Joined DataFrame (first 10 rows):")
    joined_df.show(10, truncate=False)

    logging.info("\nExecution Plan:")
    joined_df.explain()

    logging.info("Joining dataset versions and dataset tasks without BHJ")
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)  # Disable BroadcastHashJoin

    joined_df_without_BHJ = datasetVersions.join(datasetTasks, datasetVersions["datasetId"] == datasetTasks["datasetId"], how="inner")

    logging.info("\nJoined DataFrame without BHJ (first 10 rows):")
    joined_df_without_BHJ.show(10, truncate=False)

    logging.info("\nExecution Plan without BHJ:")
    joined_df_without_BHJ.explain()

except Exception as e:
    logging.error(f"An error occurred: {e}")

finally:
    spark.stop()

