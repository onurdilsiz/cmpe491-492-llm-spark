from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("RealisticBadPracticesExample").master("local[*]").getOrCreate()

# 1. Using RDD instead of DataFrame/Dataset
# Assume we have log data in a text file, with each line being "timestamp,level,message"
log_rdd = spark.sparkContext.textFile("path/to/logs.txt")

# Parsing the RDD data into structured format (timestamp, level, message)
parsed_logs_rdd = log_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
error_logs_rdd = parsed_logs_rdd.filter(lambda log: log[1] == "ERROR")
error_count = error_logs_rdd.count()
print(f"Number of ERROR logs: {error_count}")

# 2. Using repartition() instead of coalesce()
# Assume the parsed logs will be written to a CSV file for external consumption
error_logs_df = error_logs_rdd.toDF(["timestamp", "level", "message"])
repartitioned_df = error_logs_df.repartition(10)  # Inefficiently increases partitions
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# 3. Using map() instead of mapPartitions()
# Applying a transformation to extract only the timestamp from each error log
timestamps_rdd = error_logs_rdd.map(lambda log: log[0])  # Inefficient per-element processing
print("Sample timestamps from error logs:", timestamps_rdd.take(5))

# 4. Using non-optimized data format (CSV)
# Writing the error logs to CSV format instead of Parquet/ORC
repartitioned_df.write.format("csv").option("header", "true").save("path/to/error_logs_output.csv")

# Stop Spark session
spark.stop()