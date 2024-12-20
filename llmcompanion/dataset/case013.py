from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("RetailTransactionProcessing").master("local[*]").getOrCreate()

# 1. Using RDD Instead of DataFrame/Dataset
# Assume we have transaction data in a text file, with each line being "transaction_id,customer_id,amount,category"
transactions_rdd = spark.sparkContext.textFile("path/to/transactions.txt")

# Parsing the RDD into structured format (transaction_id, customer_id, amount, category)
parsed_transactions_rdd = transactions_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
filtered_transactions_rdd = parsed_transactions_rdd.filter(lambda txn: txn[3] == "Electronics")  # Filtering for "Electronics"
print("Sample filtered transactions:", filtered_transactions_rdd.take(5))

# Converting the filtered RDD to a DataFrame for further processing
filtered_transactions_df = filtered_transactions_rdd.toDF(["transaction_id", "customer_id", "amount", "category"])

# 2. Using repartition() Instead of coalesce()
# Repartitioning the DataFrame unnecessarily, causing a full shuffle
repartitioned_df = filtered_transactions_df.repartition(10)  # Inefficient partitioning
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# 4. Using Non-Optimized Data Format (CSV)
# Saving the filtered transactions in CSV format instead of a more efficient format like Parquet
repartitioned_df.write.format("csv").option("header", "true").save("path/to/electronics_transactions_output.csv")

# Stop Spark session
spark.stop()