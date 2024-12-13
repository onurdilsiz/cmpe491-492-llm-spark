from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BadPracticesExample_1_2_3").master("local[*]").getOrCreate()

# 1. Using RDD instead of DataFrame/Dataset
# Assume we have transaction data in a text file, with each line being "transaction_id,amount,currency"
transaction_rdd = spark.sparkContext.textFile("path/to/transactions.txt")

# Parsing the RDD data into structured format (transaction_id, amount, currency)
parsed_transaction_rdd = transaction_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
usd_transactions_rdd = parsed_transaction_rdd.filter(lambda txn: txn[2] == "USD")  # Filtering for USD transactions
usd_transaction_count = usd_transactions_rdd.count()
print(f"Number of USD transactions: {usd_transaction_count}")

# 2. Using repartition() Instead of coalesce()
# Assume we want to process the filtered transactions into a DataFrame and reduce partitions for writing
usd_transactions_df = usd_transactions_rdd.toDF(["transaction_id", "amount", "currency"])
repartitioned_df = usd_transactions_df.repartition(10)  # Inefficiently increasing partitions
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# 3. Using map() Instead of mapPartitions()
# Applying a transformation to convert transaction amounts from string to float
amounts_rdd = usd_transactions_rdd.map(lambda txn: float(txn[1]))  # Inefficient per-element processing
print("Sample transaction amounts:", amounts_rdd.take(5))

# Stop Spark session
spark.stop()