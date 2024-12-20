from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerTransactionProcessing").master("local[*]").getOrCreate()

# 1. Using RDD Instead of DataFrame/Dataset
# Assume we have customer transaction data in a text file, with each line being "transaction_id,customer_id,amount,category"
transactions_rdd = spark.sparkContext.textFile("path/to/transactions.txt")

# Parsing the RDD into structured format (transaction_id, customer_id, amount, category)
parsed_transactions_rdd = transactions_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
electronics_transactions_rdd = parsed_transactions_rdd.filter(lambda txn: txn[3] == "Electronics")  # Filtering for "Electronics"
print("Sample filtered transactions:", electronics_transactions_rdd.take(5))

# Converting the filtered RDD to a DataFrame
transactions_df = electronics_transactions_rdd.toDF(["transaction_id", "customer_id", "amount", "category"])

# 2. Using repartition() Instead of coalesce()
# Repartitioning the DataFrame unnecessarily, causing a full shuffle
repartitioned_df = transactions_df.repartition(10)  # Unnecessary repartition
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# 5. Using UDF Instead of Built-In Functions
# Define a UDF to create a custom message for each transaction
def generate_message(category, amount):
    return f"Category: {category}, Amount: ${amount}"

message_udf = udf(generate_message, StringType())

# Applying the UDF to create a new column with messages
transactions_with_message_df = repartitioned_df.withColumn("transaction_message", message_udf(repartitioned_df["category"], repartitioned_df["amount"]))
transactions_with_message_df.show()

# Stop Spark session
spark.stop()