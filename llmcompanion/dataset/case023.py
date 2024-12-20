from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerOrderProcessing").master("local[*]").getOrCreate()

# 1. Using RDD Instead of DataFrame/Dataset
# Assume we have order data in a text file, with each line being "order_id,customer_id,amount,category"
orders_rdd = spark.sparkContext.textFile("path/to/orders.txt")

# Parsing the RDD into structured format (order_id, customer_id, amount, category)
parsed_orders_rdd = orders_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
electronics_orders_rdd = parsed_orders_rdd.filter(lambda order: order[3] == "Electronics")  # Filtering for "Electronics"
print("Sample Electronics orders:", electronics_orders_rdd.take(5))

# 3. Using map() Instead of mapPartitions()
# Applying a transformation to calculate tax (10%) on each order amount
taxed_orders_rdd = electronics_orders_rdd.map(lambda order: (order[0], order[1], float(order[2]) * 1.1, order[3]))  # Inefficient element-wise processing
print("Sample taxed orders:", taxed_orders_rdd.take(5))

# Convert the taxed orders RDD to a DataFrame
taxed_orders_df = taxed_orders_rdd.toDF(["order_id", "customer_id", "amount_with_tax", "category"])

# 4. Using Non-Optimized Data Format (CSV)
# Saving the taxed orders in CSV format instead of an efficient format like Parquet
taxed_orders_df.write.format("csv").option("header", "true").save("path/to/taxed_orders_output.csv")

# Stop Spark session
spark.stop()