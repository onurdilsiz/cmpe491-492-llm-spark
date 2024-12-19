from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("SalesDataProcessing").master("local[*]").getOrCreate()

# Assume we have sales data in a structured format
data = [
    (1, "Electronics", 1200.50),
    (2, "Clothing", 80.00),
    (3, "Electronics", 150.75),
    (4, "Furniture", 320.00),
    (5, "Electronics", 950.25),
]
columns = ["sale_id", "category", "amount"]

# Create a DataFrame
sales_df = spark.createDataFrame(data, columns)

# 2. Using repartition() Instead of coalesce()
# Repartitioning the data into a larger number of partitions inefficiently
repartitioned_df = sales_df.repartition(10)  # Unnecessary shuffle when reducing partitions
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# 3. Using map() Instead of mapPartitions()
# Assume we want to calculate a 10% discount for each sale
sales_rdd = repartitioned_df.rdd
discounted_sales_rdd = sales_rdd.map(lambda row: (row["sale_id"], row["category"], row["amount"] * 0.9))  # Inefficient element-wise processing

# Collect and display discounted sales
print("Sample discounted sales:", discounted_sales_rdd.take(5))

# Stop Spark session
spark.stop()