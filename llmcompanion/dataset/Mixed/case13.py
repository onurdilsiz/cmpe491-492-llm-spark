from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ProductSalesProcessing").master("local[*]").getOrCreate()

# Assume we have product sales data in a structured format
data = [
    (1, "Laptop", 1200.50, "Electronics"),
    (2, "T-shirt", 20.00, "Clothing"),
    (3, "Desk", 150.75, "Furniture"),
    (4, "Headphones", 320.00, "Electronics"),
    (5, "Shoes", 50.25, "Clothing"),
]
columns = ["product_id", "product_name", "price", "category"]

# Create a DataFrame
sales_df = spark.createDataFrame(data, columns)

# 2. Using repartition() Instead of coalesce()
# Repartitioning the DataFrame unnecessarily, causing a full shuffle
repartitioned_df = sales_df.repartition(10)  # Inefficient partitioning
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# Perform some filtering
electronics_df = repartitioned_df.filter(repartitioned_df["category"] == "Electronics")

# 4. Using Non-Optimized Data Format (CSV)
# Save the filtered data in CSV format instead of a more efficient format like Parquet
electronics_df.write.format("csv").option("header", "true").save("path/to/electronics_sales_output.csv")

# Stop Spark session
spark.stop()