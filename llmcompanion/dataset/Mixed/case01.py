from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BadPracticesExample_1_2").master("local[*]").getOrCreate()

# 1. Using RDD instead of DataFrame/Dataset
# Assume we have sales data in a text file, with each line being "sale_id,amount,category"
sales_rdd = spark.sparkContext.textFile("path/to/sales.txt")

# Parsing the RDD data into structured format (sale_id, amount, category)
parsed_sales_rdd = sales_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
electronics_sales_rdd = parsed_sales_rdd.filter(lambda sale: sale[2] == "Electronics")  # Filtering for "Electronics" category
electronics_sales_count = electronics_sales_rdd.count()
print(f"Number of Electronics sales: {electronics_sales_count}")

# 2. Using repartition() Instead of coalesce()
# Convert the filtered RDD to a DataFrame
electronics_sales_df = electronics_sales_rdd.toDF(["sale_id", "amount", "category"])

# Inefficiently increasing partitions before writing to an output
repartitioned_df = electronics_sales_df.repartition(10)  # Unnecessary shuffling across nodes
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# Stop Spark session
spark.stop()