from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("PracticesExample").master("local[*]").getOrCreate()

# 1. Using RDD instead of DataFrame/Dataset
rdd = spark.sparkContext.parallelize(["1,John", "2,Jane", "3,Doe"])
rdd_result = rdd.map(lambda line: (int(line.split(",")[0]), line.split(",")[1].upper())).collect()
print("RDD Result:", rdd_result)

# 2. Using repartition() instead of coalesce()
df = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["numbers"])
repartitioned_df = df.repartition(10)  # Inefficient repartitioning
print("Number of partitions after repartition:", repartitioned_df.rdd.getNumPartitions())

# 3. Using map() instead of mapPartitions()
mapped_rdd = rdd.map(lambda line: int(line.split(",")[0]) * 2)  # Processes each element individually
print("Mapped RDD Result:", mapped_rdd.collect())

# 4. Using non-optimized data format (CSV)
csv_df = spark.read.format("csv").option("header", "true").load("path/to/data.csv")
csv_result = csv_df.select("column1").collect()
print("CSV Result:", csv_result)

# 5. Using UDF instead of built-in Spark SQL functions
def multiply_by_two(x):
    return x * 2

multiply_udf = udf(multiply_by_two, IntegerType())
result_with_udf = df.withColumn("doubled", multiply_udf(col("numbers")))
result_with_udf.show()

# Stop Spark session
spark.stop()