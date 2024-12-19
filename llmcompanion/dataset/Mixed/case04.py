from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerDataProcessing").master("local[*]").getOrCreate()

# 1. Using RDD Instead of DataFrame/Dataset
# Assume we have customer data in a text file, with each line being "customer_id,name,age,city"
customer_rdd = spark.sparkContext.textFile("path/to/customers.txt")

# Parsing the RDD data into structured format (customer_id, name, age, city)
parsed_customer_rdd = customer_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
adult_customers_rdd = parsed_customer_rdd.filter(lambda cust: int(cust[2]) >= 18)  # Filtering adult customers
print("Sample adult customers:", adult_customers_rdd.take(5))

# 5. Using UDF Instead of Built-In Functions
# Converting RDD to DataFrame for further processing
customer_df = adult_customers_rdd.toDF(["customer_id", "name", "age", "city"])

# Defining a UDF to create a greeting message for each customer
def create_greeting(name):
    return f"Hello, {name}!"

greeting_udf = udf(create_greeting, StringType())

# Adding a greeting column using the UDF
customer_with_greeting_df = customer_df.withColumn("greeting", greeting_udf(customer_df["name"]))
customer_with_greeting_df.show()

# Stop Spark session
spark.stop()