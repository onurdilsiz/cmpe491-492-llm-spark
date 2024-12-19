from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BadPracticesExample_1_3").master("local[*]").getOrCreate()

# 1. Using RDD Instead of DataFrame/Dataset
# Assume we have employee data in a text file, with each line being "employee_id,name,salary"
employee_rdd = spark.sparkContext.textFile("path/to/employees.txt")

# Parsing the RDD data into structured format (employee_id, name, salary)
parsed_employee_rdd = employee_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
high_salary_rdd = parsed_employee_rdd.filter(lambda emp: float(emp[2]) > 50000)  # Filtering for high salaries
high_salary_count = high_salary_rdd.count()
print(f"Number of employees with high salary: {high_salary_count}")

# 3. Using map() Instead of mapPartitions()
# Applying a transformation to calculate bonuses for high-salary employees
bonus_rdd = high_salary_rdd.map(lambda emp: (emp[0], emp[1], float(emp[2]) * 1.1))  # Inefficient per-element processing
print("Sample employees with bonuses:", bonus_rdd.take(5))

# Stop Spark session
spark.stop()