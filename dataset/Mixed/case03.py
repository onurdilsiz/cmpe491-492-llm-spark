from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("EmployeeDataProcessing").master("local[*]").getOrCreate()

# 1. Using RDD Instead of DataFrame/Dataset
# Assume we have employee data in a text file, with each line being "employee_id,name,department,salary"
employee_rdd = spark.sparkContext.textFile("path/to/employees.txt")

# Parsing the RDD data into structured format (employee_id, name, department, salary)
parsed_employee_rdd = employee_rdd.map(lambda line: line.split(","))  # Inefficient RDD processing
department_salary_rdd = parsed_employee_rdd.map(lambda emp: (emp[2], float(emp[3])))  # Extracting department and salary
print("Sample department-salary data:", department_salary_rdd.take(5))

# 4. Using Non-Optimized Data Format (CSV)
# Convert the RDD to a DataFrame and write it to CSV
department_salary_df = department_salary_rdd.toDF(["department", "salary"])
department_salary_df.write.format("csv").option("header", "true").save("path/to/department_salary_output.csv")

# Stop Spark session
spark.stop()