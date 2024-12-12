# Databricks notebook source
# MAGIC %md
# MAGIC UDF means User Defined Functions  
# MAGIC Functions that we can create and use them later

# COMMAND ----------

# MAGIC %run "/yt-course-india/00-utils"

# COMMAND ----------

df = create_standard_df()

# COMMAND ----------

from pyspark.sql.functions import udf, when
from pyspark.sql.types import DoubleType

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.withColumn(
  'bonus', 
  when(df.gender == 'f', 1000).otherwise(500)
)

# COMMAND ----------

# Creating function
def new_salary(salary, bonus):
  return salary  + bonus


# Registering the function
total_salary = udf(lambda s, b: new_salary(s, b), DoubleType())

# COMMAND ----------

display(
  df.withColumn('new_salary', total_salary(df.salary, df.bonus))
)

# COMMAND ----------

# Another way to register a function
@udf(returnType=DoubleType())
def new_salary(salary, bonus):
  return salary  + bonus

# COMMAND ----------

display(
  df.select(
    '*',
    new_salary(df.salary, df.bonus).alias('total_salary')
  )
)

# COMMAND ----------

# Register udf function to use with spark sql

spark.udf.register(name='total_salary', f=new_salary)

df.createOrReplaceTempView('emps')

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *, total_salary(salary, bonus) AS salary_total
# MAGIC FROM emps