import findspark
findspark.init()
import pyspark
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, DoubleType
from pyspark.sql.functions import udf
import os
os.environ["HADOOP_HOME"]= 'C:\\Travaux_2012\\Anaconda e Python\\hadoop-2.8.1'

#Spark Ui http://localhost:4040
spark = SparkSession\
    .builder\
    .appName("PySpark XGBOOST Native")\
    .getOrCreate()


print(f"Versione Pyspark = {spark.version}")



dati = [
    (1, [0, 0, 35, 48, 85, 68, 70, 80, 93, 100, 110, 120]),
    (2, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115]),
    (3, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200])
]

df = spark.createDataFrame(dati, ["chiave", "array_col"])

# per afare array f_merged = df.fillna(0, subset=['margine_0', 'margine_1]).withColumn('array_col', F.array('margine_0', 'margine_1'))
df.show(truncate=False)

@udf(ArrayType(DoubleType()))
def delta_mol(array):
    differenze = [(array[i+1] / array[i] -1)*100 if array[i] != 0.0 else 0.0 for i in range(len(array)-1) ]
    return differenze
 
dff = df.withColumn('delta_arry',delta_mol(F.col('array_col')))
dff.show(truncate=False)

