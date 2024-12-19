# In Python Page 228 of E-book
from __future__ import print_function

#importing functions
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("Demo Spark Python Cluster Program").getOrCreate()

#reading 2000-2018 file
df1 = spark.read.text("hdfs://namenode/user/controller/ncdc-orig/2000-2018.txt")
 
#creating a dataframe
df2 = df1.withColumn('Weather_Station', df1['value'].substr(5, 6))\
.withColumn('WBAN', df1['value'].substr(11, 5))\
.withColumn('Observation_Date',to_date(df1['value'].substr(16,8),"yyyyMMdd"))\
.withColumn('Observation_Hour', df1['value'].substr(24, 4).cast(IntegerType()))\
.withColumn('Latitude', df1['value'].substr(29, 6).cast('float') / 1000)\
.withColumn('Longitude', df1['value'].substr(35, 7).cast('float') / 1000)\
.withColumn('Elevation', df1['value'].substr(47, 5).cast(IntegerType()))\
.withColumn('Wind_Direction', df1['value'].substr(61, 3).cast(IntegerType()))\
.withColumn('WD_Quality_Code', df1['value'].substr(64, 1).cast(IntegerType()))\
.withColumn('Sky_Ceiling_Height', df1['value'].substr(71, 5).cast(IntegerType()))\
.withColumn('SC_Quality_Code', df1['value'].substr(76, 1).cast(IntegerType()))\
.withColumn('Visibility_Distance', df1['value'].substr(79, 6).cast(IntegerType()))\
.withColumn('VD_Quality_Code', df1['value'].substr(86, 1).cast(IntegerType()))\
.withColumn('Air_Temperature', df1['value'].substr(88, 5).cast('float') /10)\
.withColumn('AT_Quality_Code', df1['value'].substr(93, 1).cast(IntegerType()))\
.withColumn('Dew_Point', df1['value'].substr(94, 5).cast('float'))\
.withColumn('DP_Quality_Code', df1['value'].substr(99, 1).cast(IntegerType()))\
.withColumn('Atmospheric_Pressure', df1['value'].substr(100, 5).cast('float')/ 10)\
.withColumn('AP_Quality_Code', df1['value'].substr(105, 1).cast(IntegerType()))

df2.show(10)

#Filtering data for 5 years 2000 to 2005
df_5years=df2.filter(df2["Observation_Date"].between('2000-01-01','2005-12-31'))

#writing a csv file
#Chapter 05- Pg-86 of E-book -Repartition and Coalesce
#Chapter 09 Pg-164 of E-book - CSV Options -compression or codec and Apache Spark Documentation

df_5years.repartition(50).write.format("csv").option("compression","lz4").mode("overwrite").option("header","true").save("hdfs://namenode/output/itmd-521/drp/2000-2005/repartition/50")