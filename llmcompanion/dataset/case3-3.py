#%%
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
#%%
from pyspark.ml.feature import Imputer
# %%
spark = SparkSession.builder.appName("Practice").getOrCreate()
# %%
## Reading the dataset
df_pyspark = spark.read.csv("test2.csv", header=True, inferSchema=True)
df_pyspark.show()
# %%
## Drop rows with any null values
df_pyspark = df_pyspark.na.drop()
df_pyspark.show()
#%%
## Drop rows with any null values
df_pyspark = df_pyspark.na.drop(how="any")
df_pyspark.show()
# %%
## Drop rows with all null values
df_pyspark = df_pyspark.na.drop(how="all")
df_pyspark.show()
# %%
## Drop rows where there are at > 2 null values
df_pyspark = df_pyspark.na.drop(thresh=2)
df_pyspark.show()
#%%
## Drop rows with subset
df_pyspark = df_pyspark.na.drop(subset=["Experience"])
df_pyspark.show()
# %%
## Filling the Missing Values with "Missing"
df_pyspark = df_pyspark.na.fill("Missing")
df_pyspark.show()
# %%
imputer = Imputer(
    inputCols=["age", "Experience","Salary"],
    outputCols=["{}_imputed".format(c) for c in ["Age", "Experience","Salary"]]
).setStrategy("mean")
# %%
imputer.fit(df_pyspark).transform(df_pyspark).show()
# %%