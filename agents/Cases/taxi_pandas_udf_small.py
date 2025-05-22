from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf
import pandas as pd

SAMPLE_FRAC = 0.03

spark = SparkSession.builder.appName("Pandas UDF Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT trip_distance
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# BAD: row-at-a-time pandas_udf (SERIES) where built‑in avg() would be faster
@pandas_udf("double")
def add_noise(col: pd.Series) -> pd.Series:
    return col * 1.05

taxi.withColumn("distance_noise", add_noise(taxi.trip_distance)).groupBy().avg("distance_noise").show()

spark.stop()
