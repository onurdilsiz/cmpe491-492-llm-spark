from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

SAMPLE_FRAC = 0.03

spark = SparkSession.builder.appName("Python UDF Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT fare_amount, total_amount
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# BAD: Python UDF – disables Catalyst & Arrow optimizations
def calc_tip(fare, total):
    return (total - fare) / fare if fare else None

tip_udf = udf(calc_tip, DoubleType())

taxi_with_tip = taxi.withColumn("tip_pct", tip_udf(col("fare_amount"), col("total_amount")))
taxi_with_tip.groupBy().avg("tip_pct").show()

spark.stop()
