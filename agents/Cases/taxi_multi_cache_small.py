from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = SparkSession.builder.appName("Multi‑Cache Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT vendor_id, passenger_count, trip_distance
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

df1 = taxi.filter("passenger_count = 1").cache()
df2 = taxi.filter("passenger_count = 2").cache()

print(df1.count(), df2.count())   # triggers caching two large DFs

# never unpersist df1 or df2 – memory hog

spark.stop()
