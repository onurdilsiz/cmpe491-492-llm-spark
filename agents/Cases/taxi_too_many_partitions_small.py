from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = (SparkSession.builder
         .appName("Too Many Partitions Demo – small")
         .config("spark.sql.shuffle.partitions", 10000)   # absurdly high
         .getOrCreate())

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT trip_distance, passenger_count
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

by_pass = taxi.groupBy("passenger_count").avg("trip_distance")

print(by_pass.collect())   # triggers 10 k‑way shuffle
print(by_pass.count())     # triggers the identical shuffle again

spark.stop()
