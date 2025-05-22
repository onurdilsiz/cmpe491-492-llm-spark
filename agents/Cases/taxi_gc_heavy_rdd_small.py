from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.04

spark = SparkSession.builder.appName("GC Heavy RDD Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT trip_distance
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# Convert to RDD and generate many small objects → executor GC
def explode(row):
    # emit 100 tiny floats per input row
    return [row.trip_distance * i / 100 for i in range(100)]

noisy = taxi.rdd.flatMap(explode)
print(noisy.sum())

spark.stop()
