from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.04

spark = (SparkSession.builder
         .appName("Memory Spill Demo – small")
         .config("spark.memory.fraction", 0.1)   # shrink execution memory
         .getOrCreate())

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT passenger_count, trip_distance, total_amount
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# Big shuffle & sort to trigger disk spill
(taxi.repartition(200)
     .sortWithinPartitions("trip_distance")    # wide sort
     .groupBy("passenger_count")
     .avg("total_amount")
     .write.mode("overwrite").format("noop").save())

spark.stop()
