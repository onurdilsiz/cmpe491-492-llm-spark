from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.04   # little larger sample to create backlog

spark = (SparkSession.builder
         .appName("Autoscaling Backlog Demo – small")
         .config("spark.dynamicAllocation.enabled", "true")
         .config("spark.dynamicAllocation.initialExecutors", "1")  # intentionally tiny
         .config("spark.dynamicAllocation.minExecutors", "1")
         .config("spark.dynamicAllocation.maxExecutors", "100")
         .getOrCreate())

# Heavy-ish shuffle → needs many executors eventually
taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT passenger_count, trip_distance
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# Large groupBy triggers many parallel tasks
taxi.groupBy("passenger_count").avg("trip_distance").write.mode("overwrite").format("noop").save()

spark.stop()
