from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = SparkSession.builder.appName("Cache No Unpersist Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT vendor_id, trip_distance, total_amount
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

taxi.cache()    # hangs around in executor memory

vendor_stats = (taxi
                .groupBy("vendor_id")
                .avg("trip_distance", "total_amount"))

vendor_stats.show()

# intentionally *no* taxi.unpersist()

spark.stop()
