from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = SparkSession.builder.appName("Many Small Files Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT vendor_id, passenger_count
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# BAD: create thousands of tiny output files
(taxi.repartition(5000)      # 5 000 partitions → 5 000 small files
     .write.mode("overwrite")
     .parquet("gs://<your-bucket>/spark_demo/tiny_files_out"))

spark.stop()
