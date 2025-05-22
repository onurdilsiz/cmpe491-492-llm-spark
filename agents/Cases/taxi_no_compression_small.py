from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = (SparkSession.builder
         .appName("No Compression Demo – small")
         .config("spark.sql.parquet.compression.codec", "none")   # disable compression
         .getOrCreate())

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT trip_distance, total_amount
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

(taxi.repartition(1)  # single uncompressed file, but still large on disk
     .write.mode("overwrite")
     .parquet("gs://<your-bucket>/spark_demo/no_compression_out"))

spark.stop()
