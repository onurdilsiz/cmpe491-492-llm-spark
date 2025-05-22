from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = (SparkSession.builder
         .appName("Broadcast Threshold Demo – small")
         .config("spark.sql.autoBroadcastJoinThreshold", 1 * 1024 * 1024)   # 1 MB – too low
         .getOrCreate())

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT payment_type, total_amount
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# Dimension ≈ 2 MB, so would broadcast with default 10 MB but not with 1 MB
dim = spark.range(200000).selectExpr("id % 6 as payment_type", "'desc' as txt")  # ~2 MB

joined = taxi.join(dim, "payment_type")
joined.groupBy("payment_type").avg("total_amount").show()

spark.stop()
