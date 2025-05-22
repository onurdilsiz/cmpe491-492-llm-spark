from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03   # 3 % sample keeps skew pattern but trims size

spark = (SparkSession.builder
         .appName("Skewed Join Demo – small")
         .config("spark.sql.autoBroadcastJoinThreshold", -1)   # disable broadcast
         .getOrCreate())

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT payment_type, tpep_pickup_datetime
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

payment_dim = spark.createDataFrame(
    [(1, "Credit"), (2, "Cash"), (3, "No charge"), (4, "Dispute"),
     (5, "Unknown"), (6, "Voided")],
    ["payment_type", "payment_desc"]
)

taxi_skew = taxi.repartition("payment_type")       # preserves the skew
joined    = taxi_skew.join(payment_dim, "payment_type")
out       = joined.groupBy("payment_desc").count()

(out.coalesce(1)
    .write.mode("overwrite")
    .parquet("gs://<your-bucket>/spark_demo/skewed_join_out"))

spark.stop()
