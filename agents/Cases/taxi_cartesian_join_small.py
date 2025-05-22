from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.02

spark = SparkSession.builder.appName("Cartesian Join Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT payment_type, passenger_count
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

dim = spark.createDataFrame([(i,) for i in range(100)], ["dummy_key"])

# BAD: crossJoin – explosive output
cross = taxi.crossJoin(dim)
print(cross.count())

spark.stop()
