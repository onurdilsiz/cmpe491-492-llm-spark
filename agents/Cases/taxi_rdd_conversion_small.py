from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.03

spark = SparkSession.builder.appName("RDD Conversion Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT trip_distance
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

# BAD: switch to RDD, lose Tungsten / Catalyst ops vectorization
rdd = taxi.rdd.map(lambda row: row.trip_distance * 1.60934)  # convert miles to km
df_km = spark.createDataFrame(rdd, "double").toDF("trip_distance_km")

df_km.groupBy().avg("trip_distance_km").show()

spark.stop()
