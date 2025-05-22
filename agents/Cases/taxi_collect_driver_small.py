from pyspark.sql import SparkSession

SAMPLE_FRAC = 0.02      # ~2 % of full year ≈ 2 M rows → still hefty for driver

spark = SparkSession.builder.appName("Collect Driver Demo – small").getOrCreate()

taxi = (spark.read.format("bigquery")
        .option("query", f"""
            SELECT tpep_pickup_datetime, total_amount
            FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
            WHERE RAND() < {SAMPLE_FRAC}
        """)
        .load())

rows = taxi.collect()          # pulls a few million rows → driver GC spikes
print(f"Fetched {len(rows):,} rows")

spark.stop()
