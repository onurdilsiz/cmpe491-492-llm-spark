from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import upper, col

def filtered_data_tony(rdd):
    # Filter rows where '_c5' is null and show the results
    filtered_not_null_product_cat_rdd = rdd.filter(~col('_c5').isNull())
    filtered_not_null_payment_type_rdd = filtered_not_null_product_cat_rdd.filter(~col('_c6').isNull())
    filtered_not_null_qty_rdd = filtered_not_null_payment_type_rdd.filter(~col('_c7').isNull())
    filtered_not_null_price_rdd = filtered_not_null_qty_rdd.filter(~col('_c8').isNull())
    # There are no null values from c5-c8 which is what matters so this is fine

    #_c5 is product category
    # None of them contain any numbers so the data seems to be clean
    filtered_no_number_product_cat_rdd = filtered_not_null_price_rdd.filter(~col('_c5').rlike('(?=.*\\d)(?=.*[a-zA-Z])'))
    filtered_no_number_payment_type_rdd = filtered_no_number_product_cat_rdd.filter(~col('_c6').rlike('(?=.*\\d)(?=.*[a-zA-Z])'))
    
    filtered_no_number_failure_reason_rdd = filtered_no_number_payment_type_rdd.filter(
        col('_c15').isNull() | ~col('_c15').rlike('.*\\d.*')
    )

    #refined_filter_price_rdd = df.filter(~col('_c8').rlike('^[^0-9]*$') & (col('_c8') != '') & (col('_c8') != "46284y924"))

    #filtered_price_rdd.show()

    filtered_product_category_rdd = filtered_no_number_failure_reason_rdd.filter(
        ~upper(col('_c5')).contains("ERROR") &
        ~upper(col('_c5')).contains("BOOM") &
        ~upper(col('_c5')).contains("THIS") &
        ~upper(col('_c5')).contains("CORRUPTED") &
        ~upper(col('_c5')).contains("!")
    )
    #filtered_product_category_rdd.show()
    #_c6 payment_type 6 Errors for payment type
    filtered_payment_type_rdd = filtered_product_category_rdd.filter(
        ~upper(col('_c6')).contains("ERROR") &
        ~upper(col('_c6')).contains("BOOM") &
        ~upper(col('_c6')).contains("THIS") &
        ~upper(col('_c6')).contains("CORRUPTED") &
        ~upper(col('_c6')).contains("!")
    )

    #_c7 qty 10 errors found
    filtered_qty_rdd = filtered_payment_type_rdd.filter(~col('_c7').rlike('^[^0-9]*$') & (col('_c7') != ''))
    non_zero_df = filtered_qty_rdd.filter(col('_c7').cast('int') != 0)

    filtered_price_rdd = non_zero_df.filter(col('_c8').rlike('^[0-9]*\\.?[0-9]+$') & (col('_c8') != ''))
    filtered_price_rdd = filtered_price_rdd.filter(col('_c8').cast('int') != 0)
    # Filter out rows where '_c15' contains any of the keywords 11 erros
    filtered_excluded_keywords_rdd = filtered_price_rdd.filter(
        (upper(col('_c15')).contains("NETWORK") |
        upper(col('_c15')).contains("UNABLE") |
        upper(col('_c15')).contains("INSUFFICIENT") | col('_c15').isNull())
    )


    return filtered_excluded_keywords_rdd

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    # Create Spark Session 
    spark = SparkSession(sc)
    #conf = SparkConf().setAppName("Example1").setMaster("local")


    #sc = SparkContext(conf = conf)

    #team2_rdd = sc.textFile("/csv_data/data_team_2.csv")
    path = sc.textFile("file:///root/data_team_2.csv")

    df = spark.read.csv(path)

    #df.printSchema()

    filtered_df = filtered_data_tony(df)

    #filtered_df.show()

    output_path = "file:///root/filtered_data_team_2_clean/"
    filtered_df.write \
        .mode('default') \
        .option("header", "false") \
        .csv(output_path)

    # Convert DataFrame to RDD of strings
    #rdd1 = filtered_df.rdd.map(lambda row: ','.join(str(field) for field in row))
    #rdd1.saveAsTextFile("file:///root/data_team_2_clean.csv")
    # To run:  spark-submit PySpark_Example.py