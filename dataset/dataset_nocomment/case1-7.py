from pyspark.sql import functions as F 

def custom_repartitioner(df, max_records_in_each_file, distribution_keys):
    dist_cols = [key.strip() for key in distribution_keys.split(",")]
    print("distribution cols: "+str(dist_cols))
    
    agg_df = df.select(*dist_cols)\
                .withColumn('_partColAgg', F.concat(*dist_cols))\
                .drop(*dist_cols)\
                .groupBy('_partColAgg')\
                .agg(F.count(F.lit(1)).alias("records_count"))
    agg_df = agg_df.withColumn('_num', F.ceil(F.col('records_count').cast('double')/F.lit(max_records_in_each_file)))\
                    .select('_num', '_partColAgg')
        
    agg_df.cache()
    number_of_files = max(int(partition._num) for partition in agg_df.collect())
    print('max num of files: '+str(number_of_files))

    df = df.withColumn('_partColMain', F.concat(*dist_cols))
    df = df.join(F.broadcast(agg_df), F.col('_partColMain')==F.col('_partColAgg'), 'inner')\
                .drop('_partColAgg')
    
    df = df.withColumn('_unique_id', F.monotonically_increasing_id())\
            .withColumn('_salted_key', F.col('_unique_id') % F.col('_num'))
    
    df = df.drop('_partColMain', '_num', '_unique_id')
    
    df = df.repartition(number_of_files, '_salted_key')\
            .drop('_salted_key')
    
    return df 