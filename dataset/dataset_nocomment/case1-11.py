import os
import json
import subprocess
from datetime import datetime, timezone
import numpy as np
import iris
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder.appName('AWS-Take-Home').getOrCreate()


def get_bucket_links():
    path_to_variables = 'variables.json'
    with open(path_to_variables, "r") as variables_json:
        variables = json.load(variables_json)
    raw_data_bucket = variables['etl']['raw_data_bucket']
    return raw_data_bucket


def utc_timestamp(hours_since_first_epoch):
    epoch = hours_since_first_epoch * 60 * 60
    ts = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
    return ts


def create_feature_dataframe(data, feature, feature_dtype_mapping,
                             feature_index_mapping, dim_set):
    rows = []

    if dim_set == 1:
        schema = StructType([
            StructField("time", StringType(), True),
            StructField("grid_latitude", FloatType(), True),
            StructField("grid_longitude", FloatType(), True),
            StructField(feature, feature_dtype_mapping[feature], True),
        ])
    elif dim_set == 2:
        schema = StructType([
            StructField("time", StringType(), True),
            StructField("pressure", FloatType(), True),
            StructField("grid_latitude", FloatType(), True),
            StructField("grid_longitude", FloatType(), True),
            StructField(feature, feature_dtype_mapping[feature], True),
        ])

    feature_data = data.extract(feature)[feature_index_mapping[feature]]
    times = feature_data.coord('time').points
    latitudes = feature_data.coord('grid_latitude').points
    longitudes = feature_data.coord('grid_longitude').points

    if dim_set == 2:
        pressures = feature_data.coord('pressure').points

    feature_data = feature_data.data
    np.ma.set_fill_value(feature_data, -999)
    feature_data = feature_data.filled()

    if dim_set == 1:
        for i, time in enumerate(times):
            time = utc_timestamp(time)
            for j, latitude in enumerate(latitudes):
                for k, longitude in enumerate(longitudes):
                    try:
                        rows.append([time, latitude.item(), longitude.item(), feature_data[i][j][k].item()])
                    except:
                        pass

    elif dim_set == 2:
        for i, time in enumerate(times):
            time = utc_timestamp(time)
            for j, pressure in enumerate(pressures):
                for k, latitude in enumerate(latitudes):
                    for l, longitude in enumerate(longitudes):
                        try:
                            rows.append([time, pressure.item(), latitude.item(), longitude.item(), feature_data[i][j][k][l].item()])
                        except:
                            pass

    df = spark.createDataFrame(rows, schema)
    return df


def process_netcdf(file_name):
    s3 = boto3.resource('s3')
    raw_data_bucket = get_bucket_links()
    s3.Bucket(raw_data_bucket).download_file(file_name, 'tmp.nc')
    data = iris.load('tmp.nc')
    features = {
        'dew_point_temperature': 1,
        'air_temperature': 1,
        'wind_speed_of_gust': 1,
    }
    feature_dtype_mapping = {
        'dew_point_temperature': FloatType(),
        'air_temperature': FloatType(),
        'wind_speed_of_gust': FloatType(),
    }
    feature_index_mapping = {
        'dew_point_temperature': 0,
        'air_temperature': 2,
        'wind_speed_of_gust': 0,
    }

    dfs = []

    for feature, dim_set in features.items():
        df = create_feature_dataframe(data, feature, feature_dtype_mapping,
                                      feature_index_mapping, dim_set)

        df = df.withColumn("year", year(col("time").cast("timestamp")))\
               .withColumn("month", month(col("time").cast("timestamp")))\
               .withColumn("day", dayofmonth(col("time").cast("timestamp")))\
               .repartition(1000)

        df = df.sort(asc('time')).coalesce(1)
        dfs.append([df, feature])
    os.remove('tmp.nc')
    return dfs