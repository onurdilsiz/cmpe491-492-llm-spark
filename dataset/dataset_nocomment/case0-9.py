from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
rdd = spark.sparkContext.textFile("/apps/sparkbyexamples/src/pyspark-examples/data.txt")

for element in rdd.collect():
    print(element)
rdd2=rdd.flatMap(lambda x: x.split(" "))
for element in rdd2.collect():
    print(element)
rdd3=rdd2.map(lambda x: (x,1))
for element in rdd3.collect():
    print(element)
rdd4=rdd3.reduceByKey(lambda a,b: a+b)
for element in rdd4.collect():
    print(element)
rdd5 = rdd4.map(lambda x: (x[1],x[0])).sortByKey()
for element in rdd5.collect():
    print(element)
rdd6 = rdd5.filter(lambda x : 'a' in x[1])
for element in rdd6.collect():
    print(element)