from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
data=[("Z", 1),("A", 20),("B", 30),("C", 40),("B", 30),("B", 60)]
inputRDD = spark.sparkContext.parallelize(data)
  
listRdd = spark.sparkContext.parallelize([1,2,3,4,5,3,2])

seqOp = (lambda x, y: x + y)
combOp = (lambda x, y: x + y)
agg=listRdd.aggregate(0, seqOp, combOp)
seqOp2 = (lambda x, y: (x[0] + y, x[1] + 1))
combOp2 = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
agg2=listRdd.aggregate((0, 0), seqOp2, combOp2)
agg2=listRdd.treeAggregate(0,seqOp, combOp)
from operator import add
foldRes=listRdd.fold(0, add)
redRes=listRdd.reduce(add)
add = lambda x, y: x + y
redRes=listRdd.treeReduce(add)
data = listRdd.collect()
print("Count : "+str(listRdd.count()))
print("countApprox : "+str(listRdd.countApprox(1200)))
print("countApproxDistinct : "+str(listRdd.countApproxDistinct()))
print("countApproxDistinct : "+str(inputRDD.countApproxDistinct()))
print("countByValue :  "+str(listRdd.countByValue()))


print("first :  "+str(listRdd.first()))
print("first :  "+str(inputRDD.first()))

print("top : "+str(listRdd.top(2)))
print("top : "+str(inputRDD.top(2)))
print("min :  "+str(listRdd.min()))
print("min :  "+str(inputRDD.min()))
print("max :  "+str(listRdd.max()))
print("max :  "+str(inputRDD.max()))
print("take : "+str(listRdd.take(2)))
print("takeOrdered : "+ str(listRdd.takeOrdered(2)))
print("take : "+str(listRdd.takeSample()))

