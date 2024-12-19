import json
from operator import add
import random

from pyspark import SparkContext, SparkConf
from pyspark.sql import Row, SparkSession
from pyspark.streaming import StreamingContext

json_file = 'file:///Users/zhenglong/proj/spark_demo/data/people.json'
txt_file = 'file:///Users/zhenglong/proj/spark_demo/data/people.txt'


def word_count():
    word_file = 'file:///Users/zhenglong/proj/spark_demo/data/work.txt'

    sc = SparkContext('local', 'test')

    wc = sc.textFile(word_file).\
        flatMap(lambda line: line.split(" ")).\
        map(lambda word: (word, 1)).\
        reduceByKey(lambda a, b: a + b)
    wc.foreach(print)
def load_json():
    def load(s):
        return json.loads(s)

    sc = SparkContext('local', 'test')
    result = sc.textFile(json_file).map(json.loads)
    result.foreach(print)


def data_frame1():
    ss = SparkSession.builder.getOrCreate()
    df = ss.read.json(json_file)
    df.show()


def to_df1():
    def f(x):
        return {
            'name': x[0],
            'age': x[1],
        }

    sc = SparkContext('local', 'test')
    ss = SparkSession.builder.getOrCreate()

    df = sc.textFile(txt_file).\
        map(lambda line: line.split(',')).\
        map(lambda x: Row(**f(x))).\
        toDF()

    df.createOrReplaceTempView('people')

    people_df = ss.sql('select * from people where age > "19"')

    def g(t):
        return 'Name:' + t['name'] + ', ' + 'Age:' + t['age']

    people_df.rdd.map(g).foreach(print)


def to_df2():
    from pyspark.sql.types import StructType
    from pyspark.sql.types import StructField
    from pyspark.sql.types import StringType

    sc = SparkContext('local', 'test')
    ss = SparkSession.builder.getOrCreate()

    people_rdd = sc.textFile(txt_file)

    schema_string = 'name age'

    fields = list(map(
        lambda field_name: StructField(field_name, StringType(), nullable=True),
        schema_string.split(' ')))
    schema = StructType(fields)

    row_rdd = people_rdd.\
        map(lambda line: line.split(',')).\
        map(lambda attributes: Row(attributes[0], attributes[1]))

    people_df = ss.createDataFrame(row_rdd, schema)

    people_df.createOrReplaceTempView('people')

    results = ss.sql('SELECT * FROM people')
    results.rdd.\
        map(lambda attr: 'name:' + attr['name'] + ', ' + 'age:' + attr['age']).\
        foreach(print)


def d_streaming1():
    conf = SparkConf()
    conf.setAppName('TestDStream')
    conf.setMaster('local[2]')
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 20)
    log_file = 'file:///Users/zhenglong/proj/spark_demo/streaming/logfile'

    lines = ssc.textFileStream(log_file)
    words = lines.flatMap(lambda line: line.split(' '))
    wc = words.map(lambda x: (x, 1)).reduceByKey(add)
    wc.pprint()
    ssc.start()
    print('listening')
    ssc.awaitTermination()
    ssc.stop()


def d_streaming2():
    conf = SparkConf()
    conf.setAppName('TestDStream')
    conf.setMaster('local[2]')
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 5)

    lines = ssc.socketTextStream('localhost', 9999)
    words = lines.flatMap(lambda line: line.split(' '))
    wc = words.map(lambda x: (x, 1)).reduceByKey(add)
    wc.pprint()
    ssc.start()
    ssc.awaitTermination()
    ssc.stop()


def d_streaming3():
    import time
    import random
    conf = SparkConf()
    conf.setAppName('TestDStream')
    conf.setMaster('local[2]')
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)

    rdd_queue = [ssc.sparkContext.parallelize([j for j in range(1, random.choice([1001, 1101, 1201]))], 10)
                 for _ in range(5)]

    input_stream = ssc.queueStream(rdd_queue)
    mapped_stream = input_stream.map(lambda x: (x % 10, 1))
    reduced_stream = mapped_stream.reduceByKey(lambda a, b: a + b)
    reduced_stream.pprint()

    ssc.start()
    time.sleep(6)
    ssc.stop()


def d_streaming_save():
    conf = SparkConf()
    conf.setAppName('TestDStream')
    conf.setMaster('local[2]')
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)

    streaming = 'file:///Users/zhenglong/proj/spark_demo/streaming'
    ssc.checkpoint(streaming)

    initial_state_rdd = sc.parallelize([(u'hello', 1), (u'world', 1)])

    def update_func(new_values, last_sum):
        return sum(new_values) + (last_sum or 0)

    lines = ssc.socketTextStream('localhost', 9999)
    wc = lines.flatMap(lambda line: line.split(' '))
    wc = wc.map(lambda x: (x, 1))
    wc = wc.updateStateByKey(update_func, initialRDD=initial_state_rdd)

    wc.saveAsTextFiles(streaming+'/output.txt')

    wc.pprint()
    ssc.start()
    ssc.awaitTermination()
    ssc.stop()


def structured_streaming_demo():
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import explode, split

    spark = SparkSession.builder.appName('StructuredNetworkWordCount').getOrCreate()
    lines = spark.readStream.format('socket').\
        option('host', 'localhost').\
        option('port', 9999).\
        load()

    words = lines.select(
        explode(
            split(lines.value, ' ')
        ).alias('word')
    )

    wc = words.groupBy('word').count()
    query = wc.writeStream.outputMode('complete').\
        format('console').\
        start()
    query.awaitTermination()
    query.stop()



def topn(key, iter):
    sorted_iter = sorted(iter, reverse=True)
    length = len(sorted_iter)
    return map(lambda x: (key, x), sorted_iter[:min(length, 3)])


def top3_1():
    rint = random.randint
    top_file = 'file:///Users/zhenglong/proj/spark_demo/data/top.txt'
    sc = SparkContext('local[*]', 'test')
    rdd = sc.textFile(top_file)
    ret = rdd.map(lambda line: line.split(' ')) \
        .filter(lambda e: len(e) == 2) \
        .mapPartitions(lambda iter: map(lambda e: ((rint(1, 10), e[0]), e[1]), iter)) \
        .groupByKey() \
        .flatMap(lambda e: topn(e[0][1], e[1])) \
        .groupByKey() \
        .flatMap(lambda e: topn(e[0], e[1])) \
        .collect()
    print(ret)


def f(a, b):
    a.append(b)
    sorted_a = sorted(a, reverse=True)
    return sorted_a[:min(len(sorted_a), 3)]


def g(a, b):
    a.extend(b)
    sorted_a = sorted(a, reverse=True)
    return sorted_a[:min(len(sorted_a), 3)]


def top3():
    top_file = 'file:///Users/zhenglong/proj/spark_demo/data/top.txt'
    sc = SparkContext('local[*]', 'test')
    rdd = sc.textFile(top_file)
    ret = rdd.map(lambda line: line.split(' ')) \
        .filter(lambda e: len(e) == 2) \
        .aggregateByKey(zeroValue=[],
                        seqFunc=lambda a, b: f(a, b),
                        combFunc=lambda a, b: g(a, b)) \
        .collect()
    print(ret)


if __name__ == '__main__':
    structured_streaming_demo()