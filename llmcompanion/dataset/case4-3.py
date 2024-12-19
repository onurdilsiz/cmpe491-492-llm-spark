
df = spark.range(5)

def five():
    return 5

from pyspark.sql.functions import udf
five_udf = udf(five)

df.select(five_udf()).display()


from pyspark.sql.functions import udf

@udf
def five():
    return 5

df.select(five()).show()


from pyspark.sql.functions import pandas_udf

@pandas_udf(returnType="int")
def five() -> int:
    return 5

df.select(five()).show()


result_df = df.selectExpr("my_custom_scala_fn(id) as id")
result_df.display()



result_df.explain()


from pyspark.sql.functions import pandas_udf

import pandas as pd

@pandas_udf(returnType='int')
def identity(rows: pd.Series) -> pd.Series:
    return rows

df.select(identity('id')).display()



from pyspark.sql.functions import split

strings = spark.createDataFrame([
    ("50000.0#0#0#", "#"),
    ("0@1000.0@", "@"),
    ("1$", "$"),
    ("1000.00^Test_string", "^")], 'name string, delimiter string')

strings.show()


from pyspark.sql.functions import expr
strings.select(expr("split(name, delimiter)")).show()


strings.selectExpr("split(name, delimiter)").show()


strings.createOrReplaceTempView("strings_table")


spark.sql("SELECT split(name, delimiter) FROM strings_table").show()



ns = spark.range(5)



