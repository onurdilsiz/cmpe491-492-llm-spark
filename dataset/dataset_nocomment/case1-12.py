
import os
absolute_dir_path = os.path.abspath("./data/")


absolute_dir_path = "file:" + absolute_dir_path


airbnb_df = spark.read.format("delta").load(f"{absolute_dir_path}/imputed_results") 



train_df, val_df, test_df= airbnb_df.randomSplit([.7, .15, .15], seed=42)
print(train_df.cache().count())


val_df.count()


train_repartition_df, val_repartition_df, test_repartition_df = (airbnb_df
                                             .repartition(24)
                                             .randomSplit([.7, .15, .15], seed=42))

print(train_repartition_df.count())


display(train_repartition_df.limit(5))



display(train_df.limit(5))

train_df.columns


display(train_df.select("price", "review_scores_rating").summary())

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="review_scores_rating", labelCol="price")



print(lr.explainParams())

lr_model = lr.fit(train_df)

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["review_scores_rating"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)


display(vec_train_df.limit(5))

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(vec_train_df)


print(lr_model.coefficients)

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

lr_model.summary.r2

lr_model.summary.rootMeanSquaredError

lr_model.summary.r2
vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)

display(pred_df.select("review_scores_rating", "features", "price", "prediction").limit(5))

hi = (pred_df["price"] - pred_df["prediction"])**2

pred_df = pred_df.withColumn("difference", hi)

display(pred_df.select("", "features", "price" , "prediction", "difference").limit(5))

display(pred_df.select("price", "prediction").limit(5))
from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")


r2_evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol = "price", metricName = "r2")

print(r2_evaluator.evaluate(pred_df))