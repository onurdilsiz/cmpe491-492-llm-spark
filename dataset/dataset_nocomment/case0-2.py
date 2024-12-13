
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint

from pyspark.mllib.classification import LogisticRegressionWithLBFGS

sc = SparkContext("local", "filter app")

file_path_spam = 'spam.txt'
file_path_non_spam = 'Ham.txt'

spam_rdd = sc.textFile(file_path_spam)
non_spam_rdd = sc.textFile(file_path_non_spam)

spam_words = spam_rdd.flatMap(lambda email: email.split(' '))
non_spam_words = non_spam_rdd.flatMap(lambda email: email.split(' '))

tf = HashingTF(numFeatures=200)

spam_features = tf.transform(spam_words)
non_spam_features = tf.transform(non_spam_words)

spam_samples = spam_features.map(lambda features:LabeledPoint(1, features))
non_spam_samples = non_spam_features.map(lambda features:LabeledPoint(0, features))

samples = spam_samples.join(non_spam_samples)

train_samples,test_samples = samples.randomSplit([0.8, 0.2])

model = LogisticRegressionWithLBFGS.train(train_samples)

predictions = model.predict(test_samples.map(lambda x: x.features))

labels_and_preds = test_samples.map(lambda x: x.label).zip(predictions)

accuracy = labels_and_preds.filter(lambda x: x[0] == x[1]).count() / float(test_samples.count())
print("Model accuracy : {:.2f}".format(accuracy))
