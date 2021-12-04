from pyspark.ml.feature import RFormula
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, StructType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import FeatureHasher

spark = SparkSession.builder.appName("CTR").getOrCreate()
schema = StructType([
    StructField("id", StringType(), True),
    StructField("click", IntegerType(), True),
    StructField("hour", IntegerType(), True),
    StructField("C1", StringType(), True),
    StructField("banner_pos", StringType(), True),
    StructField("site_id", StringType(), True),
    StructField("site_domain", StringType(), True),
    StructField("site_category", StringType(), True),
    StructField("app_id", StringType(), True),
    StructField("app_domain", StringType(), True),
    StructField("app_category", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("device_ip", StringType(), True),
    StructField("device_model", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("device_conn_type", StringType(), True),
    StructField("C14", StringType(), True),
    StructField("C15", StringType(), True),
    StructField("C16", StringType(), True),
    StructField("C17", StringType(), True),
    StructField("C18", StringType(), True),
    StructField("C19", StringType(), True),
    StructField("C20", StringType(), True),
    StructField("C21", StringType(), True),
    ])
df = spark.read.csv("D:/AmirReza/Desktop/Aras/term 1/ML/ML-Class-HW/Project4/train1.csv", schema=schema, header=True)
df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')
df = df.withColumnRenamed("click", "label")
df_train, df_test = df.randomSplit([0.7, 0.3], 42)
categorical = df_train.columns
categorical.remove('label')

hasher = FeatureHasher(numFeatures=1000, inputCols=categorical, outputCol="features")
hasher.transform(df_train).select("features").show()
classifier = LogisticRegression(maxIter=20, regParam=0.000, elasticNetParam=0.000)
stages = [hasher, classifier]
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df_train)
predictions = model.transform(df_test)
predictions.cache()
ev = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName="areaUnderROC")
print(ev.evaluate(predictions))

cat_inter = ['C1', 'C20']
cat_no_inter = [c for c in categorical if c not in cat_inter]
concat = '+'.join(categorical)
interaction = ':'.join(cat_inter)
formula = "label ~ " + concat + '+' + interaction
print(formula)
interactor = RFormula(formula=formula, featuresCol="features", labelCol="label").setHandleInvalid("keep")
interactor.fit(df_train).transform(df_train).select("features").show()

classifier = LogisticRegression(maxIter=20, regParam=0.000, elasticNetParam=0.000)
stages = [interactor, classifier]
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df_train)
predictions = model.transform(df_test)
predictions.cache()
ev = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName="areaUnderROC")
print(ev.evaluate(predictions))
