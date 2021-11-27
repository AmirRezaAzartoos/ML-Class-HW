# pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("test").getOrCreate()
df = spark.read.csv("C:/ProgramData/Anaconda3/Lib/site-packages/pyspark/examples/src/main/resources/people.csv", header=True, sep=';')

df.show()
df.count()
df.printSchema()
df.select("name").show()
df.select(["name", "job"]).show()
df.filter(df['age'] > 31).show()