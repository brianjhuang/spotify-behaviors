from pyspark.sql import SparkSession
import os

'''
Script used to sample data from our dataset.
'''

SUBSET_SIZE = 0.005 # proportion of dataset to sample

spark = SparkSession.builder.getOrCreate()

spark_fp = os.path.join("/", "Volumes", "Marceline Jr.", "Spotify Dataset", "training_set", )

df = spark.read.load(spark_fp, format="csv", inferSchema="true", header="true")

for i in range(5):
    subset = df.sample(withReplacement=True, fraction=SUBSET_SIZE)
    subset.write.csv(f"./track_features_subset_{i}.csv", header="true")
