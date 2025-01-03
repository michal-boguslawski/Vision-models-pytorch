from pyspark.sql import SparkSession
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression

# Initialize SparkSession
spark = SparkSession.builder.appName("LoadDataFrameExample").getOrCreate()

file_path = "train.csv"

indexers = []

df = spark.read.csv(file_path, header=True, inferSchema=True)

all_column_names = df.columns
output_column = "Premium Amount"
feature_columns = all_column_names[1:-1]
string_columns = []
string_columns_index = []
feature_columns.remove("Policy Start Date")
df.show()
pandas_df = df.select(output_column).toPandas()
pandas_df = pandas_df.apply(np.log)

print(pandas_df.head())
ax = pandas_df.plot.hist()
fig = ax.get_figure()
fig.savefig('figure.jpg')