from pyspark.sql import SparkSession
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

for col, col_type in df.dtypes:
    if col_type == 'string':
        new_col_name = f"{col}_index"
        string_columns.append(col)
        string_columns_index.append(new_col_name)

        indexer = StringIndexer(inputCol=col, outputCol=new_col_name, handleInvalid="keep")
        indexers.append(indexer)
        feature_columns.remove(col)
        feature_columns.append(new_col_name)

        #df = indexer.fit(df).transform(df)

#df.show()
df = df.fillna({'Age': 0, 'Annual Income': 0, 'Health Score': 0, 'Credit Score': 0, 'Previous Claims': 0,
                'Number of Dependents': 0, 'Vehicle Age': 0, 'Insurance Duration': 0})

df = df.withColumnRenamed(output_column, "label")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features",
                            handleInvalid="keep")

#regressor = DecisionTreeRegressor(featuresCol="features", labelCol='label')

regressor = GeneralizedLinearRegression(family="gamma", link="inverse", maxIter=10, regParam=0.3)

pipeline = Pipeline(stages=indexers + [assembler, regressor])
model = pipeline.fit(df)

print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))

prediction = model.transform(df)
#transformed_df = assembler.transform(df)

#indexed = featureIndexer.transform(transformed_df)

print(string_columns)
print(string_columns_index)
print(df.columns)
df.printSchema()

prediction.select(output_column, 'prediction').show()

