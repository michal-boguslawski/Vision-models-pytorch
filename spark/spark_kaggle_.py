from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, RobustScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize SparkSession
spark = SparkSession.builder.appName("LoadDataFrameExample").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

file_path = "train.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True,
    # samplingRatio=0.01
)
#df = df.sample(fraction=0.1)
df.show()

all_column_names = df.columns
output_column = "Premium Amount"
feature_columns = all_column_names[1:-1]
# feature_columns.remove("Policy Start Date")
df = df.withColumn("Policy Start Date", F.unix_timestamp("Policy Start Date").cast("double"))

string_columns = [col for col, dtype in df.dtypes if dtype == 'string']
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") for col in string_columns]
feature_columns = [col for col in feature_columns if col not in string_columns]
feature_columns += [f"{col}_index" for col in string_columns]

# df.show()
df = df.fillna({'Age': 0, 'Annual Income': 0, 'Health Score': 0, 'Credit Score': 0, 'Previous Claims': 0,
                'Number of Dependents': 0, 'Vehicle Age': 0, 'Insurance Duration': 0})

df = df.withColumnRenamed(output_column, "label")

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features",
                            handleInvalid="keep")

# regressor = DecisionTreeRegressor(featuresCol="features", labelCol='label')


max_iter = 25
reg_param = 0.01

glm = GeneralizedLinearRegression(family="gamma",
                                  link="log",
                                  featuresCol='scaled_features',
                                  # predictionCol='glm_gamma_log_prediction',
                                  maxIter=max_iter,
                                  regParam=reg_param
                                  )

glm_gamma_ident_regressor = GeneralizedLinearRegression(family="gamma",
                                                        link="identity",
                                                        featuresCol='scaled_features',
                                                        # predictionCol='glm_gamma_identity_prediction',
                                                        maxIter=max_iter,
                                                        regParam=reg_param
                                                        )

glm_gamma_inverse_regressor = GeneralizedLinearRegression(family="gamma",
                                                          link="inverse",
                                                          featuresCol='scaled_features',
                                                          # predictionCol='glm_gamma_inverse_prediction',
                                                          maxIter=max_iter,
                                                          regParam=reg_param
                                                          )

glm_gaussian_log_regressor = GeneralizedLinearRegression(family="gaussian",
                                                         link="log",
                                                         featuresCol='scaled_features',
                                                         # predictionCol='glm_gaussian_log_prediction',
                                                         maxIter=max_iter,
                                                         regParam=reg_param
                                                         )

gbt = GBTRegressor(featuresCol="features",
                   maxIter=max_iter,
                   maxDepth=10,
                   maxBins=32,
                   lossType='squared',
                   impurity='variance'
                   )

rf = RandomForestRegressor(featuresCol="features",
                           maxDepth=10,
                           numTrees=30)

regressors = [
    glm, glm_gamma_ident_regressor, glm_gamma_inverse_regressor, glm_gaussian_log_regressor,
    gbt, rf]

df = df.cache()

std_scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)
rob_scaler = RobustScaler(inputCol='features', outputCol='scaled_features',
                          lower=0.25, upper=0.75)

for regressor in regressors:
    for scaler in [std_scaler, rob_scaler]:
        print(type(regressor).__name__, type(scaler).__name__)

        if type(regressor).__name__ in ['GBTRegressor', 'RandomForestRegressor']:

            pipeline = Pipeline(stages=indexers + [assembler,
                                                   regressor,
                                                   ])
        else:
            pipeline = Pipeline(stages=indexers + [assembler,
                                                   scaler,
                                                   regressor,
                                                   ])

        model = pipeline.fit(df)
        prediction = model.transform(df)
        prediction.select('label',
                          'prediction'
                          ).show()
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(prediction)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
        usedModel = model.stages[-1]
        print(usedModel)
