#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Transformer
from typing import Iterable
from pyspark.sql import DataFrame


# CUSTOM TRANSFORMER ----------------------------------------------------------------
class NumericalTransform(Transformer):
    """
    A custom Transformer to transfrom numerical features from string to float and fill in null values
    """

    def __init__(self, numerical: Iterable[str]):
        super(NumericalTransform, self).__init__()
        self.numerical = numerical

    def _transform(self, df: DataFrame) -> DataFrame:
        for col_name in numerical:
            df = df.withColumn(col_name, col(col_name).cast('float'))
        
        df = df.withColumn('isFraud', col('isFraud').cast('float'))
        df = df.na.fill(-999, numerical)
        return df
    
# CUSTOM TRANSFORMER ----------------------------------------------------------------
class CategoricalFillNull(Transformer):
    """
    A custom Transformer to fill in null values for categorical features after the StringIndexer
    """

    def __init__(self, categorical: Iterable[str]):
        super(CategoricalFillNull, self).__init__()
        self.categorical = categorical

    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.na.fill(-999, categorical)
        return df

spark = SparkSession     .builder     .master("local[*]")     .config("spark.driver.memory", "20g")     .appName("Train_ml")     .getOrCreate()

train = spark.read.options(header='true').csv("10k.csv")

#missing = train.select([count(when(col(c).isNull(), c)).alias(c) for c in train.columns])
#null_counts = missing.collect()[0].asDict()
#to_drop = [k for k, v in null_counts.items() if v > (1000*0.001)]
#print(len(to_drop))
#print(to_drop)
#train = train.drop(*to_drop)
#train = train.sample(0.1)
#print(len(train.columns))
#print(train.count())


# In[2]:


null_drop = ['dist2', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'id_03', 'id_04', 'id_07', 'id_08', 'id_09', 'id_10', 'id_14', 'id_18', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_30', 'id_32', 'id_33', 'id_34']
corr_drop_v = ['V3', 'V5', 'V6', 'V8', 'V9', 'V11', 'V16', 'V18', 'V22', 'V26', 'V28', 'V30', 'V32', 'V34', 'V43', 'V46', 'V49', 'V52', 'V58', 'V60', 'V70', 'V72', 'V74','V91', 'V93', 'V97', 'V103', 'V106', 'V128', 'V132', 'V133', 'V134', 'V167', 'V168', 'V178', 'V182', 'V189', 'V21', 'V33', 'V81', 'V101', 'V102', 'V126', 'V127', 'V177', 'V179', 'V193', 'V195', 'V196', 'V197', 'V201', 'V204', 'V211', 'V212', 'V219', 'V222', 'V225', 'V231', 'V232', 'V233', 'V237', 'V241', 'V244', 'V247', 'V249', 'V251', 'V254', 'V256', 'V259', 'V269', 'V272', 'V273', 'V279', 'V280','V292', 'V295', 'V298', 'V304', 'V306', 'V307', 'V316', 'V317', 'V318', 'V293', 'V295', 'V306', 'V308', 'V316', 'V318']
corr_drop_c = ['C2', 'C4', 'C6', 'C8', 'C10', 'C11', 'C12', 'C14']
corr_drop_d = ['D2']
to_drop = null_drop + corr_drop_v + corr_drop_c + corr_drop_d

train = train.drop(*to_drop)
train.printSchema()


# In[3]:


raw_features = train.columns
raw_features.remove('isFraud')

categorical = ['ProductCD',
               'card1','card2','card3','card4','card5','card6', 
               'addr1', 'addr2',
               'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
               'P_emaildomain', 'R_emaildomain',
               'DeviceType','DeviceInfo',
               'id_12',  'id_13', 'id_14',  'id_15',
               'id_16',  'id_17',  'id_18',  'id_19',  'id_20',
               'id_21',  'id_22',  'id_23',  'id_24',  'id_25',
               'id_26',  'id_27',  'id_28',  'id_29',  'id_30',
               'id_31',  'id_32',  'id_33',  'id_34',  'id_35',  'id_36',  'id_37',  'id_38']
categorical = list(set(categorical) - set(to_drop))
print(categorical)


# In[4]: 


numerical = list(set(raw_features) - set(categorical))
numerical_transform = NumericalTransform(numerical = numerical)


# In[5]:


categorical_output = [x + '_index' for x in categorical]
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCols=categorical, outputCols=categorical_output, handleInvalid="keep")
#train = indexer.fit(train).transform(train)

categorical_fill_null = CategoricalFillNull(categorical = categorical_output)
#train = categorical_fill_null.transform(train)
#train.printSchema()


# In[6]:


from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler

M_features = [s for s in categorical_output if "M" in s]
C_features = [s for s in numerical if "C" in s]
D_features = [s for s in numerical if "D" in s]
V_features = [s for s in numerical if "V" in s]
id_num = [s for s in numerical if "id" in s]
id_cat = [s for s in categorical_output if "id" in s]

all_pca = M_features + C_features + D_features + V_features + id_num + id_cat
all_features = categorical_output + numerical
kept_features = list(set(all_features) - set(all_pca + ['TransactionID']))

assembler_M = VectorAssembler(inputCols=M_features,outputCol="M_features")
pca_M = PCA(k=1, inputCol="M_features", outputCol='pca_M')

assembler_C = VectorAssembler(inputCols=C_features,outputCol="C_features")
pca_C = PCA(k=1, inputCol="C_features", outputCol='pca_C')

assembler_D = VectorAssembler(inputCols=D_features,outputCol="D_features")
pca_D = PCA(k=1, inputCol="D_features", outputCol='pca_D')

assembler_V = VectorAssembler(inputCols=V_features,outputCol="V_features")
pca_V = PCA(k=30, inputCol="V_features", outputCol='pca_V')

assembler_id_num = VectorAssembler(inputCols=id_num,outputCol="id_num")
pca_id_num = PCA(k=1, inputCol="id_num", outputCol='pca_id_num')

assembler_id_cat = VectorAssembler(inputCols=id_cat,outputCol="id_cat")
pca_id_cat = PCA(k=1, inputCol="id_cat", outputCol='pca_id_cat')

pca_features = ['pca_M', 'pca_C', 'pca_D','pca_V','pca_id_num','pca_id_cat']
final_features = pca_features + kept_features
assembler_final = VectorAssembler(inputCols=final_features,outputCol="features")


# In[7]:


# from pyspark.ml.classification import LogisticRegression
# lr = LogisticRegression(labelCol="isFraud", featuresCol="features")

from pyspark.mllib.classification import SVMWithSGD
svm = SVMWithSGD.train(features, iterations=100, step=1.0, miniBatchFraction=1.0, regParam=0.01, regType="l2")





# In[8]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[numerical_transform, indexer, categorical_fill_null,
                           assembler_M, assembler_C, assembler_D, assembler_V, assembler_id_num, assembler_id_cat,
                           pca_M, pca_C, pca_D, pca_V, pca_id_num, pca_id_cat,
                           assembler_final,
                           svm])


# In[ ]:


# cross-fold
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

(trainingData, testData) = train.randomSplit([0.7, 0.3])

# grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 2]).build()
grid = ParamGridBuilder().addGrid(svm.iterations, [10, 100, 1000]) \
						.addGrid(svm.step , [0.01, 1, 10]) \
						.addGrid(svm.miniBatchFraction, [0.1, 0.5, 1]) \
						.addGrid(svm.regParam, [0.01, 0.1, 1]) \
						.addGrid(svm.regType, [“l2”, ”l1”, None]).build()
evaluator = MulticlassClassificationEvaluator(labelCol="isFraud", predictionCol="prediction", metricName="accuracy")

cv = CrossValidator(estimator=pipeline, 
           estimatorParamMaps=grid,
           evaluator=evaluator,
           numFolds=5,
           parallelism=2)
cvModel = cv.fit(trainingData) # run the crossValidation and automatically select the best combination of paramerters

prediction = cvModel.transform(testData)
evaluator.evaluate(prediction)
print("Test Accracy = %g" % (accuracy))

# In[ ]:


'''
(trainingData, testData) = train.randomSplit([0.7, 0.3])
model = pipeline.fit(trainingData)
'''


# In[ ]:


'''
predictions = model.transform(testData)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="isFraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accracy = %g" % (accuracy))
'''


# In[ ]:




