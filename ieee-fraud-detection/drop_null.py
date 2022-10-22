import os.path

current_path = %pwd
filePath = os.path.join(os.path.abspath(os.path.dirname(current_path) + os.path.sep + "."),'ieee-fraud-detection')
    
fileName_train_trans = 'train_transaction.csv'
fileDir_train_trans = os.path.join(filePath, fileName_train_trans)

fileName_train_id = 'train_identity.csv'
fileDir_train_id = os.path.join(filePath, fileName_train_id)
    
df = spark.read.options(header='true').csv(fileDir_train_trans)
df2 = spark.read.options(header='true').csv(fileDir_train_id)
train = df.join(df2, df.TransactionID == df2.TransactionID)
train.drop('TransactionID')

missing = train.select([count(when(col(c).isNull(), c)).alias(c) for c in train.columns])
null_counts = missing.collect()[0].asDict()
to_drop = [k for k, v in null_counts.items() if v > (590540*0.8)]
len(to_drop)
train.drop(*to_drop)
