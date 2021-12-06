
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext
import json
import numpy as np
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

sc = SparkContext()#conf=SparkConf().setAppName("MyApp").setMaster("local"))
spark = SparkSession(sc)
ssc = StreamingContext(sc, batchDuration= 10)


def f(x):
    global counter
    global features
    global all_records
    data = json.loads(x)
    for key,values in data.items():
        temp = list(values.values())
        all_records.append(temp)
        counter+=1
        if len(features) == 0:
            features = list(values.keys())
    all_records = np.asarray(all_records)
    '''
    df = pd.DataFrame(all_records, columns=features)
    mydf = spark.createDataFrame(df)
    print(mydf.show())
    '''
    

def process_stream(record,spark):
    if not record.isEmpty():
        #record.foreach(f)
        #print(str(counter)+" ==== "+str(all_records))
        x = record.take(100)
        x = x[0]
        all_records = []
        features = []
        data = json.loads(x)
        for key,values in data.items():
            temp = list(values.values())
            all_records.append(temp)
            if len(features) == 0:
                features = list(values.keys())
        all_records = np.asarray(all_records)
        df = pd.DataFrame(all_records, columns=features)
        mydf = spark.createDataFrame(df)
        assembler = VectorAssembler(inputCols=features, outputCol='features',handleInvalid="skip")
        transformed_data = assembler.transform(mydf)

        indexer = StringIndexer(inputCol="label",outputCol="indexlabel",handleInvalid="skip")
        transformed_data = indexer.fit(transformed_data).transform(transformed_data)

        (training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])
        #creating random forest ML object and training on train data
        decision_tree = RandomForestClassifier(featuresCol = 'features',labelCol = 'indexlabel', numTrees=500)
        decision_tree_model = decision_tree.fit(training_data)
        #predicting on test data
        predictions = decision_tree_model.transform(test_data)
        #calculating accuracy
        evaluator = MulticlassClassificationEvaluator(labelCol='indexlabel',metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("\nRandom Forest Classification Test Error = %g" % (1.0 - accuracy))
        print("\nRandom Forest Accuracy = %g" % (accuracy))
        

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']

def process(time, rdd):
    print("========= %s =========" % str(time))
    spark = getSparkSessionInstance(rdd.context.getConf())
    rowRdd = rdd.map(lambda w: Row(word=w))
    
           

# Create a DStream that will connect to hostname:port, like localhost:9991
lines = ssc.socketTextStream("localhost", 6100)
lines.foreachRDD(lambda rdd: process_stream(rdd,spark))
#print(str(all_records.shape)+" ===== "+str(len(features)))
ssc.start()
ssc.awaitTermination()
