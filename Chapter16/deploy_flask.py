from flask import Flask
from flask import request
from pyspark.ml import PipelineModel
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.context import SparkContext

sc = SparkContext('local', 'test')
sql = SQLContext(sc)

app = Flask(__name__)

loaded_model = PipelineModel.load('/tmp/linear-model')

schema = StructType([StructField('crim', DoubleType(), True),
                     StructField('zn', DoubleType(), True),
                     StructField('indus', DoubleType(), True)])

@app.route('/predict', methods=['GET'])
def predict():
    crim = float(request.args.get('crim'))
    zn = float(request.args.get('zn'))
    indus = float(request.args.get('indus'))
    predict_df = sql.createDataFrame([Row(crim=crim, zn=zn, indus=indus)],schema=schema) 
    prediction = loaded_model.transform(predict_df).collect()[0].prediction
    return str(prediction)

