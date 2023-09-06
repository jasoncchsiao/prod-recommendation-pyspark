from pyspark.sql import SparkSession 
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row, DataFrame 
from pyspark.conf import SparkConf 
from pyspark import SparkContext 
from pyspark.sql.window import Window 

from pyspark.ml import Pipeline 
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.recommendation import ALS, ALSModel 
from pyspark.ml.feature import StringIndexer 
from pyspark.ml.feature import VectorAssembler 
from pyspark.ml.feature import StandardScaler, MinMaxScaler 
from pyspark.ml.linalg import Vectors 
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix
from pyspark.mllib.linalg import vectors, DenseVector 

import abc 
import sys 
import boto3
from datetime import datetime 
import pandas as pd 
import numpy as np
import json, pickle 
from collections import defaultdict 
import itertools 

__all__ = ['PostprocProdrecEngine']

__author__ = 'Jason Hsiao'
__version__ = '0.1.2'
__status__ = 'Development'

class PostprocProdrecEngine:
  def __init__(self, peerPath, prodPath, pupPath, itemMapPath):
    self.peers = spark.read.parquet(peerPath)
    self.prod = spark.read.parquet(prodPath)
    self.pup = spark.read.option('header','true') \
                    .option('inferSchema','true') \
                    .csv(pupPath) \
                    .orderBy(col('userId').asc(), col('rating').desc()) \
                    .withColumn('is_used', lit(1))
    self.itemMap = spark.read.option('header','true').option('inferSchema','true') \
                        .csv(itemMapPath)
    self.unlistVec_udf = udf(PostprocProdrecEngine.unlistVec, returnType=FloatType())
    
  @staticmethod
  def unlistVec(row: 'vector') -> float:
    return row.toArray().tolist()[0]

  def getNormalizedFeat(self, colNames: list):
    for i, colName in enumerate(colNames):
      assembler = VectorAssembler(inputCols=[colName], outputCol=colName+'_vec')
      scaler = MinMaxScaler(inputCol=colName+'_vec', outputCol=colName+'_sc')
      pipeline = Pipeline(stages=[assembler, scaler])
      if i == 0:
        normalizedDF = pipeline.fit(self.prod).transform(self.prod)
      else:
        normalizedDF = pipeline.fit(normalizedDF).transform(normalizedDF)
      normalizedDF = normalizedDF \
                      .withColumn(colName+'_sc', self.unlistVec_udf(col(colName+'_sc'))) \
                      .drop(colName+'_vec').drop(colName) \
                      .withColumnRenamed(colName+'_sc', colName)
      normalizedDF.select(colNames).show()
      print(normalizedDF.select(colNames).describe().show())
      return normalizedDF

  def getConfidenceTable(self):
    self.peers.select('tgt_coname','src_coname','score') \
        .withColumnRenamed('tgt_coname','prospect') \
        .withColumnRenamed('src_coname','coname') \
        .join(self.getNormalizedFeat(COLS_TO_NORMALIZE).select('coname','sku','pup_measure_name',
                  'pup_product_sub_family','pup_product_family','rating'),
              on='coname',
              how='right') \
        .select('prospect','coname','score','pup_measure_name','rating') \
        .withColumn('confidence', col('score')*col('rating')/NUM_PEERS) \
        .orderBy(col('prospect').asc(), col('coname').asc(),
                 col('confidence').desc()) \
        .filter(col('prospect').isNotNull()).filter(col('prospect')!='RST') \
        .drop('coname','score','rating') \
        .groupBy('prospect','pup_measure_name').agg({'confidence':'sum'}) \
        .orderBy(col('prospect').asc(), col('sum(confidence)').desc()) \
        .withColumn('rank', rank().over(Window.partitionBy('prospect').orderBy(desc('sum(confidence)')))) \
        .filter(col('rank')<=TOPK_PRODUCTS) \
        .withColumnRenamed('sum(confidence)', 'confidence') \
        .withColumn('pup_measure_name',trim(col('pup_measure_name'))) \
        .join(self.itemMap.drop('itemId').withColumn('pup_measure_name', trim(col('pup_measure_name'))),
              on='pup_measure_name', how='left') \
        .orderBy('prospect','pup_product_family','pup_product_sub_family') \
        .select('prospect','pup_product_family','pup_product_sub_family','pup_measure_name','sku',
                'confidence','rank') \
        .coalesce(1).write.mode('overwrite') \
        .option('header','true').option('inferSchema','true') \
        .csv(confidencePath)

  def getPenetrationTable(self):
    self.peers.select('tgt_coname','src_coname','score') \
        .withColumnRenamed('tgt_coname','prospect') \
        .withColumnRenamed('src_coname','coname') \
        .join(self.prod.select('coname','sku','pup_measure_name',
                               'pup_product_sub_family','pup_product_family','rating'),
              on='coname',
              how='right') \
        .select('prospect','coname','score','pup_measure_name','rating') \
        .orderBy(col('prospect').asc(), col('coname').asc()) \
        .filter(col('prospect').isNotNull()).filter(col('prospect')!='RST') \
        .drop('coname','score') \
        .filter((col('rating').isNotNull())|(col('rating')!=0)) \
        .groupBy('prospect','pup_measure_name').agg({'rating': 'count'}) \
        .withColumn('penetration',col('count(rating)')/NUM_PEERS).drop('count(rating)') \
        .orderBy(col('prospect').asc(), col('penetration').desc()) \
        .withColumn('rank',rank().over(Window.partitionBy('prospect').orderBy('penetration')))) \
        .filter(col('rank')<=TOPK_PRODUCTS) \
        .withColumn('pup_measure_name', trim(col('pup_measure_name'))) \
        .orderBy('prospect','pup_product_family','pup_product_sub_family') \
        .select('prospect','pup_product_family','pup_product_sub_family','pup_measure_name','sku',
                'penetration','rank') \
        .coalesce(1).write.mode('overwrite') \
        .option('header','true').option('inferSchema','true') \
        .csv(penetrationPath)

  def main(self):
    self.getConfidenceTable()
    self.getPenetrationTable()


if __name__ == '__main__':
  from pyspark.sql import SparkSession 
  from pyspark.sql.functions import *
  from pyspark.sql.types import *
  from pyspark.sql import Row, DataFrame 
  from pyspark.conf import SparkConf 
  from pyspark import SparkContext 
  from pyspark.sql.window import Window 
  
  from pyspark.ml import Pipeline 
  from pyspark.ml.evaluation import RegressionEvaluator 
  from pyspark.ml.recommendation import ALS, ALSModel 
  from pyspark.ml.feature import StringIndexer 
  from pyspark.ml.feature import VectorAssembler 
  from pyspark.ml.feature import StandardScaler, MinMaxScaler 
  from pyspark.ml.linalg import Vectors 
  from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix
  from pyspark.mllib.linalg import vectors, DenseVector 
  
  import abc 
  import sys 
  import boto3
  from datetime import datetime 
  import pandas as pd 
  import numpy as np
  import json, pickle 
  from collections import defaultdict 
  import itertools 

  isLocal = False 
  isEMR = False 
  is_edh = False 
  is_aws = False 
  t_start = datetime.now()
  writeDate = '20200101'
  NUM_PEERS = 30
  TOPK_PRODUCTS = 20

  # kms_key = '.....'
  # bucket = '....'
  # bucket2 = '...'

  if isLocal:
    conf = (SparkConf() \
            .set('spark.executor.extraJavaOptions','-Dorg.xerial.snappy.tempdir=/opt/jupyter') \
            .set('spark.driver.extraJavaOptions','-Dorg.xerial.snappy.tempdir=/opt/jupyter'))
    sc = SparkContext(conf=conf)
    spark = SparkSession \
            .builder \
            .config(conf=conf) \
            .appName('Postproc') \
            .getOrCreate()

  else:
    if isEMR:
      conf = (SparkConf() \
              .set("fs.s3a.server-side-encryption-algorithm", "SSE-KMS") \
              .set("fs.s3a.server-side-encryption-key", kms_key))
      sc = SparkContext(conf=conf)
      spark = SparkSession.builder.config(config=config).appName('PostProc').getOrCreate()

    else:
      conf = (SparkConf() \
              .set("fs.s3a.server-side-encryption-algorithm", "SSE-KMS") \
              .set("fs.s3a.server-side-encryption-key", kms_key) \
              .set('spark.executor.extraJavaOptions','-Dorg.xerial.snappy.tempdir=/opt/jupyter') \
              .set('spark.driver.extraJavaOptions','-Dorg.xerial.snappy.tempdir=/opt/jupyter') \
              .setMaster('local')
      spark = SparkSession.builder.config(config=config).getOrCreate()
      sc = SparkContext.getOrCreate(conf=conf)

  if isLocal:
    peerPath = 'prod-rec/'+writeDate+'/peer_results/1/'
    prodPath = 'product-recommendation/prod_rec/'+writeDate+'/'
    pupPath = 'product-recommendation/pup_edited/'+writeDate+'/'
    itemMapPath = 'product-recommendation/itemMap/'+writeDate+'/'
    confidencePath = 'product-recommendation/prod_confidence_table/'+writeDate+'/'
    penetrationPath = 'product-recommendatoin/prod_penetration_table/'+writeDate+'/'
  else:
    if isEMR:
      bucket = sys.argv[1]
      bucket2 = sys.argv[2]
      writeDate = sys.argv[3]

    peerPath = 'prod-rec/'+writeDate+'/peer_results/1/'
    prodPath = 'product-recommendation/prod_rec/'+writeDate+'/'
    pupPath = 'product-recommendation/pup_edited/'+writeDate+'/'
    itemMapPath = 'product-recommendation/itemMap/'+writeDate+'/'
    confidencePath = 's3a://'+bucket+'/test/prod-rec/postprocessing/prod_confidence_table/'+writeDate+'/pct/'
    penetrationPath = 's3a://'+bucket+'/test/prod-rec/postprocessing/prod_penetration_table/'+writeDate+'/ppt/'

  COLS_TO_NORMALIZE = ['rating']
  ppe = PostprocProdrecEngine(peerPath, prodPath, pupPath, itemMapPath)
  ppe.main()
  spark.stop()
  print("total time to compute=".upper(), datetime.now() - t_start)
