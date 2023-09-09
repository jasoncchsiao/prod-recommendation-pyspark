#!/bin/env python 
# recommender_prod_rec.py 

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row, DataFrame
from pyspark.conf import SparkConf
from pyspark import SparkContext

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Imputer
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler, MinMaxScaler
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, RowMatrix
from pyspark.mllib.linalg import Vectors, DenseVector
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import OneHotEncoder
import numpy as np
#from haversine import haversine

import abc
import sys
import os
import boto3
from datetime import datetime
from typing import List, Set, Dict
import pandas as pd 
import numpy as np
from collections import defaultdict 
import itertools 
from utilities import Utilities


__all__ = ['GetterSetter', 'ProdRec']

__author__ = 'Jason Hsiao'
__status__ = 'Development'

class GetterSetter:
  __metaclass__ = abc.ABCMeta 

  @abc.abstractmethod
  def set_dataframe(self, inputPath: str, *args, **kwargs):
    """Set a value in the instance."""
    return 

  @abc.abstractmethod 
  def get_dataframe(self):
    """Get and return a value from the instance."""
    return 


class ProdRec(GetterSetter):
  def __init__(self, prodRecHashMap: Dict):
    self.getRecItem_udf = udf(ProdRec.getRecItem, returnType=IntegerType())
    self.getRecRating_udf = udf(ProdRec.getRecRating, returnType=FloatType())
    self.prodRecHashMap = prodRecHashMap

  @staticmethod
  def getRecItem(arr: List) -> int:
    if not arr: return None 
    return arr[0]

  @staticmethod
  def getRecRating(arr: List) -> float:
    if not arr: return None 
    return arr[1]

  def set_dataframe(self, inputPath: str, is_aws: bool=True, is_edh: bool=True) -> None:
    if is_aws:
      self.val = self.get_data_frm_dir(inputPath)
    else:
      if is_edh:
        pass
      else:
        self.val = self.get_data_frm_dir(inputPath, False)

  def get_dataframe(self) -> DataFrame:
    return self.val 

  def get_data_frm_dir(self, src_dir: str, is_aws: bool=True) -> DataFrame:
    def get_prefix(full_path):
      parsed = full_path.split('//')[1].split('/')[1:]
      for idx, ele in enumerate(parsed):
        if all([char.isdigit() for char in ele]):
          cur_date = ele
          date_idx = idx 
      prefix = '/'.join([ele for idx, ele in enumerate(parsed) if idx < date_idx]) + '/'
      suffix = '/'+'/'.join([ele for idx, ele in enumerate(parsed) if idx > date_idx])
      return cur_date, prefix, suffix 

    def get_update_full_path(full_path):
      bucket = full_path.split('//')[1].split('/')[0]
      max_date, prefix, suffix = get_prefix(full_path)
      max_date = datetime.strptime(max_date, '%Y%m%d')
      s3 = boto3.client('s3')
      res = s3.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
      dates = []
      for obj in res.get('CommonPrefixes'):
        date = obj.get('Prefix')
        date = date.replace(prefix,'').split('/')[0]
        if not all([char.isdigit() for char in date]):
          continue 
        dates.append(date)
      max_date = max_date if max_date in dates else sorted(dates)[-1]
      max_date = datetime.strptime(max_date, '%Y%m%d')

      for obj in res.get('CommonPrefixes'):
        cur_date = obj.get('Prefix')
        cur_date = cur_date.replace(prefix, '').split('/')[0]
        if not all([char.isdigit() for char in cur_date]):
          continue 
        cur_date = datetime.strptime(cur_date, '%Y%m%d')
        if cur_date > max_date:
          max_date = cur_date
      max_date = datetime.strftime(max_date, '%Y%m%d')
      update_full_path = 's3a://'+bucket+'/'+prefix+max_date+suffix 
      return update_full_path 
    if is_aws == False:
      pass 
    else:
      dirname = os.path.dirname(src_dir)
      if os.path.exists(dirname):
        pass 
      else:
        os.makedirs(dirname)
        with open(src_dir, 'wt') as fh:
          fh.write('')
      src_dir = get_update_full_path(src_dir)

    if src_dir.find('.txt')==-1 or src_dir.find('.dat')==-1 or src_dir.find('.csv')==-1 or src_dir.find('.parquet')==-1:
      data = spark.read.format('csv') \
                  .option('header','true').option('delimiter','|') \
                  .load(src_dir)
    if src_dir.find('.csv') != -1:
      data = spark.read.format('csv') \
                  .option('header','true').option('inferSchema','true').csv(src_dir)
    if src_dir.find('.parquet') != -1:
      data = spark.read.format('parquet') \
                  .option('header','true') \
                  .load(src_dir)

    return data 

  def write_data_frm_ouptut(self, inputDF, output_dir, output_format: str='parquet',
                            key_word: str='overwrite') -> None:
    if output_format == 'csv':
      inputDF.coalesce(1).write.mode(key_word) \
             .option('header','true') \
             .option('inferSchema','true') \
             .csv(output_dir)
    elif output_format == 'json':
      inputDF.coalesce(1).write.mode(key_word) \
             .option('header','true') \
             .option('inferSchema','true').json(output_dir)
    else:
      inputDF.coalesce(1).write.mode(key_word) \
             .format('parquet').option('header','true').parquet(output_dir)

  def getEci2CidDF(self, raw_pup: DataFrame, raw_eci: DataFrame) -> DataFrame:
    resDF = raw_pup.join(raw_eci, on='eci', how='left') \
                   .drop('eci') \
                   .withColumnRenamed('cid','eci')
    return resDF 

  def getPreprocessDF(self, raw: DataFrame, level: str='eci') -> DataFrame:
    windowSpec = Window.partitionBy('eci','sku').orderBy('primary_intensity_value')
    preprocessDF = \
    raw.select('eci','sku','primary_intensity_value') \
       .na.drop(subset=['eci','sku']) \
       .withColumn('primary_intensity_value', percent_rank().over(windowSpec)) \
       .withColumn('primary_intensity_value', col('primary_intensity_value')+1e-6)
    preprocessDF = preprocessDF.filter(col('primary_intensity_value').isNotNull())
    self.prodRecHashMap['preprocessDF_ROW_COUNT'] = preprocessDF.count()
    return preprocessDF

  def getAccVolDF(self, df: DataFrame) -> DataFrame:
    accDF = \
    df.groupBy('eci','sku').sum('primary_intensity_value') \
      .withColumnRenamed('sum(primary_intensity_value)','primary_intensity_value')
    return accDF

  def get_idxedDF(self, raw_df: DataFrame, col_name: str='idx') -> DataFrame:
    schema = StructType([(StructField(col_name, IntegerType(), False))] + raw_df.schema.fields)
    idxedDF = spark.createDataFrame(raw_df.rdd.zipWithIndex().map(lambda row: Row(**row[0].asDict(),
                                                                                  idx=row[1])), schema)
    return idxedDF 

  def getTrainedDF(self, df: DataFrame, rawDF: DataFrame) -> DataFrame:
    idxCltMap = self.get_idxedDF(df.select('eci').distinct())
    idxProdMap = self.get_idxedDF(df.select('sku').distinct())
    trainedDF = \
    df.join(idxCltMap, on='eci', how='left').drop('eci').withColumnRenamed('idx','userId') \
      .join(idxProdMap, on='sku', how='left').drop('sku').withColumnRenamed('idx','itemId') \
      .withColumnRenamed('primary_intensity_value','rating') \
      .select('userId','itemId','rating')
    idxCltMap = idxCltMap.join(rawDF.select('eci').drop_duplicates(), on='eci', how='left') \
                         .withColumn('idx','userId')
    idxProdMap = idxProdMap.join(rawDF.select('sku','col2','col3','col4').drop_duplicates(),
                                 on='sku', how='left') \
                           .withColumn('idx','itemId')
    return trainedDF, idxCltMap, idxProdMap

  def getscaledDF(self, df: DataFrame, colname: str, strategy: str='minmax') -> DataFrame:
    scaler = MinMaxScaler(inputCol=colname, outputCol='scaled_'+colname)
    pass 

  def als_train(self, df: DataFrame, best_rank: int, best_iter: int, best_regparam: float,
                implicitPrefs: bool, is_grid_search: bool=False):
    ratings = df 
    (training, test) = ratings.randomSplit([1.0, 0.0], seed=0)
    als = ALS(
      rank=best_rank,
      maxIter=best_iter,
      regParam=best_regparam,
      implicitPrefs=implicitPrefs,
      userCol='userId',
      itemCol='itemId',
      ratingCol='rating',
      coldStartStrategy='drop',
      nonnegative=True,
      seed=0
    )
    model = als.fit(training)
    if is_grid_search:
      hf = None
    hf = model.userFactors 
    return hf, model, ratings 

  def model_evaluation(self, model, ratings):
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=0)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName='rmse',
                                    labelCol='rating',
                                    predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    return rmse 

  def grid_search(self, df):
    rank = [15]
    maxIter = [5]
    regParam = [0.001]
    best_rmse = float('inf')
    best_rank, best_iter, best_regparam = None, None, None 
    for r in rank:
      for m in maxIter:
        for p in regParam:
          _, model, ratings = self.als_train(df, r, m, p, True, True)
          rmse = self.model_evaluation(model, ratings)
          if rmse < best_rmse:
            best_rmse = rmse 
            best_rank = r 
            best_iter = m
            best_regparam = p 
    return best_rank, best_iter, best_regparam, best_rmse

  def getProdRec(self, prodRecDF: DataFrame, userMap: DataFrame, itemMap: DataFrame) -> DataFrame:
    prodRecDF = prodRecDF.withColumn('recommendations', explode('recommendations'))
    prodRecDF = prodRecDF.withColumn('itemId', self.getRecItem_udf('recommendations')) \
                         .withColumn('rating', self.getRecRating_udf('recommendations')) \
                         .drop('recommendations')
    resDF = \
    prodRecDF.join(userMap, on='userId', how='left') \
             .join(itemMap, on='itemId', how='left')
    return resDF 




    
