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


    
