#!/usr/bin/env python
# search_ranking_peers.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import Row, DataFrame
from pyspark.conf import SparkConf
from pyspark import SparkContext

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

__all__ = ['GetterSetter', 'SearchRankPeers']
__author__ = 'Jason Hsiao'
__version__ = '0.6.4'
__status__ = 'Development'

######################################################################################
### introduce jpmc-division around 20 categories to mitigate the overweighted Hops ###
### changes from output: added jpmc_division #########################################
### under experiments: best weighting so far #########################################
### w_rev = 0.6; w_age = w_pub = w_emp = 0.05; w_ult=0; w_jpmc = 0.25#################
### overall weighting, W_numerical = 0.7; W_hops = 0.2; W_havs = 0.1 #################
### refactored the main function specifically ########################################
### added s3 read/write capability ###################################################
### avaoid importing haversine use numpy direct implementation instead ###############
### incorporate get_data_from_dir func to get the latest foloer in s3 ################
######################################################################################

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

class SearchRankPeers(GetterSetter):
    def __init__(self):
        self.unlistVec_udf = udf(SearchRankPeers.unlistVec, returnType=FloatType())
        self.toDense_udf = udf(SearchRankPeers.toDense, returnType=ArrayType(FloatType()))

    @staticmethod
    def unlistVec(row: 'vector') -> float:
        return row.toArray().tolist()[0]

    @staticmethod
    def toDense(v: 'vector') -> List:
        v = DenseVector(v)
        new_array = list([float(x) for x in v])
        return new_array

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
        """
        @src_dir: str, any given directory for the source data
        @is_aws: bool, if program is running on AWS or not
        """
        def get_prefix(full_path: str):
            parsed = full_path.split('//')[1].split('/')[1:]
            for idx, ele in enumerate(parsed):
                if all([char.isdigit() for char in ele]):
                    cur_date = ele
                    data_idx = idx
            prefix = '/'.join([ele for idx, ele in enumerate(parsed) if idx < date_idx])+'/'
            suffix = '/'+'/'.join([ele for idx, ele in enumerate(parsed) if idx > date_idx])

            return cur_date, prefix, suffix

        def get_update_full_path(full_path: str):
            """
            ex. s3://app-id-..xxxxxxxx/data/raw/eci/20200315/eci/*.parquet
            @full_path: str, s3://bucket/prefix/20200325/eci/*.parquet
            @return:    str, s3://bucket/prefix/20200415/eci/*.parquet
            """
            bucket = full_path.split('//')[1].split('/')[0]
            max_date, prefix, suffix = get_prefix(full_path)
            max_date = datetime.strptime(max_date, '%Y%m%d')
            s3 = boto3.client('s3')
            res = s3.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')
            #######################################################################
            dates = []
            for obj in res.get('CommonPrefixes'):
                date = obj.get('Prefix')
                date = date.replace(prefix, '').split('/')[0]
                if not all([char.isdigit() for char in date]):
                    continue
                dates.append(date)

            max_date = max_date if max_date in dates else sorted(dates)[-1]
            max_date = datetime.strptime(max_date, '%Y%m%d')
            #######################################################################
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
                #dirname = os.path.dirname(src_dir)
                os.makedirs(dirname)
                with open(src_dir, 'wt') as fh:
                    fh.write('')

            src_dir = get_update_full_path(src_dir)

        if src_dir.find('.txt')==-1 or src_dir.find('.dat')==-1 or src_dir.find('.csv')==-1 or src_dir.find('.parquet')==-1:
            data = spark.read.format('csv') \
                        .option('header','true').option('delimiter','|') \
                        .load(src_dir)
        if src_dir.find('.csv')!=-1:
            data = spark.read.format('csv') \
                        .option('header','true').option('inferSchema','true') \
                        .csv(src_dir)
        if src_dir.find('.parquet') !=-1:
            data = spark.read.format('parquet') \
                        .option('header','true') \
                        .load(src_dir)

        return data

    def write_data_frm_output(self, inputDF: DataFrame, output_dir: str,
                              output_format: str='parquet', keyword: str='overwrite')-> None:
        if output_format != 'parquet':
            inputDF.coalesce(1).write.mode(key_word) \
                   .option('header','true') \
                   .option('inferSchema','true').csv(output_dir)
        else:
            inputDF.write.mode(key_word).format('parquet') \
                   .option('header','true').parquet(output_dir)

    def get_dedupDF(self, inputDF: DataFrame)-> DataFrame:
        col_names = inputDF.columns
        collectmap = {k: v for k, v in zip([col_name for col_name in col_names
                      if col_name not in ['cid']],
                      ['first' for _ in range(len(col_names)-1)])}
        inputDF = inputDF.groupBy('cid').agg(collectmap)
        for col_name in inputDF.columns:
            if col_name not in ['cid']:
                inputDF = inputDF.withColumnRenamed(col_name, col_name.split('(')[-1][:-1])

        return inputDF

    def drop_na(self, inputDF: DataFrame) -> DataFrame:
        return inputDF.na.drop()

    def get_tgt_src_split(self, inputDF: DataFrame, isRandom: bool=False):
        source, target = None, None
        if isRandom:
            source, target = inputDF.randomSplit([0.9,0.1], seed=100)
        else:
            pass
        return source, target

    def get_idxedDF(self, raw_df: DataFrame, col_name: str='idx')-> DataFrame:
        schema= StructType([(StructField(col_name, IntegerType(), False))] + raw_df.schema.fields)
        idxedDF = spark.createDataFrame(raw_df.rdd.zipWithIndex().map(lambda row: Row(**row[0].asDict(),
                                        idx=row[1])), schema)
        return idxedDF

    def get_datatransform(self, inputDF: DataFrame)-> DataFrame:
        pass

    def df2mat_mdf_normalized(self, df_mdf):
        features_d = df_mdf.select('features') \
                           .rdd.map(lambda x: x[0]).map(lambda x: x.toArray()/x.norm(2)).collect()
        idxedRows = sc.parallelize([(i, vec) for i, vec in enumerate(features_d)])
        idxRMat = IndexedRowMatrix(rows=idxedRows)
        return idxRMat

    def df2mat_mdf_normalized_big(self, df_mdf):
        from pyspark.mllib.linalg.distribtued import IndexedRow, IndexedRowMatrix, RowMatrix
        from pyspark.mllib.linalg import Vectors, DenseVector
        idxedRows = df_mdf.select('idx','features') \
                          .rdd.map(lambda x: IndexedRow(x[0],
                                   DenseVector(Vectors.fromML(x[1]))/x[1].norm(2)))
        idxRMat = IndexedRowMatrix(rows=idxedRows)
        return idxRMat

    def getHavs(cols_lat: str, col1_lon: str, col2_lat: List[], col2_lon: List[])-> List[]:
        def havs_score(lat1: float, lon1: float, lat2: float, lon2: float)-> float:
            if not lat1 or not lon1 or not lat2 or not lon2:
                return 0.5
            return 1.0 - haversine((lat1, lon1), (lat2, lon2), unit='km') / 20020
        return [havs_score(col1_lat, col1_lon, col2_lat[i], col2_lon[i])
                for i in range(len(col2_lat))]

    def one_batch(self, leftData: DataFrame, rightData: DataFrame,
                  left_naics: list, right_naics: list,
                  left_lats: list, left_lons: list,
                  right_lats: list, right_lons: list, batch_num: str):
        leftMat = self.df2mat_mdf_normalized_big(leftData)
        rightMat = self.df2mat_mdf_normalized(rightData.select('idx','features'))
        local_dm = rightMat.toBlockMatrix().transpose().toLocalMatrix()
        bcast_local_dm = sc.broadcast(local_dm)
        matmul = leftMat.multiply(bcast_local_dm.value)
        cs = matmul.rows
        #############################################################################
        ############## nested helper function #######################################
        def find_matches_in_record(record: 'spark.rdd', threshold: float=.75,
                                   topK: int=2, w_hops: float=.5, w_havs: float=.1):
            def havs_score(lat1: flaot, lon1: float, lat2: float, lon2: float)-> float:
                if not lat1 or not lon1 or not lat2 or not lon2:
                    return 0.5
                def havs_distance(lat1, lon1, lat2, lon2):
                    R = 6373.0
                    lat1 = lat1*np.pi/180.0
                    lon1 = np.deg2rad(lon1)
                    lat2 = np.deg2rad(lat2)
                    lon2 = np.deg2rad(lon2)
                    d = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2-lon1)/2)**2
                    return float(2*R*np.arcsin(np.sqrt(d)))

                return 1.0 - havs_distance(lat1, lon1, lat2, lon2) / 20020

            def hops_dist(s1: str, s2: str):
                if len(s1)!=6 or len(s2)!=6:
                    return -1
                l, r = 0, 0
                if s1[l] != s2[r]:
                    return 0
                while l < len(s1) and r < len(s2):
                    if s1[l] == s2[r]:
                        l += 1
                        r += 1
                    else:
                        return 1 - ((len(s1)-1)*2/10)
                return 1

            def getTopK(tups, topK):
                def quickSelect(tups, topK, start, end):
                    if start == end: return tups[start]
                    left, right = start, end
                    pivot = tups[(start+end)//2][-1]
                    while left <= right:
                        while left <= right and tups[left][-1] > pivot:
                            left += 1
                        while left <= right and tups[right][-1] < pivot:
                            right -= 1
                        if left <= right:
                            tups[left], tups[right] = tups[right], tups[left]
                            left += 1
                            right -=1
                    if start + topK - 1 <= right:
                        return quickSelect(tups, topK, start, right)
                    if start + topK - 1 >= left:
                        return quickSelect(tups, topK-(left-start), left, end)
                    return tups[right+1]

                if not tups: return None
                quickSelect(tups, topK, 0, len(tups)-1)
                res = tups[:topK]
                # res = sorted(res, key=lambda x: x[-1], reverse=True)
                return res
            #####################################################################################
            tgt_idx = int(record.index)
            tgt_naics = left_naics[tgt_idx]
            tgt_lats = left_lats[tgt_idx]
            tgt_lons = left_lons[tgt_idx]
            hops = [hops_dist(tgt_naics, right_naics[i]) for i in range(len(right_naics))]
            havs = [havs_score(tgt_lats, tgt_lons, right_lats[i], right_lons[i])
                    for i in range(len(right_naics))]
            cosim = record.vector.toArray().tolist()
            pairs = [(i, ((1-w_hops-w_havs)*cos+w_hops*hop+w_havs*hav))
                     for i, (cos, hop, hav) in enumerate(zip(cosim, hops, havs))
                     if ((1-w_hops-w_havs)*cos+w_hops*hop+w_havs*hav) >= threshold]
            pairs_topK = getTopK(pairs, topK)
            for (idx, score) in pairs:
                if (idx, score) in pairs_topK:
                    yield (tgt_idx, idx, score)

        ####################################################################################
        matches = cs.flatMap(lambda x: find_matches_in_record(x, threshold=th,
                                                              topK=topK,
                                                              w_hops=w_hops,
                                                              w_havs=w_havs),
                             preservesPartitioning=True)
        pdf = matches.toDF(schema=['tgt_idx', 'src_idx', 'score'])
        return pdf

    def getPeerList(self, srcDF, tgtDF, matchDF, batch_num: str)-> DataFrame:
        COLS = ['idx','eci','cid','coname','annual_revenue',
                'company_age','L6','employees','public_flag','city','state',
                'jpmc_division','jpmc_class']
        source = srcDF.select(COLS)
        target = tgtDF.select(COLS)
        for name in COLS:
            source = source.withColumnRenamed(name, 'tgt_'+name)
            target = target.withColumnRenamed(name, 'src_'+name)
        resDF = matchDF.join(source, on='tgt_idx', how='left') \
                       .join(broadcast(target), on='src_idx', how='left')
        resDF = resDF.orderBy(col('tgt_coname').asc(), col('score').desc())
        self.write_data_frm_output(resDF, peerBatchPath+batch_num+'/','csv')
        return resDF

    def filterRawDF(self, rawDF: DataFrame)-> DataFrame:
        resDF = rawDF.select(DESIRED_COLS) \
                     .filter(col('coname')!='RST') \
                     .filter(col('international')==0) \
                     .withColumn('annual_revenue', col('annual_revenue').cast(FloatType())) \
                     .withColumn('lat',col('lat').cast(FloatType())) \
                     .withColumn('lon',col('lon').cast(FloatType()))
        return resDF

    def feat_imputer(self, inputDF: DataFrame, col_name: str, strategy: str='median')-> DataFrame:
        """impute column has to change to double dtype first
           strategies include mode/median/mean"""
        inputDF = self.get_dedupDF(inputDF) \
                      .withColumn(col_name, col(col_name).cast(DoubleType()))
        imputer = Imputer(strategy=strategy, inputCols=[col_name], outputCols=[col_name])
        resDf = imputer.fit(inputDF).transform(inputDF)
        return resDF

    def get_string_indexer(self, inputDF: DataFrame, COLS_FOR_STRINGINDEX: list)-> DataFrame:
        for i, name in enumerate(COLS_FOR_STRINGINDEX):
            if i == 0:
                indexer = StringIndexer(inputCol=name, outputCol=name+'_cat',
                                        handleInvalid='keep')
                resDF = indexer.fit(inputDF).transform(inputDF)
            else:
                indexer = StringIndexer(inputCols=name, outputCol=name+'_cat',
                                        handleInvalid='keep')
                resDF = indexer.fit(resDF).transform(resDF)
        return resDF

    def get_bucketized(self, inputDF: DataFrame, COLS_FOR_STRINGINDEX: list)-> DataFrame:
        _MAXS = [inputDF.select(max(col(name+'_cat'))).collect()[0][0] for name in COLS_FOR_STRINGINDEX]
        pairs = [(name, _max) for name, _max in
                 zip(COLS_FOR_STRINGINDEX, _MAXS)]
        for i, (name, _max) in enumerate(pairs):
            if i == 0:
                bucketizer = Bucketizer(splits=np.linspace(0, _max+1, 9).tolist(),
                                        inputCol=name+'_cat', outputCol=name+'_buc',
                                        handleInvalid='keep')
                resDF = bucketizer.transform(inputDF)
            elif i!= len(pairs)-1:
                bucketizer = Bucketizer(splits=np.linspace(0, _max+1, 20).tolist(),
                                        inputCol=name+'_cat', outputCol=name+'_buc',
                                        handleInvalid='keep')
                resDF = bucketizer.transform(resDF)
            else:
                bucketizer = Bucketizer(splits=np.linspace(0, _max+1, 4).tolist(),
                                        inputCol=name+'_cat', outputCol=name+'_buc',
                                        handleInvalid='keep')
                resDF = bucketizer.transform(resDF)
        return resDF

    def get_one_hot_encoding(self, inputDF: DataFrame, OH_COLS: list)-> DataFrame:
        for i, name in enumerate(OH_COLS):
            if i == 0:
                ohencoder = OneHotEncoder(inputCol=name+'_cat', outputCol=name+'_ohe')
                resDF = ohencoder.transform(inputDF)
            else:
                ohencoder = OneHotEncoder(inputCol=name+'_cat', outputCol=name+'_ohe')
                resDF = ohencoder.transform(resDF)
        return resDF

    def train_jpmc_ohe_feature(self, inputDF: DataFrame, hierarchy: str='jpmc_division')->DataFrame:
        testDF = inputDF.select('cid',hierarchy+'_ohe')
        num_jpmc = testDF.select(hierarchy+'_ohe').distinct().count()
        testDF = testDF.withColumn(hierarchy+'_ohe', self.toDense_udf(hierarchy+'_ohe'))
        testDF = testDF.select(['cid']+[testDF[hierarchy+'_ohe'][i] for i in range(num_jpmc)])
        for i, name in enumerate(testDF.columns):
            if name != 'cid':
                testDF = testDF.withColumn(name, w_jpmc*col(name)) \
                               .withColumnRenamed(name, 'jpmc_'+str(i))
        JPMC_COLS = [val for val in testDF.columns if val != 'cid']
        resDF = inputDF.join(testDF, on='cid', how='inner').drop(hierarchy+'_ohe')

        return resDF, JPMC_COLS

    def get_feature_weights(self, inputDF: DataFrame)-> DataFrame:
        inputDF = inputDF.withColumn('annual_revenue', when((col('annual_revenue')<=0.0)
                                                            |(col('annual_revenue').isNull()), lit(1.0))
                                                      .otherwise(col('annual_revenue'))) \
                         .withColumn('annual_revenue', log10(col('annual_revenue')))
        for feat in FEATURE_SET:
            assembler = VectorAssembler(inputCols=[feat], outputCol=feat+'_vec')

            # minmaxscaler transformation
            if feat == 'annual_revenue':
                # scaler = StandardScaler(inputCol=feat+'_vec', outputCol=feat+'_sc')
                scaler = MinMaxScaler(inputCol=feat+'_vec', outputCol=feat+'_sc')
            else:
                scaler = MinMaxScaler(inputCol=feat+'_vec', outputCol=feat+'_sc')

            # Pipeline of vectorassembler and minmaxscaler
            pipeline = Pipeline(stages=[assembler, scaler])

            # Fitting pipeline on dataframe
            inputDF = pipeline.fit(inputDF).transform(inputDF)
            inputDF = inputDF \
                            .withColumn(feat+'_sc', self.unlistVec_udf(col(feat+'_sc'))) \
                            .drop(feat+'_vec').drop(feat) \
                            .withColumnRenamed(feat+'_sc', feat)
        w_age = w_pub = w_emp = (1-w_rev-w_jpmc)/3; w_ult = 0; #w_jpmc=0.25
        resDF = inputDF.withColumn('annual_revenue', w_rev*col('annual_revenue')) \
                       .withColumn('company_age', w_age*col('company_age')) \
                       .withColumn('ult', w_ult*col('ult')) \
                       .withColumn('public_flag', w_pub*col('public_flag')) \
                       .withColumn('employees', w_emp*col('employees'))
        return resDF

    def get_assembled_feats(self, inputDF: DataFrame, JPMC_COLS: list)-> DataFrame:
        assembler = VectorAssembler(inputCols=FEATURE_SET+JPMC_COLS,
                                    outputCol='features',
                                    handleInvalid='keep')
        resDF = assembler.transform(inputDF)
        return resDF

    def main(self):
        self.set_dataframe(firmographics_path, is_aws, is_edh)
        rawDF = self.get_dataframe()
        rightDF = self.filterRawDF(rawDF)
        self.set_dataframe(targetList_path, is_aws, is_edh)
        rawDF = self.get_dataframe()
        leftDF = self.filterRawDF(rawDF)
        spry = spark.createDataFrame([('0223948678','1016793652','Spry Methods Inc',12700000.0,1,
                                       0,20,'334614',0,83,
                                       'McLean',upper(),'VA',38.9436,-77.1943,
                                       'Technology, Media & Telecommunications','Media')],
                                       DESIRED_COLS)
        leftDF = leftDF.unionByName(spry) \
                       .withColumn('lat',col('lat').cast(FloatType())) \
                       .withColumn('lon',col('lon').cast(FloatType()))

        COLS_TO_IMPUTE = ['employees','lat','lon']
        for col_name in COLS_TO_IMPUTE:
            rightDF = self.feat_imputer(rightDF, col_name)
            leftDF = self.feat_imputer(leftDF, col_name)

        #####################################################################
        FIXED_COLS = rightDF.columns
        leftDF = leftDF.select(FIXED_COLS)
        leftDF = leftDF.na.drop()
        rightDF = rightDF.na.drop()

        overall = self.get_dedupDF(leftDF.unionByName(rightDF))

        source = self.get_idxedDF(leftDF)
        self.write_data_frm_output(source, leftDFPath, 'csv')
        batches = []
        for idx, batch in enumerate(rightDF.randomSplit([1.0/MIN_BACTH
                                                        for _ in range(MIN_BACTH)], seed=101), 1):
            tmp = self.get_idxedDF(batch)
            tmp = tmp.select(['idx']+FIXED_COLS)
            self.write_data_frm_output(tmp, rightDFPath+str(idx)+'/', 'csv')
            batches.append(tmp)

        ##### change some datatypes in columns#################################################
        sample = \
        overall.withColumn('annual_revenue', col('annual_revenue').cast(FloatType())) \
               .withColumn('ult', col('ult').cast(IntegerType())) \
               .withColumn('public_flag', col('public_flag').cast(IntegerType())) \
               .withColumn('employees', col('employees').cast(IntegerType()))
        ####################################################################################
        ######### StringIndexing ###########################################################
        print('STARTING FEATURE ENGINEERING')
        COLS_FOR_STRINGINDEX = ['L6','jpmc_division']
        indexed = self.get_string_indexer(sample, COLS_FOR_STRINGINDEX)
        ####################################################################################
        ################bucketizing attributes #############################################
        bucketed = self.get_bucketized(indexed, COLS_FOR_STRINGINDEX)
        ####################################################################################
        #####################OneHotEncoder #################################################
        OH_COLS = ['jpmc_division']
        ohencoded = self.get_one_hot_encoding(bucketed, OH_COLS)
        ####################################################################################
        ########################JPMC-industry-divion #######################################
        ohencoded, JPMC_COLS = self.train_jpmc_ohe_feature(ohencoded)
        ####################################################################################
        ######################## put weights on numerical features #########################
        ohencoded = self.get_feature_weights(ohencoded)
        ####################################################################################
        ######################## vector assembling #########################################
        output = self.get_assembled_feats(ohencoded, JPMC_COLS)
        ####################################################################################
        print('DONE FEATURE ENGINEERING')

        ############### output to join back source/target w/ features
        source = source.join(output.select('cid','features'), on='cid', how='inner')
        bnews = []
        for b in batches:
            tmp = b.join(output.select('cid','features'), on='cid', how='inner')
            bnews.append(tmp)
        print('complete join back src/tgt files')
        ################## dataframe trimming for matmul ###################################
        ####################################################################################
        df1 = source.select('idx', 'features')
        for i, b in enumerate(bnews, 1):
            b_trim = b.select('idx','features')
            left_naics = [val for sublist in source.select('L6').collect()
                          for val in sublist]
            right_naics = [val for sublist in b.select('L6').collect()
                           for val in sublist]
            left_lats = [val for sublist in source.select('lat').collect()
                         for val in sublist]
            left_lons = [val for sublist in source.select('lon').collect()
                         for val in sublist]
            right_lats = [val for sublist in b.select('lat').collect()
                          for val in sublist]
            right_lons = [val for sublist in b.select('lon').collect()
                          for val in sublist]
            pdf = self.one_batch(df1, b_trim, left_naics, right_naics,
                                 left_lats, left_lons, right_lats,
                                 right_lons, str(i))
            print("DOING MATMUL PROCESS")
            pdf.show(10)
            resDF = self.getPeerList(source, b, pdf, str(i))
            if i == 1:
                accDF = resDF
            else:
                accDF = accDF.unionByName(resDF)

        accDF = \
        accDF.orderBy(col('tgt_coname').asc(), col('score').desc()) \
             .withColumnRenamed('tgt_L6', 'tgt_naics') \
             .withColumnRenamed('src_L6', 'src_naics') \
             .select('tgt_eci','tgt_cid','tgt_coname','src_eci','src_cid','src_coname',
                     'tgt_annual_revenue','src_annual_revenue','tgt_company_age','src_company_age',
                     'tgt_employees','src_employees','tgt_public_flag','src_public_flag',
                     'tgt_naics','src_naics','tgt_jpmc_division','src_jpmc_division',
                     'tgt_jpmc_class','src_jpmc_class','tgt_city','src_city',
                     'tgt_state','src_state','score')
        self.write_data_frm_output(accDF, reportPath, 'csv')

if __name__ == '__main__':
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.sql import Row, DataFrame
    from pyspark.conf import SparkConf
    from pyspark import SparkContext

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

    t_start = datetime.now()
    isLocal = False # True
    isEMR = True # False
    is_edh = False
    is_aws = True
    isBatchRun = True
    MIN_BATCH = 1 # from 5 to 52

    w_hops = 0.399
    w_havs = 0.001
    w_jpmc = 0.05
    w_rev = 0.80
    th = 0.0
    topK = 15
    DESIRED_COLS = ['eci','cid','coname','annual_revenue','ult','public_flag',
                    'company_age','L6','international','employees',
                    'city','state','lat','lon','jpmc_division','jpmc_class']
    FEATURE_SET = ['annual_revenue','company_age','ult','employees','public_flag']
    # numerical features to learn in cosine similarity
    kms_key = '*******************'
    if isLocal:
        conf = (SparkConf() \
                .set('spark.executor.extraJavaOptions','-Dorg.xerial.snappy.tempdir=/opt/omniai/work/instance1/jupyter') \
                .set('spark.driver.extraJavaOptions','-Dorg.xerial.snappy.tempdir=/opt/omniai/work/instance1/jupyter'))
        sc=SparkContext(conf=conf)
        spark = SparkSession \
                .builder \
                .config(conf=conf) \
                .appName('SearchRankPeers') \
                .getOrCreate()
    else:
        if isEMR:
            conf = (SparkConf() \
                    .set('fs.s3a.server-side-encryption-algorithm', 'SSE-KMS') \
                    .set('fs.s3a.server-side-encryption-key', kms_key))
            sc=SparkContext(conf=conf)
            spark = SparkSession.builder.config(conf=conf).appName('SearchRankPeers').getOrCreate()
        else:
            conf = SparkConf()
            conf.setMaster('local')

            spark = SparkSession \
                        .builder \
                        .config('spark.executor.extraJavaOptions',
                                '-Dorg.xerial.snappy.tempdir=/opt/omniai/work/instance1/jupyter') \
                        .config('spark.driver.extraJavaOptions',
                                '-Dorg.xerial.snappy.tempdir=/opt/omniai/work/instance1/jupyter') \
                        .config('spark.driver.memory','8g') \
                        .config('fs.s3a.server-side-encryption-algorithm', 'SSE-KMS') \
                        .config('fs.s3a.server-side-encryption-key', kms_key) \
                        .getOrCreate()

            sc = SparkContext.getOrCreate(conf=conf)

    if not isLocal:
        if isEMR:
            bucket = sys.argv[1]
            WRITE_DATE = sys.argv[2]
        else:
            bucket = 'app-idXXXXXXXXXXXXXXXXXXX'
            WRITE_DATE = '20210808'
        firmographics_path = 's3a://'+bucket+ \
                             '/prod-rec/preprocessing/curated/tuned_feat_pup/client/20210101/c/*.parquet'
        targetList_path = 's3a://'+bucket+ \
                          '/prod-rec/preprocessing/curated/tuned_feat_pup/prospect/20210101/p/*.parquet'
        leftDFPath = 's3a://'+bucket+'/prod-rec/peer_cal/curated/prospects/'+WRITE_DATE+'/p/'
        rightDFPath = 's3a://'+bucket+'/prod-rec/peer_cal/curated/clients/'+WRITE_DATE+'/c/'
        peerBatchPath = 's3a://'+bucket+'/prod-rec/peer_cal/curated/peer_results/'+WRITE_DATE+'/pr/'
        reportPath = 's3a://'+bucket+'/prod-rec/peer_cal/curated/peer_for_report/'+WRITE_DATE+'/pfr/'
    else:
        pass

    srp = SearchRankPeers()
    srp.main()
    spark.stop()
    print('TOTAL TIME TO COMPUTE=', datetime.now()-t_start)
