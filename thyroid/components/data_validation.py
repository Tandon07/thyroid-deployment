from thyroid.entity import artifact_entity,config_entity
from thyroid.exception import ThyroidException
from thyroid.logger import logging
# from scipy.stats import ks_2samp
from typing import Optional
import os,sys 
import pandas as pd
from thyroid import utils
import numpy as np
from thyroid.config import TARGET_COLUMN
from sklearn.model_selection import train_test_split



class DataValidation:


    def __init__(self,
                    data_validation_config:config_entity.DataValidationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.validation_error=dict()
        except Exception as e:
            raise ThyroidException(e, sys)

    

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold
        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =====================================================================================
        returns Pandas DataFrame if atleast a single column is available after missing columns drop else None
        """
        try:
            
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            #selecting column name which contains null
            logging.info(f"selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            #return None no columns left
            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise ThyroidException(e, sys)

    

    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"Reading base dataframe")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"?":np.NAN},inplace=True)
            logging.info(f"Replace na value in base df")
            #base_df has '?' as null

            logging.info(f"Reading dataframe")
            df = pd.read_csv(self.data_ingestion_artifact.dataset_file_path)  #this df is current df



            logging.info("Splitting test and train and saving to datasplit folder")
            dataset_dir = os.path.dirname(self.data_validation_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)


                   #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)
            train_df,test_df = train_test_split(df,test_size=self.data_validation_config.test_size,random_state=0)
            train_df.to_csv(path_or_buf=self.data_validation_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_validation_config.test_file_path,index=False,header=True)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path,
                train_file_path=self.data_validation_config.train_file_path, 
                test_file_path=self.data_validation_config.test_file_path)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise ThyroidException(e, sys)