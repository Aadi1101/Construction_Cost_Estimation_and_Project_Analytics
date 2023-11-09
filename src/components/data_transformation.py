import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('src/models',"preprocessor.pkl")
    categorical_encoder_obj_file_path = os.path.join('src/models','categorical_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name="Total"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            categorical_columns = [
                'Commodity Code','Item Description','Project Name','Greenfield/ Brownfield','Client','Market Sector/Industry','Delivery Method','Item Type','coordinates','state','city','suburb'
            ]
            encoder = LabelEncoder()
            for column in range(len(categorical_columns)):
                input_feature_train_df[categorical_columns[column]] = encoder.fit_transform(input_feature_train_df[categorical_columns[column]])
                input_feature_test_df[categorical_columns[column]] = encoder.fit_transform(input_feature_test_df[categorical_columns[column]])

            scaler = StandardScaler()
            input_feature_train_arr = scaler.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                filepath=self.data_transformation_config.preprocessor_obj_file_path,
                obj=scaler
            )
            save_object(
                filepath=self.data_transformation_config.categorical_encoder_obj_file_path,
                obj=encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.categorical_encoder_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
