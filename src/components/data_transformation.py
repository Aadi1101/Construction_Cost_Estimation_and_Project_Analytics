"""
This module handles the data transformation process, particularly focusing on encoding categorical \
    features and preparing data for model training.

It reads the training and testing datasets, applies a target encoding to the specified categorical\
    columns, and returns the transformed datasets along with the saved encoder object \
        for future use.
"""
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import category_encoders as ce
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for the data transformation process.

    Attributes:
        preprocessor_obj_file_path (str): Path to save the preprocessing object \
            (not used in the current code).
        categorical_encoder_obj_file_path (str): Path to save the categorical encoder \
            object (Target Encoder).
    """
    preprocessor_obj_file_path=os.path.join('src/models',"preprocessor.pkl")
    categorical_encoder_obj_file_path = os.path.join('src/models','categorical_encoder.pkl')

class DataTransformation:
    """
    Class responsible for data transformation, including categorical encoding and 
    preparing training and testing data for model training.

    This class reads training and testing datasets, applies a target encoding to \
        categorical columns, and saves the encoding object for future use. \
            It returns transformed datasets that can be used for training and testing \
                machine learning models.

    Attributes:
        data_transformation_config (DataTransformationConfig): Configuration object \
            that holds paths for saving the transformation objects.
    """
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path):
        """
        Initiates the data transformation process by applying target encoding \
            on the categorical columns.

        Args:
            train_path (str): The file path to the training dataset (CSV).
            test_path (str): The file path to the testing dataset (CSV).

        Returns:
            tuple: A tuple containing the following:
                - numpy.ndarray: Transformed training dataset as a NumPy array.
                - numpy.ndarray: Transformed testing dataset as a NumPy array.
                - str: The file path where the categorical encoder object is saved.
        
        Raises:
            CustomException: If any exception occurs during the data transformation process, \
                it raises a custom exception.
        """
        try:
            categorical_columns = ['Commodity Code', 'Item Description', 'Project Name',
                                    'Greenfield/ Brownfield', 'Client', 'Market Sector/Industry', 
                                    'Delivery Method', 'Item Type', 'coordinates', 'state', 'city', 
                                    'suburb']
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            target_column_name="Total"

            input_feature_train_df=train_df.drop(columns=[target_column_name,'Attribute 4'],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name,'Attribute 4'],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            target_encoder = ce.TargetEncoder(cols=categorical_columns)
            input_feature_train_df[categorical_columns] = target_encoder.fit_transform(
                input_feature_train_df[categorical_columns],target_feature_train_df)
            input_feature_test_df[categorical_columns] = target_encoder.transform(
                input_feature_test_df[categorical_columns])

            train_arr = np.c_[
                np.array(input_feature_train_df), np.array(target_feature_train_df)
            ]
            test_arr = np.c_[np.array(input_feature_test_df), np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                filepath=self.data_transformation_config.categorical_encoder_obj_file_path,
                obj=target_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.categorical_encoder_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys) from e
