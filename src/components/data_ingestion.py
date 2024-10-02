"""
This module handles data ingestion, which involves reading a dataset from a CSV file, 
splitting the data into training and testing sets, and saving the \
    resulting datasets to specified paths. 

It also defines configuration parameters for where to store the raw, train, and test datasets. 
The module raises custom exceptions in case of errors during \
    the ingestion process and logs each step for better traceability.
"""
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    """
    Configuration class for data ingestion paths.
    
    Attributes:
        raw_path (str): Path to save the raw dataset.
        train_path (str): Path to save the training dataset.
        test_path (str): Path to save the testing dataset.
    """
    raw_path:str = os.path.join('artifacts','raw.csv')
    train_path:str = os.path.join('artifacts','train.csv')
    test_path:str = os.path.join('artifacts','test.csv')

class DataIngestion():
    """
    Class responsible for data ingestion.

    This class reads data from a source CSV file, performs a train-test split, and saves
    the split datasets to specified file paths. It logs the process and raises custom exceptions
    in case of any issues.

    Attributes:
        data_ingestion_config (DataIngestionConfig): Configuration object that holds the paths 
        for saving the raw, train, and test datasets.
    """
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        This method reads a dataset from a CSV file, splits the data into \
            training and testing sets, and saves them to their respective \
                file paths. The steps are logged for better monitoring, and \
                    custom exceptions are raised for any errors.

        Returns:
            tuple: Paths to the training and testing datasets.
        """
        try:
            logging.info('Started Data Ingestion.')
            df = pd.read_csv(r'notebooks\cleaned_data.csv')
            logging.info("Read the dataset.")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_path,index=False,header=True)
            logging.info("Train Test Split initiated.")
            train_set,test_set = train_test_split(df,test_size=0.1,random_state=0)

            train_set.to_csv(self.data_ingestion_config.train_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_path,index=False,header=True)

            logging.info("Ingestion of data is completed.")

            return(
                self.data_ingestion_config.train_path,
                self.data_ingestion_config.test_path
            )
        except Exception as e:
            raise CustomException(e,sys) from e

if __name__ == '__main__':
    obj = DataIngestion()
    trainset,testset = obj.initiate_data_ingestion()
    transform_obj = DataTransformation()
    train_array,test_array,_= transform_obj.initiate_data_transformation(
        train_path=trainset,test_path=testset
    )
    model_obj = ModelTrainer()
    model_obj.initiate_model_trainer(train_arr=train_array,test_arr=test_array)
