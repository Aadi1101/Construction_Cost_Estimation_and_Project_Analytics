"""
This module provides utility functions for model evaluation and saving objects in different formats.
It includes functions to save machine learning models, evaluate their \
    performance using various metrics, and save the evaluation results in JSON format.\
          The module also handles exceptions by raising custom exceptions with \
            detailed error messages.
"""
import os
import sys
import json
import dill
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(filepath,obj):
    """
    Saves a Python object to a specified file using the dill serialization library.

    Args:
        filepath (str): The path where the object will be saved.
        obj: The Python object to be saved.

    Raises:
        CustomException: If there is an error during the file creation or object saving process.
    """
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
    except Exception as e:
        raise CustomException(e,sys) from e

def evaluate_models(x_train,y_train,x_test,y_test,models,params): # pylint: disable=R0913,R0917,R0914
    """
    Evaluates multiple machine learning models using Grid Search for hyperparameter tuning, 
    and computes various performance metrics for both training and testing datasets.

    Args:
        x_train (np.ndarray): The feature set for the training data.
        y_train (np.ndarray): The target values for the training data.
        x_test (np.ndarray): The feature set for the testing data.
        y_test (np.ndarray): The target values for the testing data.
        models (dict): A dictionary of models to evaluate.
        params (dict): A dictionary of hyperparameter grids for the respective models.

    Returns:
        tuple: A tuple containing training and testing performance metrics (MAE, MSE, R2 scores).

    Raises:
        CustomException: If an error occurs during model evaluation or fitting.
    """
    try:
        train_report_score = {}
        train_report_mae = {}
        train_report_mse = {}

        test_report_score = {}
        test_report_mae = {}
        test_report_mse = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            logging.info(f"Evaluation initiated for {model}.")
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            logging.info(f"GridSearchCV initiated for {model}.")
            gs.fit(x_train,y_train)
            logging.info(f"GridSearchCV fit done and set_params initiated for {model}.")
            model.set_params(**gs.best_params_)
            logging.info(f"setting parameters completed and fitting initiated for {model}.")
            model.fit(x_train,y_train)
            logging.info(f"prediction initiated for {model}.")
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info(f"Getting the r2score for train and test data for {model}")

            train_model_mae = mean_absolute_error(y_train,y_train_pred)
            test_model_mae = mean_absolute_error(y_test,y_test_pred)

            train_model_mse = mean_squared_error(y_train,y_train_pred)
            test_model_mse = mean_squared_error(y_test,y_test_pred)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            train_report_score[list(models.keys())[i]] = train_model_score
            train_report_mae[list(models.keys())[i]] = train_model_mae
            train_report_mse[list(models.keys())[i]] = train_model_mse

            test_report_score[list(models.keys())[i]] = test_model_score
            test_report_mae[list(models.keys())[i]] = test_model_mae
            test_report_mse[list(models.keys())[i]] = test_model_mse

            logging.info(f"Obtained r2score of {test_model_score} and completed with {model}.")
        return (
            train_report_mae,train_report_mse,train_report_score,
            test_report_mae,test_report_mse,test_report_score
        )
    except Exception as e:
        raise CustomException(e,sys) from e

def save_json_object(file_path,obj):
    """
    Saves a Python object to a specified file in JSON format.

    Args:
        file_path (str): The path where the JSON object will be saved.
        obj: The Python object to be saved in JSON format.

    Raises:
        CustomException: If there is an error during the file creation or object saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w',encoding="utf-8") as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys) from e
