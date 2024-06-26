import os,sys,dill,json
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(filepath,obj):
    try:
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath,exist_ok=True)
        with open(filepath,'wb') as fileobj:
            dill.dump(obj,fileobj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
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
        return train_report_mae,train_report_mse,train_report_score,test_report_mae,test_report_mse,test_report_score
    except Exception as e:
        raise CustomException(e,sys)
    
def save_json_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)