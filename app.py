"""
A Flask web application for predicting Construction Costs based on input data.
This application provides a web interface and endpoints for users
to submit data and receive predictions.
"""
from flask import Flask,request,render_template
import dill
import numpy as np
import pandas as pd
import jsonify

app=Flask('__name__')
@app.route('/')
def read_main():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def generate_output():
    """Generate predictions based on input data.

    This function retrieves input data from the request, processes it, and
    returns a prediction for PCOS.

    Returns:
        dict: A dictionary containing the prediction result.
    """
    json_data = False
    input_data = request.args.get('data')
    if input_data is None:
        input_data = request.get_json()
        json_data = True
    pcos = process_and_predict(input_text=input_data,json_data=json_data)
    return {'predicted':pcos}

def process_and_predict(input_text,json_data):
    """
    Process input text and make a prediction.

    Args:
        input_text (str or dict): The input data for prediction.
        json_data (bool): Flag indicating if the input is in JSON format.

    Returns:
        str: Construction cost
    """
    input_columns = [
    'Commodity Code','Item Description','Qty','PE Amount','BM Amount',
    'LB hrs','LB Amount','CE Amount','Major SC Amount','Fuel usage (L)',
    'Attribute 1','Attribute 2','Attribute 3','project_number','total_new',
    'Single Unit Price','epic_embodied_carbon','aus_lci_embodied_carbon',
    'carbon_allowance','construction_carbon','Default PE Unit Price',
    'Default BM Unit Price','Default LB Unit Hrs','Default SC Unit Rate',
    'Project Name','Greenfield/ Brownfield','Client','Market Sector/Industry',
    'Latitude','Longitude','Delivery Method','Item Type','Flag','coordinates',
    'state','city','suburb'
]
    categorical_columns = ['Commodity Code', 'Item Description', 'Project Name',
                                    'Greenfield/ Brownfield', 'Client', 'Market Sector/Industry', 
                                    'Delivery Method', 'Item Type', 'coordinates', 'state', 'city', 
                                    'suburb']
    if json_data is True:
        output_text = [item for item in input_text['data']]
    else:
        output_text = [item for item in input_text.split(',')]
    input_dict = {}
    for i in range(len(input_columns)):
        input_dict[input_columns[i]] = output_text[i]
    with open('src/models/categorical_encoder.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    df = pd.DataFrame([input_dict])
    df[categorical_columns] = preprocessor.transform(df[categorical_columns])
    input_array = np.array(df)
    with open(r'src/models/best_model.pkl', 'rb') as m:
        model = dill.load(m)
    input_array = input_array.reshape(1,-1)
    prices = model.predict(input_array)
    prices = prices.tolist()
    return str(prices[0])
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)