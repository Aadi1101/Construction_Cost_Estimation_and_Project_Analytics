# CONSTRUCTION COST ESTIMATION AND PROJECT ANALYTICS

## Project Description
This project is a Flask web application designed to predict construction costs based on various input parameters. Users can submit data through a web interface and receive predictions, enabling better budgeting and planning for construction projects.

## Table of Contents
- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [API Endpoints](#api-endpoints)
- [ML Model](#ml-model)
- [Model Tuning and Preprocessing](#model-tuning-and-preprocessing)
- [Future Enhancements](#future-enhancements)
- [References or Documentation Links](#references-or-documentation-links)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
- Python 3.7 or higher
- Pip (Python package manager)

## Technologies Used
- Flask
- Numpy
- Pandas
- Scikit-learn
- XGBoost
- CatBoost
- Seaborn
- Dill
- OpenPyXL
- Category Encoders

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/construction-cost-prediction.git
   cd construction-cost-prediction
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000/`.

## Input Parameters
The model accepts the following input parameters:
- Commodity Code
- Item Description
- Qty
- PE Amount
- BM Amount
- LB hrs
- LB Amount
- CE Amount
- Major SC Amount
- Fuel usage (L)
- Attribute 1
- Attribute 2
- Attribute 3
- Project Number
- Total New
- Single Unit Price
- Epic Embodied Carbon
- AUS LCI Embodied Carbon
- Carbon Allowance
- Construction Carbon
- Default PE Unit Price
- Default BM Unit Price
- Default LB Unit Hrs
- Default SC Unit Rate
- Project Name
- Greenfield/Brownfield
- Client
- Market Sector/Industry
- Latitude
- Longitude
- Delivery Method
- Item Type
- Flag
- Coordinates
- State
- City
- Suburb

## API Endpoints
- **GET `/predict`**: Use this endpoint to generate predictions based on the input data.
- Docker Image: [Docker Hub - Construction Cost Prediction](https://hub.docker.com/repository/docker/gogetama/construction_cost_estimation_and_project_analytics/general)

## ML Model
The project utilizes various machine learning models to predict construction costs, including:
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- XGBRegressor
- CatBoosting Regressor
- AdaBoost Regressor

## Model Tuning and Preprocessing
Hyperparameter tuning is performed using GridSearchCV for models like Random Forest, XGBoost, and CatBoost to ensure optimal performance. Preprocessing includes encoding categorical variables and scaling numerical features.

## Future Enhancements
- Implement user authentication for secure access.
- Add more machine learning algorithms for better performance comparison.
- Incorporate real-time data updates and predictions.
- Enhance the web interface for better user experience.

## References or Documentation Links
- [Flask Documentation](https://flask.palletsprojects.com/en/3.0.x/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/python/index.html)
- [CatBoost Documentation](https://catboost.ai/en/docs/concepts/python-reference_catboostregressor)

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Aadi1101/Construction_Cost_Estimation_and_Project_Analytics/blob/main/LICENSE) file for more details.