# Industrial-Copper-Modeling
This project aims to develop two machine learning models for the copper industry to address the challenges of predicting selling price and lead classification. There are factors like quantity tons, thickness, application, width etc that will be very useful for prediction.


## Data Preprocessing: 
Preprocess the data like handing missing values, treat outliers if necessary, checking duplicates etc.. to clean and structure it for machine learning model. Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation, boxcox transformation, or other techniques, to handle high skewness in continuous variables.

## Feature Engineering:
Extract relevant features from the dataset and Create any additional features that may enhance prediction accuracy.

## Encoding Categorical Variables:
Encode the categorical variables to numerical ones using label encoding or one-hot-encoding.

## Model Selection and Training: 
* Choose an appropriate machine learning model for regression (e.g. decision trees, random forests, ExtraTreesClassifier, XGBClassifier or any other algorithms). Train the model using a portion of the dataset for training and the rest of the data for testing.
* Choose an appropriate machine learning model for classification (e.g. decision trees, random forests, ExtraTreesClassifier, XGBClassifier or any other algorithms). Train the model using a portion of the dataset for training and the rest of the data for testing.

## Model Evaluation:
* Evaluate the model's predictive performance for regression using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) and R2 Score.
* Evaluate the model's predictive performance for classification using classification metrics such as accuracy, recall, precision and F1 Score.

## Choosing Model:
* After model evaluation, choosing random forest regressor model for predicting the selling price.
* After model evaluation, choosing extra trees classifier model for predicting the status (won or lost).

## Streamlit Web Application: 
Develop a user-friendly web application using Streamlit that allows users to input details (quantity tons, thickness, application, width etc.). Utilize the trained machine learning model to predict the selling price and status based on user inputs.

* ML Regression model which predicts continuous variable ‘Selling_Price’
* ML Classification model which predicts Status: WON or LOST.

