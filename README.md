# Customer Churn Prediction System

## Project Overview

This project is a Machine Learning based web application developed to predict whether a customer is likely to churn or stay on an eCommerce platform.

Customer churn refers to the situation where a customer stops engaging with the business or stops making purchases. Since retaining existing customers is usually more cost effective than acquiring new ones, predicting churn in advance helps businesses take proactive retention measures.

The system uses historical customer behaviour data such as login activity, session duration, browsing behaviour, cart abandonment, email engagement, customer support interaction, and purchase related information to estimate churn probability.

An interactive frontend has also been developed using Streamlit, allowing users to enter customer details, generate predictions, view churn probability, compare behaviour against the average customer, and observe supporting visual insights.

---

## Objectives

- To predict whether a customer is likely to churn or stay
- To identify behavioural indicators associated with churn
- To build an interactive dashboard for real time prediction
- To compare entered customer behaviour against average customer behaviour
- To support customer retention decision making using machine learning insights

---

## Technologies Used

### Programming Language
- Python

### Libraries and Frameworks
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pickle
- Streamlit
- Pillow

### Machine Learning Algorithm
- Logistic Regression

---

## Dataset Information

This project uses the **Ecommerce Customer Behaviour Dataset** obtained from Kaggle.

The dataset contains **50000 customer records** and **25 columns**, including demographic, behavioural, engagement, and purchase related features.

### Target Variable
- `Churned`
  - `0` means customer stayed
  - `1` means customer churned

### Important Input Features
- Age
- Gender
- Country
- City
- Membership_Years
- Login_Frequency
- Session_Duration_Avg
- Pages_Per_Session
- Cart_Abandonment_Rate
- Wishlist_Items
- Total_Purchases
- Average_Order_Value
- Days_Since_Last_Purchase
- Discount_Usage_Rate
- Returns_Rate
- Email_Open_Rate
- Customer_Service_Calls
- Product_Reviews_Written
- Social_Media_Engagement_Score
- Mobile_App_Usage
- Payment_Method_Diversity
- Lifetime_Value
- Credit_Balance
- Signup_Quarter

---

## Problem Statement

Businesses often lose customers without clear warning signs. If churn can be predicted early, companies can take action through targeted offers, improved support, loyalty rewards, or personalised engagement strategies.

The goal of this project is to build a churn prediction system that can classify whether a customer is likely to stay or churn based on past behaviour and activity patterns.

---

## Project Workflow

### 1. Data Collection
The dataset was downloaded from Kaggle and loaded into Python using Pandas.

### 2. Data Inspection
Initial inspection steps included:
- checking dataset shape
- checking column names
- checking missing values
- checking duplicate rows
- checking target variable distribution

### 3. Data Cleaning
Only clearly invalid values were corrected. For example:
- unrealistic age values were replaced with missing values
- negative purchase values were corrected
- percentage based columns exceeding 100 were handled properly

Missing values were retained until the train test split stage to avoid leakage.

### 4. Train Test Split
The dataset was divided into:
- 80 percent training data
- 20 percent testing data

Stratified splitting was used so that churn proportion remained consistent in both training and testing sets.

### 5. Preprocessing
The following preprocessing steps were applied:
- missing numerical values were filled using training median
- missing categorical values were filled using training mode
- categorical columns were converted using one hot encoding
- training and testing columns were aligned to ensure the same feature structure

### 6. Model Training
A Logistic Regression model was trained on the processed training dataset.

Logistic Regression was chosen because:
- it is easy to interpret
- it works well for binary classification
- it is suitable for academic mini projects
- its outputs can be explained clearly in viva and documentation

### 7. Model Evaluation
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The initial model performed reasonably well, but recall for churn customers was moderate. To improve practical usefulness, threshold tuning was performed.

### 8. Threshold Tuning
Instead of using the default threshold of 0.50, the prediction threshold was reduced to 0.40.

This improved the model’s ability to detect actual churn customers by increasing churn recall, while keeping overall accuracy almost similar.

### 9. Model Saving
The trained model and final feature columns were saved using Pickle.

Saved files:
- `churn_model.pkl`
- `model_columns.pkl`

### 10. Frontend Development
A Streamlit based dashboard was created for interactive churn prediction.

Users can:
- enter customer behaviour values
- generate predictions
- view churn probability
- check customer vs dataset average comparison
- view visual insights

---

## Exploratory Data Analysis Performed

The following visualisations were created during the analysis phase:

### 1. Customer Churn Distribution
This chart showed that approximately:
- 71 percent customers stayed
- 29 percent customers churned

This indicated a moderate class imbalance and justified the need for churn modelling.

### 2. Feature Correlation Heatmap
This helped in understanding the relationship between behavioural variables.

It showed that highly engaged customers tend to have:
- higher login frequency
- longer sessions
- more pages per session
- more app usage
- stronger digital engagement

Churn showed negative relation with engagement and positive association with friction related features such as cart abandonment and customer service calls.

### 3. Churn by Country
This visualisation showed churn counts across countries. Markets such as the USA had higher churn counts mainly due to a larger customer base.

---

## Model Performance

The Logistic Regression model achieved good baseline performance for a mini project.

### Initial Evaluation
- Accuracy was around 77 percent
- Churn precision was moderate
- Churn recall was lower than desired

### After Threshold Tuning
By lowering the threshold from 0.50 to 0.40:
- more churn customers were correctly detected
- churn recall improved significantly
- the model became more useful for retention oriented business scenarios

This was important because in churn prediction, missing a churn customer can be costlier than slightly lowering accuracy.

---

## Frontend Features

The project includes a Streamlit dashboard with the following features:

### 1. Customer Input Panel
Users can enter customer information using:
- sliders
- dropdowns
- text input fields

### 2. Prediction Result
The system shows whether the customer is:
- likely to stay
- at high risk of churn

### 3. Churn Probability
The dashboard displays churn probability as a percentage along with a progress bar.

### 4. Risk Level
Customers are categorised into:
- Low Risk
- Medium Risk
- High Risk

### 5. Recommendation Section
The dashboard provides business recommendations based on the predicted churn risk.

Examples:
- low risk customers can be retained through loyalty strategies
- medium risk customers can be targeted with personalised offers
- high risk customers may require discounts, re-engagement emails, or support follow up

### 6. Customer vs Dataset Average Comparison
A comparative bar chart shows how the entered customer differs from the average customer on selected behavioural metrics such as:
- login frequency
- session duration
- pages per session
- cart abandonment rate
- email open rate
- customer service calls

### 7. Visual Insights
The dashboard also displays static insights from the notebook analysis:
- churn distribution
- feature correlation heatmap
- churn by country

---

## Example Inputs for Demo

### Example 1: Likely Churn Customer
Use values like:
- low login frequency
- low session duration
- low pages per session
- high cart abandonment
- low email open rate
- high customer service calls
- low mobile app usage
- low lifetime value

This profile usually produces a high churn probability.

### Example 2: Loyal Customer
Use values like:
- high login frequency
- high session duration
- high pages per session
- low cart abandonment
- high email open rate
- low customer service calls
- high app usage
- strong lifetime value

This profile is more likely to be predicted as retained.

---

## Folder Structure

```text
Customer-Churn-Prediction/
│
├── churn_model.pkl
├── model_columns.pkl
├── ecommerce_customer_churn_dataset.csv
├── ChurnPrediction.ipynb
├── README.md
│
└── src/
    ├── frontend.py
    ├── requirements.txt
    └── Visualisations/
        ├── insight1.png
        ├── insight2.png
        └── insight3.png
