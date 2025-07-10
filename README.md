# Customer Churn Predictor 📊

Machine learning project to predict customer churn using the Telco Customer Churn dataset, with an interactive demonstration component. 
Available @ https://predictchurn.juleslemee.com

## Project Overview

This project analyzes customer churn patterns in telecommunications data to build predictive models that identify customers likely to cancel their service. Originally developed as a group project for a pandas-focused programming class, it includes both analytical insights and an interactive component for real-time predictions.

## Dataset

Using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) from Kaggle, which includes:
- 7,043 customer records
- 20+ features including demographics, services, and billing information
- Binary churn indicator (Yes/No)

## Features Analyzed

- **Demographics**: Gender, age range, dependents
- **Services**: Phone, internet, streaming subscriptions
- **Account Info**: Contract type, payment method, tenure
- **Financial**: Monthly charges, total charges

## Technical Implementation

### Data Processing
- Cleaned and preprocessed data using pandas
- Handled missing values and categorical encoding
- Feature engineering for improved predictions

### Machine Learning Models
Explored multiple algorithms with scikit-learn:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- Support Vector Machines

### Interactive Component
Built an interactive dashboard where users can:
- Input their own service parameters
- Get real-time churn predictions
- Understand which factors contribute most to churn risk

## Key Findings

- Contract type is the strongest predictor of churn
- Month-to-month customers have 3x higher churn rate
- Electronic check payment correlates with higher churn
- Fiber optic internet customers show higher churn (possibly due to service issues)

## Project Structure

```
customer-churn-predictor/
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks for analysis
├── models/              # Saved ML models
├── dashboard/           # Interactive prediction interface
└── presentation/        # Class presentation materials
```

## Usage

### Running the Analysis
```bash
jupyter notebook analysis.ipynb
```

### Running the Interactive Dashboard
```bash
python dashboard.py
```

## Results

- Best model: Gradient Boosting with 82% accuracy
- Precision: 79%
- Recall: 85%
- Successfully demonstrated live predictions in class

## Future Improvements

- Implement deep learning models
- Add customer segmentation analysis
- Create retention strategy recommendations
- Deploy as web application

## Contributors
Jules Lemée
Maxence Dhondt
Cyprien Boustiha
Sandrine Trinh

