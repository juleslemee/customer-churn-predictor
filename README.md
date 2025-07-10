# Customer Churn Predictor 📊

Machine learning project predicting telecommunications customer churn with interactive web interface for real-time predictions.

**🌐 Live Demo:** https://predictchurn.juleslemee.com

![Churn Analysis](20704plot.png)

## Project Overview

This project investigates customer churn in telecommunications through machine learning techniques, developed and presented at HEC Montréal under AI Scientist Jonathan Moatti's oversight. We analyzed IBM's sample dataset using scikit-learn, pandas, and matplotlib, applying advanced techniques like SMOTE (Synthetic Minority Oversampling Technique) for handling imbalanced data.

The project combines rigorous data science methodology with practical application through an interactive Flask web application that predicts churn likelihood for individual customers.

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
- **Data Cleaning**: Removed irrelevant variables and handled missing values
- **Categorical Encoding**: Applied one-hot encoding to convert categorical variables into binary format
- **Class Balancing**: Used SMOTE (Synthetic Minority Oversampling Technique) to address limited churn cases
- **Threshold Optimization**: Adjusted classification threshold for optimal churn prediction

### Machine Learning Models
We chose **Logistic Regression** as our final model over Cox regression based on database constraints and project goals, achieving:
- **80% accuracy** for non-churn prediction
- **70% accuracy** for churn prediction

Other models explored:
- Random Forest
- Gradient Boosting  
- Support Vector Machines

### Interactive Web Application
Built with Python's Flask library, the web interface allows users to:
- Input their telecommunications service parameters
- Get real-time churn probability predictions
- Experience the model that was demonstrated to classmates during our HEC Montréal presentation

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

### Running the Interactive Web App
```bash
python app.py
```
Then visit http://localhost:5000 to use the churn predictor.

## Results

- **Final Model**: Logistic Regression
- **Non-Churn Accuracy**: 80%
- **Churn Detection Accuracy**: 70%
- **Key Achievement**: Successfully demonstrated live predictions to classmates at HEC Montréal
- **Interactive Demo**: Deployed web application for real-time churn prediction

## Technologies Used

- **Data Science**: scikit-learn, pandas, matplotlib
- **Machine Learning**: Logistic Regression, SMOTE
- **Web Development**: Flask, HTML/CSS
- **Deployment**: Custom domain hosting

## Contributors
Jules Lemée
Maxence Dhondt
Cyprien Boustiha
Sandrine Trinh

