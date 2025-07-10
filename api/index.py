from flask import Flask, redirect, url_for, render_template, request
from math import exp
import os

app = Flask(__name__)

# Set template and static folders to parent directory
app.template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
app.static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

def sigmoid(logit):
    return 1/ (1+ exp(-logit)) 

def get_digits_from_probability(probability):
    percentage = int(100*probability)
    if percentage < 10:
        first_digit = 0
        second_digit = percentage
    else:
        first_digit = percentage // 10
        second_digit = percentage % 10
    return first_digit, second_digit

def calculate_churn_probability(tenure, contract_type, payment_method, monthly_payment):
    intercept = 0.67335711
    senior_citizen = 0 * 0.06050222
    partner = 0 * 0.06050222
    dependents = 0 * -0.47939529
    internet_service = 1*0.310115

    match tenure:
        case '0-6':
            months_tenure_0_6 = 1 * 1.17514769
            months_tenure_7_18 = 0
            months_tenure_37_72 = 0
        case '7-18':
            months_tenure_0_6 = 0
            months_tenure_7_18 = 1 * 0.23683483
            months_tenure_37_72 = 0
        case '19+':
            months_tenure_0_6 = 0
            months_tenure_7_18 = 0
            months_tenure_37_72 = 1 * (-0.34801774)

    match contract_type:
        case 'month_to_month':
            contract_month_to_month = 1 * (-0.92931593)
            contract_one_year = 0
            contract_two_year = 0
        case 'one_year':
            contract_month_to_month = 0
            contract_one_year = 1 * (-1.93593759)
            contract_two_year = 0
        case 'two_year':
            contract_month_to_month = 0
            contract_one_year = 0
            contract_two_year = 1 * (-3.27136096)

    match payment_method:
        case 'bank_transfer':
            payment_method_bank_transfer = 1 * (-0.12306059)
            payment_method_credit_card = 0
            payment_method_electronic_check = 0
        case 'credit_card':
            payment_method_bank_transfer = 0
            payment_method_credit_card = 1 * (-0.20074088)
            payment_method_electronic_check = 0
        case 'electronic_check':
            payment_method_bank_transfer = 0
            payment_method_credit_card = 0
            payment_method_electronic_check = 1 * 0.38102869

    match monthly_payment:
        case '0-74':
            monthly_charges_0_74 = 1 * (-0.94640383)
            monthly_charges_75_100 = 0
            monthly_charges_100 = 0
        case '75-100':
            monthly_charges_0_74 = 0
            monthly_charges_75_100 = 1 * 0.48418353
            monthly_charges_100 = 0
        case '100+':
            monthly_charges_0_74 = 0
            monthly_charges_75_100 = 0
            monthly_charges_100 = 1 * 1.03457548

    logit = (
        intercept +
        senior_citizen +
        partner +
        dependents +
        internet_service +
        months_tenure_0_6 +
        months_tenure_7_18 +
        months_tenure_37_72 +
        contract_month_to_month +
        contract_one_year +
        contract_two_year +
        payment_method_bank_transfer +
        payment_method_credit_card +
        payment_method_electronic_check +
        monthly_charges_0_74 +
        monthly_charges_75_100 +
        monthly_charges_100
    )
    
    probability = sigmoid(logit)
    return probability

@app.route('/get_value_from_answers',methods=['POST','GET'])
def get_value_from_answers():
    tenure = request.form['tenure']
    contract_type = request.form['contract_type']
    payment_method = request.form['payment_method']
    monthly_payment = request.form['monthly_payment']

    churn_probability = calculate_churn_probability(tenure,
                                                    contract_type, 
                                                    payment_method, 
                                                    monthly_payment)
    
    first_digit = get_digits_from_probability(churn_probability)[0]
    second_digit = get_digits_from_probability(churn_probability)[1]

    return render_template("result.html", first_digit = first_digit, second_digit=second_digit)

@app.route('/restart_questions')
def restart_questions():
    return render_template("questions.html")

# For Vercel
def handler(request):
    return app(request.environ, lambda status, headers: None)