from flask import Flask, render_template, request
import joblib
import numpy as np  # Importez NumPy
import pandas as pd
from datetime import datetime  # Importez datetime

app = Flask(__name__)

# Charger vos modèles et scaler à partir des fichiers .pkl
conservative_model = joblib.load('model_ConservativeInvestors.pkl')
moderate_risk_model = joblib.load('model_ModerateRiskInvestors.pkl')
scaler = joblib.load('process_scaler.pkl')
print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
@app.route('/')
def formulaire():
    return render_template('formulaire.html')

@app.route('/predict', methods=['POST'])
def predict():
    loan_amnt = float(request.form['loan_amnt'])
    term = float(request.form['term'])
    int_rate = float(request.form['int_rate'])
    grade = float(request.form['grade'])
    emp_length = float(request.form['emp_length'])
    annual_inc = float(request.form['annual_inc'])
    dti = float(request.form['dti'])
    delinq_2yrs = float(request.form['delinq_2yrs'])
    earliest_cr_line = request.form['earliest_cr_line']
    inq_last_6mths = float(request.form['inq_last_6mths'])
    open_acc = float(request.form['open_acc'])
    pub_rec = float(request.form['pub_rec'])
    revol_bal = float(request.form['revol_bal'])
    revol_util = float(request.form['revol_util'])
    total_acc = float(request.form['total_acc'])
    last_credit_pull_d = request.form['last_credit_pull_d']
    last_fico_range_high = float(request.form['last_fico_range_high'])
    fico_score = float(request.form['fico_score'])
    home_ownership_OTHER = float(request.form['home_ownership_OTHER'])
    home_ownership_OWN = float(request.form['home_ownership_OWN'])
    home_ownership_RENT = float(request.form['home_ownership_RENT'])
    verification_status_Source_Verified = float(request.form['verification_status_Source Verified'])
    verification_status_Verified = float(request.form['verification_status_Verified'])
    purpose_credit_card = float(request.form['purpose_credit_card'])
    purpose_debt_consolidation = float(request.form['purpose_debt_consolidation'])
    purpose_educational = float(request.form['purpose_educational'])
    purpose_home_improvement = float(request.form['purpose_home_improvement'])
    purpose_house = float(request.form['purpose_house'])
    purpose_major_purchase = float(request.form['purpose_major_purchase'])
    purpose_medical = float(request.form['purpose_medical'])
    purpose_moving = float(request.form['purpose_moving'])
    purpose_other = float(request.form['purpose_other'])
    purpose_renewable_energy = float(request.form['purpose_renewable_energy'])
    purpose_small_business = float(request.form['purpose_small_business'])
    purpose_vacation = float(request.form['purpose_vacation'])
    purpose_wedding = float(request.form['purpose_wedding'])
    addr_state_AL = float(request.form['addr_state_AL'])
    addr_state_AR = float(request.form['addr_state_AR'])
    addr_state_AZ = float(request.form['addr_state_AZ'])
    addr_state_CA = float(request.form['addr_state_CA'])
    addr_state_CO = float(request.form['addr_state_CO'])
    addr_state_CT = float(request.form['addr_state_CT'])
    addr_state_DC = float(request.form['addr_state_DC'])
    addr_state_DE = float(request.form['addr_state_DE'])
    addr_state_FL = float(request.form['addr_state_FL'])
    addr_state_GA = float(request.form['addr_state_GA'])
    addr_state_HI = float(request.form['addr_state_HI'])
    addr_state_IA = float(request.form['addr_state_IA'])
    addr_state_ID = float(request.form['addr_state_ID'])
    addr_state_IL = float(request.form['addr_state_IL'])
    addr_state_IN = float(request.form['addr_state_IN'])
    addr_state_KS = float(request.form['addr_state_KS'])
    addr_state_KY = float(request.form['addr_state_KY'])
    addr_state_LA = float(request.form['addr_state_LA'])
    addr_state_MA = float(request.form['addr_state_MA'])
    addr_state_MD = float(request.form['addr_state_MD'])
    addr_state_ME = float(request.form['addr_state_ME'])
    addr_state_MI = float(request.form['addr_state_MI'])
    addr_state_MN = float(request.form['addr_state_MN'])
    addr_state_MO = float(request.form['addr_state_MO'])
    addr_state_MS = float(request.form['addr_state_MS'])
    addr_state_MT = float(request.form['addr_state_MT'])
    addr_state_NC = float(request.form['addr_state_NC'])
    addr_state_NE = float(request.form['addr_state_NE'])
    addr_state_NH = float(request.form['addr_state_NH'])
    addr_state_NJ = float(request.form['addr_state_NJ'])
    addr_state_NM = float(request.form['addr_state_NM'])
    addr_state_NV = float(request.form['addr_state_NV'])
    addr_state_NY = float(request.form['addr_state_NY'])
    addr_state_OH = float(request.form['addr_state_OH'])
    addr_state_OK = float(request.form['addr_state_OK'])
    addr_state_OR = float(request.form['addr_state_OR'])
    addr_state_PA = float(request.form['addr_state_PA'])
    addr_state_RI = float(request.form['addr_state_RI'])
    addr_state_SC = float(request.form['addr_state_SC'])
    addr_state_SD = float(request.form['addr_state_SD'])
    addr_state_TN = float(request.form['addr_state_TN'])
    addr_state_TX = float(request.form['addr_state_TX'])
    addr_state_UT = float(request.form['addr_state_UT'])
    addr_state_VA = float(request.form['addr_state_VA'])
    addr_state_VT = float(request.form['addr_state_VT'])
    addr_state_WA = float(request.form['addr_state_WA'])
    addr_state_WI = float(request.form['addr_state_WI'])
    addr_state_WV = float(request.form['addr_state_WV'])
    addr_state_WY = float(request.form['addr_state_WY'])

    # Appliquez les transformations aux champs nécessaires
    annual_inc = np.log10(annual_inc)

    # Effectuez des prédictions avec vos modèles
    input_data = [
        loan_amnt, term, int_rate, grade, emp_length, annual_inc, dti, delinq_2yrs, earliest_cr_line,
        inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util, total_acc, last_credit_pull_d,
        last_fico_range_high, fico_score, home_ownership_OTHER, home_ownership_OWN, home_ownership_RENT,
        verification_status_Source_Verified, verification_status_Verified, purpose_credit_card,
        purpose_debt_consolidation, purpose_educational, purpose_home_improvement, purpose_house,
        purpose_major_purchase, purpose_medical, purpose_moving, purpose_other, purpose_renewable_energy,
        purpose_small_business, purpose_vacation, purpose_wedding, addr_state_AL, addr_state_AR,
        addr_state_AZ, addr_state_CA, addr_state_CO, addr_state_CT, addr_state_DC, addr_state_DE,
        addr_state_FL, addr_state_GA, addr_state_HI, addr_state_IA, addr_state_ID, addr_state_IL,
        addr_state_IN, addr_state_KS, addr_state_KY, addr_state_LA, addr_state_MA, addr_state_MD,
        addr_state_ME, addr_state_MI, addr_state_MN, addr_state_MO, addr_state_MS, addr_state_MT,
        addr_state_NC, addr_state_NE, addr_state_NH, addr_state_NJ, addr_state_NM, addr_state_NV,
        addr_state_NY, addr_state_OH, addr_state_OK, addr_state_OR, addr_state_PA, addr_state_RI,
        addr_state_SC, addr_state_SD, addr_state_TN, addr_state_TX, addr_state_UT, addr_state_VA,
        addr_state_VT, addr_state_WA, addr_state_WI, addr_state_WV, addr_state_WY
    ]

    scaled_data = scaler.transform([input_data])
    conservative_prediction = conservative_model.predict(scaled_data)
    moderate_risk_prediction = moderate_risk_model.predict(scaled_data)
    
    return render_template('result.html', conservative=conservative_prediction[0], moderate_risk=moderate_risk_prediction[0])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
