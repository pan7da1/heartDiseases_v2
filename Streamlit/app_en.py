import streamlit as st
import pickle
import numpy as np

MODEL_NAME = 'model_cardio_04_03_23.pcl'


# definition load-model function 
def load():
    with open(os.path.dirname(__file__) + f'/models/{MODEL_NAME}', 'rb') as fid:
        return pickle.load(fid)

# function define arterial pressure level
def ag_step(ap_sis):
    if ap_sis < 140:
        return 0
    elif 140 <= ap_sis < 160:
        return 1
    elif 160 <= ap_sis < 180:
        return 2
    elif ap_sis >= 180:
        return 3
    else:
        return 4 
    

# definition preprocessing input data for model prediction
def preprocess():
    global age, gend, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active
    height_cm = height * 100
    mass_idx = weight / (height**2)
    years = age // 365
    avrg_ap = np.round((2 * ap_lo + ap_hi) / 3, 1)
    aphi_chol_gluc = (avrg_ap + years) * (chol + gluc)
    ag_st = ag_step(ap_hi)
    ssz_risk = years + avrg_ap + (aphi_chol_gluc // 10)
    gender = (1 if gend == 'Female' else 2)
    return [age, gender, height_cm, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active,
            mass_idx, years, avrg_ap, aphi_chol_gluc, ag_st, ssz_risk]


# function return factors risk cardio-vasc-diseases
def risk_factor(**kwargs):
    factors = ''
    if kwargs['chol'] > 1:
        factors += 'Cholesterol level is hi!\n\n'
    if kwargs['ap_hi'] > 130:
        factors += 'Systolic blood pressure is hi!\n\n'
    if kwargs['ap_lo'] > 90:
        factors += 'Diastolic blood pressure is hi!\n\n'
    if (kwargs['weight'] // kwargs['height']**2) > 30:
        factors += 'Weight is hi!\n\n'
    if kwargs['gluc'] > 2:
        factors += 'Glucose level is hi!\n\n'
    if kwargs['active'] == 0:
        factors += 'You are not physically active enough!\n\n'
    return factors


# output Header
st.title('Prediction of cardiovascular diseases')
st.subheader('Enter your data to predict the probability of having cardiovascular diseases')
# 3 columns for user data input
lcol, mcol, rcol = st.columns(3)
# left column
age = lcol.selectbox('Your age, year:', [*range(1, 120)], index=30, key='age')
height = lcol.slider('Your height, m', 0.5, 2.5, 1.75)
weight = lcol.slider('Your weight, lbs:', 40, 200, 80)
# middle column
ap_hi = mcol.selectbox('Your systolic blood pressure (SBP), mm Hg:',
                       [*range(40, 271)], index=80, key='ap_hi')
ap_lo = mcol.selectbox('Your diastolic blood pressure (DBP), mm Hg:',
                       [*range(20, 161)], index=60, key='ap_lo')
chol = mcol.selectbox('Your cholesterol level:', [1, 2, 3], key='cholesteerol')
gluc = mcol.selectbox('Your glucose level:', [1, 2, 3], key='glucose')
# right column
rcol.text('Do you smoke?')
smoke = rcol.checkbox('Yes', key='smoke')
rcol.text('Do you drink alchogol?')
alco = rcol.checkbox('Yes', key='alco')
rcol.text('Are you physically active?')
active = rcol.checkbox('Yes', key='active')
gend = rcol.radio('Gender', ('Male', 'Female'), index=0, key='gender')
# call all functions
# data preprocessing and get model predict
model = load()
data = preprocess()
predict = model.predict_proba(np.array(data).reshape((1,-1)))[:, 1]
# data out for user
st.text('The probability of heart and vascular diseases is:')
st.write(int(predict * 100), '%')
# suggest output
if predict > 0.5:
    st.write('Please go through a medical diagnostic examination! It is also worth paying attention to the following risk factors:')
    risk_f = risk_factor(chol=chol, ap_hi=ap_hi, ap_lo=ap_lo, height=height, weight=weight, gluc=gluc, active=active)
    st.write(risk_f)
else:
    st.write('Everything is fine! The risk of cardiovascular diseases is not high.') 