import streamlit as st
import pickle
import numpy as np
import os

MODEL_NAME = 'model_cardio.pcl'


# функция для загрузки модели
def load():
    with open(os.path.dirname(__file__) + f'/models/{MODEL_NAME}', 'rb') as fid:
        return pickle.load(fid)

    
# функция, определяющая степень артериальной гиепртензии по систалическому давлению
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
    

# функция подгостовки введенных данных для предсказания модели
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


# функция, возвращающая факторы риска ССЗ
def risk_factor(**kwargs):
    factors = ''
    if kwargs['chol'] > 1:
        factors += 'Уровень холестерина высокий!\n\n'
    if kwargs['ap_hi'] > 130:
        factors += 'Повышенное систалическое давление!\n\n'
    if kwargs['ap_lo'] > 90:
        factors += 'Повышенное диастолическое давление!\n\n'
    if (kwargs['weight'] // kwargs['height']**2) > 30:
        factors += 'Избыточная масса тела!\n\n'
    if kwargs['gluc'] > 2:
        factors += 'Уровень глюкозы высок!\n\n'
    if kwargs['active'] == 0:
        factors += 'Вы ведете физически малоактивный образ жизни!\n\n'
    return factors


# вывод заголовка
st.title('Предсказание риска сердечно-сосудистых заболеваний.')
st.subheader('Введите данныe для предсказания вероятности наличия сердечно-сосудистых заболеваний:')
# создано 3 колонки для ввода данных
lcol, mcol, rcol = st.columns(3)
# левая колонка
age = lcol.selectbox('Ваш возраст, лет:', [*range(1, 120)], index=30, key='age')
height = lcol.slider('Ваш рост, м', 0.5, 2.5, 1.75)
weight = lcol.slider('Ваш вес, кг:', 40, 200, 80)
# средняя колонка
ap_hi = mcol.selectbox('Верхнее (систолическое) среднее давление, мм рт.ст:',
                       [*range(40, 271)], index=80, key='ap_hi')
ap_lo = mcol.selectbox('Нижнее (диастолическое) среднее давление, мм рт.ст:',
                       [*range(20, 161)], index=60, key='ap_lo')
chol = mcol.selectbox('Ваш уровень холестерина:', [1, 2, 3], key='cholesteerol')
gluc = mcol.selectbox('Ваш уровень глюкозы', [1, 2, 3], key='glucose')
# правая колонка
rcol.write('Вы курите? ')
smoke = rcol.checkbox('Да', key='smoke')
rcol.write('Вы употребляете алкоголь?')
alco = rcol.checkbox('Да', key='alco')
rcol.write('Вы ведёте физически активный образ жизни?')
active = rcol.checkbox('Да', key='active')
gend = rcol.radio('Ваш пол', ('Муж.', 'Жен.'), index=0, key='gender')
# загружаем модель и
# вызываем функцию подготовки данных перед отправкой в модель 
model = load()
data = preprocess()
predict = model.predict_proba(np.array(data).reshape((1,-1)))[:, 1]
# вывод информации и результатов для пользователя
st.text('Ваша вероятность наличия сердечно-сосудистых заболеваний:')
st.write(int(predict * 100), '%')
# вывод рекомендаций
if predict > 0.5:
    st.write('Пожалуйста пройдите медицинское обследование! Так же обратите внимание на нижеперечисленные факторы риска сердечно-сосудистых заболеваний:')
    risk_f = risk_factor(chol=chol, ap_hi=ap_hi, ap_lo=ap_lo, height=height, weight=weight, gluc=gluc, active=active)
    st.write(risk_f)
else:
    st.write('Всё хорошо! Риск наличия сердечно-сосудистых заболеваний не высок!') 