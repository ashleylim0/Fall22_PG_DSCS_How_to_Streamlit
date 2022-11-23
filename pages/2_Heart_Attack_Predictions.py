import pandas as pd
import streamlit as st
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title('Prediction Machines Workshop')

knn = load('./heart_attack/knn.joblib')
logistic = load('./heart_attack/logistic.joblib')
rf = load('./heart_attack/rf.joblib')
svc = load('./heart_attack/svc.joblib')

col1, col2, col3 = st.columns(3)

with col1:
  age = st.slider('Age in years', 29, 77)
  trtbps = st.slider('Resting Blood Pressure', 94, 200)
  restecg_values = {
    'normal': 0,
    'ST-T wave abnormality': 1,
    'ventricular hypertrophy': 2
  }
  restecg = restecg_values[st.selectbox(
    'Esting Electrocardiographic Results',
    ('normal', 'ST-T wave abnormality', 'ventricular hypertrophy'))]
  oldpeak = st.slider('ST depression', 0.0, 6.2, 0.0, 0.1)
  thall_values = {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
  thall = thall_values[st.selectbox(
    'Thall', ('normal', 'fixed defect', 'reversable defect'))]

with col2:
  sex = 1 if st.radio('Sex', ('male', 'female')) == 'male' else 0
  chol = st.slider('Serum Cholestoral', 126, 564)
  thalachh = st.slider('Maximum Heart Rate', 71, 202)
  slp_values = {'upsloping': 0, 'flat': 1, 'downsloping': 2}
  slp = slp_values[st.selectbox('Slope', ('upsloping', 'flat', 'downsloping'))]

with col3:
  cp_values = {
    'Typical angina': 0,
    'Atypical angina': 1,
    'Non-anginal pain': 2,
    'Asympomatic': 3
  }
  cp = cp_values[st.selectbox(
    'Chest Pain Type',
    ('Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asympomatic'))]
  fbps = 1 if st.radio('Fasting Blood Pressure',
                       ("> 120 mg/dL", "<= 120 mg/dL")) == "> 120 mg/dL" else 0
  exng = 1 if st.radio('Exercise Induced Angina',
                       ('yes', 'no')) == 'yes' else 0
  caa = st.slider('Number of major Vessels colored', 0, 4, 0, 1)

st.header('Predictions')

data = pd.read_csv("./heart_attack/heart.csv")
"""Only 14 attributes used:
13. #51 (thal):  3 = normal; 6 = fixed defect; 7 = reversable defect
"""


def preprocess_inputs(df):
  df = df.copy()

  # Split X and y: X being all of the explanatory variables, y being the outcome to predict
  X = df.drop('output', axis=1)
  y = df['output']

  # Train and test sets: These are four
  X_train, X_test, y_train, y_test = train_test_split(X,
                                                      y,
                                                      train_size=0.8,
                                                      shuffle=True,
                                                      random_state=0)

  X_copy = X_train
  # Scale X with a standard scaler: This scales the data so that we can use it more easily for seperation
  scaler = StandardScaler()
  X_train = pd.DataFrame(scaler.fit_transform(X_train),
                         index=X_train.index,
                         columns=X_train.columns)
  X_test = pd.DataFrame(scaler.transform(X_test),
                        index=X_test.index,
                        columns=X_test.columns)

  return X_train, X_test, y_train, y_test, X_copy


X_train, X_test, y_train, y_test, X_copy = preprocess_inputs(data)

input_list = [[age], [sex], [cp], [trtbps], [chol], [fbps], [restecg],
              [thalachh], [exng], [oldpeak], [slp], [caa], [thall]]

#Make a dict:
dct = {}

for i in range(0, len(X_train.columns)):
  dct[X_train.columns[i]] = input_list[i]

#input = pd.DataFrame(input_list, columns = X_train.columns)
input = pd.DataFrame.from_dict(dct)
input_manipulated = X_copy.append(input)

input_manipulated = input_manipulated.reset_index(drop=True)

scaler = StandardScaler()

x_input = pd.DataFrame(scaler.fit_transform(input_manipulated),
                       columns=input_manipulated.columns)

pred_knn = knn.predict(x_input)
pred_log = logistic.predict(x_input)
pred_rf = rf.predict(x_input)
pred_svc = svc.predict(x_input)

col4, col5, col6, col7 = st.columns(4)
col4.metric('KNN', pred_knn[242])
col5.metric('Log', pred_log[242])
col6.metric('RF', pred_rf[242])
col7.metric('SVC', pred_svc[242])

st.markdown(
  '''[Source](https://github.com/TechClubHSG/Fall22_PG_DSCS_How_to_ML.git)''')
