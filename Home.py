import streamlit as st

st.header("Introduction")

st.markdown('''
This is a quick introduction of this project.
In this project, I am trying to predict Apple, Inc. stock prices based on historic data using machine learning.

Here is a quick overview of the dataset:
''')

# Adapted from: https://www.kaggle.com/code/ryanholbrook/forecasting-with-machine-learning/tutorial
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import n_colors

from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from joblib import dump, load

apple_stock = pd.read_csv("./finance/apple.csv", index_col='Date')
apple_stock.index = pd.to_datetime(apple_stock.index)
apple_stock.sort_index(inplace=True)
with st.expander('Raw Data'):
  st.dataframe(apple_stock)

st.header("Method")
st.markdown('''
We want to predict the price of the next 7 days on the last 30 days' closing price.
''')


def make_lags(ts, lags, lead_time=1):
  return pd.concat(
    {f'y_lag_{i}': ts.shift(i)
     for i in range(lead_time, lags + lead_time)},
    axis=1)


# Thirty days of lag features
y = apple_stock.Close.copy()
X = make_lags(y, lags=30).fillna(0.0)
with st.expander('Manipulated Data'):
  st.dataframe(X)

st.markdown('''
I am doing a standard train-test split which is non-shuffled with test being 25%
''')

code = '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)'''
st.code(code, language='python')

tab1, tab2, tab3 = st.tabs(['LinReg', 'XGBoost', 'New Model'])


def create_visualization(y_test: pd.DataFrame,
                         y_pred: pd.DataFrame) -> go.Figure:
  fig = go.Figure()

  colorscale = n_colors('rgb(67, 198, 172)',
                        'rgb(25, 22, 84)',
                        len(y_pred),
                        colortype='rgb')

  for i, (index, row) in enumerate(y_pred.iterrows()):
    fig.add_trace(
      go.Scatter(x=pd.period_range(start=index,
                                   periods=len(row)).to_timestamp(),
                 y=row,
                 line=dict(color=colorscale[i])))
  fig.add_trace(
    go.Scatter(x=y_test.index, y=y_test.y_step_1, line=dict(color='black')))
  fig.update_layout(showlegend=False)

  return fig


apple_stock = pd.read_csv("./finance/apple.csv", index_col='Date')
apple_stock.index = pd.to_datetime(apple_stock.index)
apple_stock.sort_index(inplace=True)


def make_multistep_target(ts, steps):
  return pd.concat({f'y_step_{i + 1}': ts.shift(-i)
                    for i in range(steps)},
                   axis=1)


# 7 Day forecast
y = make_multistep_target(y, steps=7).dropna()

# Shifting has created indexes that don't match. Only keep times for
# which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)

# Create splits
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    shuffle=False)

with tab1:
  st.subheader('Linear Regression')
  ### Model 1: Basic Direct LinReg
  model = LinearRegression()
  model.fit(X_train, y_train)
  dump(model, './finance/dir_linreg.joblib')

  y_fit = pd.DataFrame(model.predict(X_train),
                       index=X_train.index,
                       columns=y.columns)
  y_pred = pd.DataFrame(model.predict(X_test),
                        index=X_test.index,
                        columns=y.columns)

  train_rmse = mean_squared_error(y_train, y_fit, squared=False)
  test_rmse = mean_squared_error(y_test, y_pred, squared=False)
  print((f"Train RMSE: {train_rmse:.2f}\n"
         f"Test RMSE: {test_rmse:.2f}"))

  st.plotly_chart(create_visualization(y_train, y_fit), use_container_width=True)
  st.plotly_chart(create_visualization(y_test, y_pred), use_container_width=True)