# Local Imports
import preprocess
# Primary Data Libraries
import jd_ds as jd
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
# py.init_notebook_mode()
# %matplotlib inline


# Settings
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# ipython --cache-size=5


orders = pd.read_csv('orders.csv', nrows = 200*10**3)
items = pd.read_csv('items.csv')
data = preprocess.process_data(orders, items)
del items; del orders

series = preprocess.select_series(data, time_period='day',revenue='net')


from preprocess import series_to_supervised
from matplotlib import pyplot

# transform the time series data into supervised learning
data = series_to_supervised(series, n_in=6).values
# evaluate
mae, y, yhat = walk_forward_validation(data, 13)
print('MAE: %.3f' % mae)
# plot expected vs preducted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()

def walk_forward_validation(data, n_test):
    from sklearn.metrics import mean_absolute_error
    # walk-forward validation for univariate data
    # Currently only works to output one predictor column
    predictions = list()
    # split dataset
    train, test = time_series_train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train] # list of row arrays
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, 1], predictions

# split a univariate dataset into train/test sets
def time_series_train_test_split(data, n_test):
    '''
    data must be np array format.

    If each row in the data set is a day, and n_test is 12, then trainY will be the last 12 rows/days.
    '''
    return data[:-n_test, :], data[-n_test:, :]


# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, testX):
    from xgboost import XGBRegressor
    from numpy import asarray
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict(testX)
    # Would I just return more than [0] to make it a multistep prediction? 
    return yhat[0]


params = {
    'max_depth': [],
    'eta': [], # learning rate?
    'objective':[],
    'num_class':[],
    'max_depth':[],
    'min_child_weight':[]
}
epochs = 10
