# Local Imports
import preprocess
# Primary Data Libraries
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot
import plotly.offline as py
import warnings
warnings.filterwarnings("ignore")
# py.init_notebook_mode()
# %matplotlib inline


# Settings
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# ipython --cache-size=5


orders = pd.read_csv('orders.csv')
items = pd.read_csv('items.csv')
data = preprocess.process_data(orders, items)
del items; del orders
# data = data.loc['2019-06':'2019-09'] # leaving out covid months, halves data set so my computer doesn't freeze
# data.set_index(['date'], inplace=True)
# data.reset_index(level=0, inplace=True)
daily = preprocess.select_series(data, time_period='day',revenue='net', group0='Apparel')
daily.plot(x='date',y='net_revenue')



'''
Data from 2019-06-01 to 2020-09-24

Unique of Group0: 
Group0 = ['Apparel', 'Equipment', 'Other', 'Footwear', 'Nutrition','Services']
and Nans

Unique of Category:
Category = ['football', 'fitness', 'lifestyle', 'running']
and Nans

Unique of Group1: 
    'Pants', nan, 'Football shoes', 'T-Shirts', 'Football socks',
    'Caps', 'Sweatshirts', 'Jackets', 'Running shoes', 'Fitness Shoes',
    'Jerseys', 'Other Equipment', 'Tracksuits', 'Guards',
    'Other Footwear', 'Slides', 'Socks', 'Balls', 'Other Apparel',
    'Bras', 'Underwear', 'Gloves', 'Training', 'Glasses', 'Vests',
    'Beanies', 'Bags', 'Skirts', 'Lamps', 'Dress', 'Sports gloves',
    'Gymsacks', 'Backpacks', 'Hydrate', 'Other nutrition'

Unique of Site:
    '11teamsports.ro', 'Other', '11teamsports.cz', '11teamsports.hu',
    '11teamsports.sk', 'top4football.fr', 'top4sport.cz',
    'top4fitness.sk', 'top4running.hu', 'top4fitness.hu',
    'top4running.cz', 'top4running.es', 'top4running.de',
    'top4running.sk', 'top4running.hr', 'top4running.com',
    'top4running.fr', 'top4fitness.es', 'top4street.cz',
    'top4fitness.cz', 'top4street.sk', 'top4street.hu',
    'top4running.ro', 'top4fitness.ro', 'top4street.ro',
    'top4fitness.de', 'top4football.de', '11teamsports.hr',
    'top4football.com', 'top4fitness.com', 'top4fitness.hr',
    'top4football.es', 'top4running.at', 'top4fitness.at',
    'top4fitness.fr', 'top4football.it', 'top4running.it',
    'top4fitness.it'
'''
def prophet_columns(df):
    '''Take  dataframe of only datetime column and value column and rename their columns to the form FBProphet requires'''
    df.columns = ['ds','y']
    return df
#==================================================================
#              Prophet Prototype
#==================================================================

train = daily.rename(columns={'date':'ds','net_revenue':'y'})
train.y = np.log(train.y +1 )
train.y.hist()
validate = train.tail(60)
train.drop(train.tail(60).index, inplace=True)

# prophet_columns(daily[ ('2019-06'<= daily.date) & (daily.date<='2020-06')])
pr = Prophet(yearly_seasonality=False)
model = pr.fit(train)

future = model.make_future_dataframe(periods=30*2, freq='D', include_history=False)

forecast = model.predict(future)
forecast.yhat = np.exp(forecast.yhat-1)


timeseries_evaluation_metrics_func(validate.y, forecast.yhat)



fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
fig
fig2 = model.plot_components(forecast)


# Detrend using HP Filter
# Used for removing short term fluctuations
from statsmodels.tsa.filters.hp_filter import hpfilter
cycle, trend = hpfilter(daily['net_revenue'], lamb=1600)
fig = plt.plot(daily['date'], (daily['net_revenue'] - trend))
plt.xticks(rotation=90)
plt.show()


# Detect Seasonality
## Using Box Plots
# create a boxplot of monthly data

series = daily.set_index('date')['2019']
groups = series.groupby(pd.Grouper(freq='M'))
months = pd.concat([pd.DataFrame(x[1].values) for x in groups], axis=1)
months = pd.DataFrame(months)
months.columns = range(6,13)
months.boxplot()
plt.show()

'''
Can see from boxplot seasonality that there are more outliers in summer, but higher mean sales in December. Using only 2019 Data
'''

# Autocorrelation plot to check randomness in data. 
# https://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
pd.plotting.autocorrelation_plot(series)
'''
Only shows statistically significant autocorrelation with lag of first 1-5 days
'''


# Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(
    preprocess.select_series(data, time_period='day',revenue='net', group0='Apparel', site='top4running.cz').set_index('date')
    , model='add')
result.plot()

def timeseries_evaluation_metrics_func(y_true, y_pred):
    from sklearn import metrics
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    #print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}', end='\n\n')



#==================================================================
#              LSTM Trial
#==================================================================
import time
import tensorflow as tf
from sklearn import preprocessing
seed = 124
tf.random.set_seed(seed)
np.random.seed(seed)

# Select data for LSTM
series_lstm = preprocess.select_series(data, time_period='day',revenue='net', group0='Apparel', site='top4running.cz').set_index('date')

# Scale data for LSTM
scaler = preprocessing.MinMaxScaler()
x_rescaled = scaler.fit_transform(series_lstm.values.reshape(-1,1))

def custom_ts_univariate_data_prep(dataset, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start,end):
        indicesx = range(i-window, i)
        X.append(np.reshape(dataset[indicesx], (window,1)))
        indicesy = range(i,i+horizon)
        y.append(dataset[indicesy])
    return np.array(X), np.array(y)

univar_hist_window = 30 # days
horizon = 1 
train_split = int( x_rescaled.shape[0]*0.7)
x_train, y_train = custom_ts_univariate_data_prep(x_rescaled, 0, train_split, univar_hist_window, horizon)

x_val, y_val = custom_ts_univariate_data_prep(x_rescaled, train_split, None, univar_hist_window, horizon)


batch_size = 256
buffer_size = 150
train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_univariate = val_univariate.batch(batch_size).repeat()

modelpath = f'/media/mint/HD Shared/LHL-coursework/LHL Final Project/saved_models/LSTM_univariate_{time.strftime("%Y%m%d-%H%M%S")}.h5'

lstm_model = tf.keras.models.Sequential([
tf.keras.layers.LSTM(100, input_shape = x_train.shape[-2:], return_sequences=True),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.LSTM(units=50, return_sequences=False),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(units=1)
])

lstm_model.compile(optimizer='adam',loss='mse')

evaluation_interval = 100
epochs = 150
history = lstm_model.fit(train_univariate, epochs=epochs, steps_per_epoch=evaluation_interval, validation_data=val_univariate, validation_steps=50, verbose = 1, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=10, verbose=1, mode='min'),
    tf.keras.callbacks.ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True,mode='min',verbose=0)])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.show()

#==================================================================
#    Establish Baseline Performance with Persistance Model
#==================================================================
import sklearn.metrics as skm
series = preprocess.select_series(data, time_period='day',revenue='net', group0='Apparel')

df = preprocess.series_to_supervised(series.set_index('date')['net_revenue'])


def model_persistence(x):
    return x

def df_train_test(df, train_pct):
    ''' Create train test split keeping dataframe form instead of numpy array'''
    train_size = int(len(df) * train_pct)
    train, test = df.iloc[:train_size,], df.iloc[train_size:,]
    trainX, trainY = train.iloc[:,0], train.iloc[:,1]
    testX, testY = test.iloc[:,0], test.iloc[:,1]
    return trainX, trainY, testX, testY

trainX, trainY, testX, testY = df_train_test(df, 0.66)

predictions = []
for x in testX:
    yhat = model_persistence(x)
    predictions.append(yhat)
predictions = pd.Series(predictions, index=testY.index)
test_score = np.sqrt(skm.mean_squared_error(testY, predictions))

plt.plot(trainY)
plt.plot(testY)
plt.plot(predictions)
plt.show()