# Primary Data Libraries
import pandas as pd 

#==================================================================
#              Pre Processing
#==================================================================
def process_data(orders_df, items_df):
    '''Preprocesses orders dataframe and items dataframe for time series analysis'''
    SKIP_OWNERS = [
        'VBFB.czO',
        'VBFB.esO',
        'VBFB.huO',
        'VBFB.roO',
        'VBFB.skO',
        'VBRU.czO',
        'VBRU.deO',
        'VBRU.skO']

    # Instructed to remove these owners
    orders_df = orders_df[~orders_df.owner.str.contains('|'.join(SKIP_OWNERS))]
    # This line is only if data set has first column called: "Unamed: 0"
    orders_df.drop(columns=orders_df.columns[0], inplace=True)
    # Dropped because I didn't think they were useful for analysis.
    orders_df.drop(columns=['transport_id', 'transport'], inplace=True)
    # Change to string because sci-notation of long number prevented reading
    orders_df.order_id = orders_df.order_id.astype(str) 
    orders_df.date = pd.to_datetime(orders_df.date)
    orders_df['net_revenue'] = orders_df.quantity * orders_df.unit_price_vat_excl
    orders_df['gross_revenue'] = orders_df.quantity * orders_df.unit_cogs
    orders_df['margin_revenue'] = orders_df.quantity * (orders_df.unit_price_vat_excl - orders_df.unit_cogs)

    items_cols_keep = ['item_code','item_name','style','name','group0','group1','category']

    data = pd.merge( orders_df, items_df[items_cols_keep], on='item_code')
    return data


#==================================================================
#              Utility Functions
#==================================================================
def select_series(data, time_period='day', group0=None, group1=None, site=None, category=None, revenue = 'net'):

    '''
    Function that will return a time series filtered down to focus on the time period or category we would like to predict. Such as predict monthly sales overall, or predict monthly sales for the category Shoes.
    
    Alternatively filter down so it is specific to a certain region or website, ie: focusing on the "Where
    
    time_period options: 'day','week','month','quarter','year'
    
    revenue options: 'net','gross','margin'
    '''
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    time_dict = {
        'day': 'D',
        'week': 'W',
        'month': 'MS',
        'quarter': 'QS',
        'year': 'YS'}
    
    revenue_dict = {
        'net':'net_revenue',
        'gross':'gross_revenue',
        'margin':'margin_revenue'}

    # Filter data based on function parameters

    if site and isinstance(site,list):
        data = data[data.site.isin(site)]
    if group0 and isinstance(group0,list):
        data = data[data.group0.isin(group0)]
    if group1 and isinstance(group1,list):
        data = data[data.group1.isin(group1)]
    if category and isinstance(category,list):
        data = data[data.category.isin(category)]


    if site and isinstance(site,str):
        data = data[data.site == site]
    if group0 and isinstance(group0,str):
        data = data[data.group0 == group0]
    if group1 and isinstance(group1,str):
        data = data[data.group1 == group1]
    if category and isinstance(category,str):
        data = data[data.category == category]

    data = data.set_index('date')
    data = data[revenue_dict[revenue]].resample(time_dict[time_period]).sum()
    data = pd.DataFrame(data).reset_index()
    return data


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    from pandas import DataFrame
    from pandas import concat
    from pandas import Series
    '''
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X) to predict from.
        n_out: Number of observations as output (y). The predicted observations.
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.

    Resources:
    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    '''

    n_vars = 1 if isinstance(data, Series) else data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [f'{data.name if n_vars==1 else data.columns[j]}(t-{i})' for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
        if i == 0:
            names += [f'{data.name if n_vars==1 else data.columns[j]}(t)' for j in range(n_vars)]
        else:
            names += [f'{data.name if n_vars==1 else data.columns[j]}(t+{i})' for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
