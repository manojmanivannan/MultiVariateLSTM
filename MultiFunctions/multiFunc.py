from os import pread
from altair.vegalite.v4.schema.core import Value
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import datetime


# fix random seed for reproducibility
np.random.seed(7)

def file_upload(name,separator=','):
    uploaded_file = st.sidebar.file_uploader('%s' % (name),key='%s' % (name),accept_multiple_files=False)
    content = False
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file,sep=separator)
            content = True
            return content, uploaded_df, uploaded_file.name.split('.')[0]
        except:
            try:
                uploaded_df = pd.read_excel(uploaded_file)
                content = True
                return content, uploaded_df, uploaded_file.name.split('.')[0]
            except:
                st.error('Please ensure file is .csv or .xlsx format and/or reupload file')
                return content, None, None
    else:
        return content, None, None

def extract_features_from_date(df):
    format_col1, format_col2 = st.columns(2)
    with format_col1: date_format = st.selectbox('Date format', ['YYYY MM DD','MM DD YYYY','DD MM YYYY'], index=0)
    with format_col2: separator = st.radio("Date separator", ('-', '/'))


    if date_format == 'MM DD YYYY':
        d_format = "%m#%d#%Y".replace('#',separator)
    if date_format == 'YYYY MM DD':
        d_format = "%Y#%m#%d".replace('#',separator)
    if date_format == 'DD MM YYYY':
        d_format = "%d#%m#%Y".replace('#',separator)
    
    with st.expander('Time format'):
        is_time = st.radio("Format Time column", ('Yes', 'No'), index=1)
        if is_time == 'Yes':
            format_col3, format_col4 = st.columns(2)
            with format_col3: time_format = st.selectbox('Time format',['HH MM','HH MM SS'], index=1)
            with format_col4: time_separator = st.radio("Time separator", ('.', ':'))

            if time_format == 'HH MM':
                t_format = "%H#%M".replace('#',time_separator)
            if time_format == 'HH MM SS':
                t_format = "%H#%M#%S".replace('#',time_separator)

    try:
        if is_time == 'No':
            df["DateTime"]=pd.to_datetime(df.Date,format=d_format)
        else:
            d_format = d_format+' '+t_format
            df['DateTime'] = df['Date'].str.cat(df['Time'],sep=" ")
            df['DateTime'] = pd.to_datetime(df['DateTime'], format=d_format)
 
        df.index=df['DateTime']
        df.drop('DateTime',axis=1,inplace=True)
        df = df.sort_index(axis=0)
        return df
    except Exception as e:
        st.write(f'{e}')
        st.stop()

def drop_columns_from_df(df):
    to_drop_columns = st.multiselect('Select columns to drop (Usually data/time or columns of no value)',list(df))
    df.drop(to_drop_columns,axis=1,inplace=True)

def fill_na_records(df):
    # nan_values = st.text_input('Comma separated values that are conisder NaN')
    # nan_values = nan_values.split(',')
    # df.replace(nan_values,np.nan)
    df.dropna(axis=0,inplace=True)

def load_sample_data():

    # load data
    def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')
    dataset = pd.read_csv('sample_data/pollution.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
        
    # save to file
    # dataset.to_csv('sample_data/pollution_cleaned.csv')
    return dataset



def create_dataset(dataset, look_back=1):
    '''
    For the features (x), we will always append the last 50 prices, 
    and for the label (y), we will append the next price. 
    Then we will use numpy to convert it into an array.
    '''
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def create_period_shift(df,remove_col,period=4):
    tmp_df = df.copy()
    col_names = list(df)
    for col in col_names:
        for i in range(period):
            new_col = (col+'_'+str(i+1))
            tmp_df[new_col] = tmp_df[col].shift(-(i+1))
    

    return tmp_df.drop([ x for x in col_names if x != remove_col],axis=1).dropna()

def filter_df(df,message='Select range to filter'):
    
    try:
        start_date = df.index.to_pydatetime()[0]
        end_date = df.index.to_pydatetime()[-1]
    except Exception as e:
        st.write("Please drop incompatible columns !")
        st.stop()
    slider_3, slider_4 = st.slider('%s' % (message),start_date,end_date,(start_date,end_date))
    df = df.loc[slider_3:slider_4][:]

    return df



import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_desc):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        fig = plt.figure(figsize=(12, 12))
        draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if m == 0:
                ax.text(n*h_spacing + left,layer_top*1.1 - m*v_spacing,layer_desc[n])

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)
