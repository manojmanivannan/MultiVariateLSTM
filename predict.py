import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math, time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model
from MultiFunctions.multiFunc import *
from MultiFunctions.information import information
from streamlit_echarts import st_echarts
import datetime as dt
from dateutil.relativedelta import relativedelta # to add days or years
import cufflinks as cf
from keras import backend as K
import io


st.set_page_config(
    page_title="Air Pollution Prediction", 
    page_icon='https://cdn-icons-png.flaticon.com/512/2640/2640447.png')

st.sidebar.subheader('Dataset')
separator = st.sidebar.selectbox('CSV column separator',[',',';','|'])
status, df, file_name = file_upload('Please upload a multivariate dataset',separator)
st.sidebar.markdown(information['profile'],unsafe_allow_html=True)

st.title('Multivariate Time-Series Prediction')
st.subheader('Using Keras Long-Short Term Memory (LSTM) Neural Network')
st.text("")
sample = False
with st.expander('Download from yahoo finance !'):
    company = st.selectbox('Select company',['AAPL','TSLA','BTC-USD'], index=1)
    st_dt, ed_dt = st.columns(2)
    with st_dt: start_date = st.date_input('From', dt.date(2021,1,1))
    with ed_dt: end_date = st.date_input('To', dt.date(2021,9,1))

    period1 = int(time.mktime(start_date.timetuple()))
    period2 = int(time.mktime(end_date.timetuple()))
    interval = '1d' # 1d, 1m

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df_downloaded = pd.read_csv(query_string)
    st.write(df_downloaded)
    st.download_button('Download stock data', df_downloaded.to_csv(), file_name=f'{company}.csv')
if not status:
    if st.radio('Use Sample Air pollution dataset', ['Yes','No'], index=1) == 'Yes':
        df = load_sample_data()
        status = True
        sample = True
    else:
        st.write('Please use the sidebar to upload your dataset !')
        st.write('There are many ways to download data, check out <a target="_blank" rel="noopener noreferrer" href=\'https://archive.ics.uci.edu/ml/index.php\'>this one</a> for example or download one from  <a target="_blank" rel="noopener noreferrer" href=\'https://github.com/manojmanivannan/DataScience/blob/master/Air-Quality/data/AirQualityUCI.csv\'>here</a>',unsafe_allow_html=True)


if status:
    
    with st.expander('View dataset'):
        st.write(df)
    if not sample:
        extract_features_from_date(df)
    
    drop_columns_from_df(df)
    fill_na_records(df)
    
    df = filter_df(df)

    with st.expander('View final dataset'):
        st.write(df)

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1: feature_cols = st.multiselect('Please select columns to plot',list(df),key='plot_col')
    with feat_col2: scale_plot = st.radio('Normalize plot [Min-Max]', ['Yes','No'], index=1)

    df_non_numeric = df.select_dtypes(include=['object'])
    ignore_cols = list(df_non_numeric)

    if feature_cols:


        new_feature_cols = [i for i in feature_cols if i not in ignore_cols]
        st.write(f'Can\'t plot {list(ignore_cols)} as it\'s not a numeric column')
  
        if scale_plot == 'Yes':
            nor_data = (df[new_feature_cols]-df[new_feature_cols].min())/(df[new_feature_cols].max()-df[new_feature_cols].min())
        else:
            nor_data = df[new_feature_cols]
            
        # nor_data.index.name = "Date"
        # data = nor_data.reset_index().melt('date',var_name='Feature',value_name='Value')
        # a = alt.Chart(data).mark_line().encode(
        #     x='date',
        #     y='Value',
        #     color='Feature'
        #     ).interactive()

        # st.altair_chart(a,use_container_width=True)

        fig_plot = nor_data.iplot(asFigure=True)
        fig_plot.update_layout(plot_bgcolor='rgba(17,17,17,0)',paper_bgcolor ='rgba(10,10,10,0)', legend_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_plot)


if status == True:
    col_names = list(df)

    st.title('Training')
    st.subheader('Parameters')
    col1, col2, col4 = st.columns(3)

    with col1:
        label_col = st.selectbox('Column to predict',col_names)
    with col2:
        test_size_ratio = st.number_input('Test size',0.01,0.99,0.25,0.05)

    with col4:
        no_layers = int(st.number_input('# of layers',2,10,2,1))

    if label_col == 'Date':
        st.write('Can\'t apply model on \'Date\' column. Select another column to proceed !')
        st.stop()
    
    with st.expander('Advanced Parameters'):
        col4_1, col4_2, col3 = st.columns(3)
        with col4_1:
            optimizer = st.selectbox('Solver',['Adam','Adagrad','RMSprop','Nadam'])
        with col4_2:
            loss = st.selectbox('Loss Function',['mean_squared_error','mean_absolute_error'])
        with col3:
            period = int(st.number_input('Lookback Period',1,10,4,1,help="Learning window"))
        col4_3, col4_4, col4_5 = st.columns(3)
        with col4_3:
            epochs = int(st.number_input('Training epoch',1,100,2,1))
        with col4_4:
            batchsize = int(st.number_input('Batch Size',1,100,50,5))
        with col4_5:
            treat_as_single = st.radio('Treat as single variate',['Yes','No'],index=1,help="Ignore all columns other than \"{}\" from the model".format(label_col))
            if treat_as_single == 'Yes': df = df[[label_col]]

    l_col1, l_col2, *l_colx = st.columns(no_layers)

    with l_col1: layer_1 = st.slider('Layer 1 Nodes', min_value=1, max_value=100, value=20, step=1)
    with l_col2: layer_2 = st.slider('Layer 2 Nodes', min_value=1, max_value=100, value=10, step=1)
    layer_list = [period,layer_1, layer_2]
    layer_desc = ['Input','Layer 1', 'Layer 2']
    if no_layers>2:
        i=2
        for each in l_colx:
            with each: 
                globals()['layer_'+str(i+1)] = st.slider(f'Layer {i+1} Nodes', min_value=1, max_value=100, value=10, step=1)
            layer_list.append(globals()['layer_'+str(i+1)])
            layer_desc.append('Layer '+str(i+1))
            i+=1
    layer_list.append(1)
    layer_desc.append('Output')


    # encode non-numberic data
    encoder = LabelEncoder()
    for each in ignore_cols:
        if each in list(df):
            df[each] =  encoder.fit_transform(df[each])


    shifted = create_period_shift(df,label_col,period=period)
    features_shifted = shifted.drop(label_col,axis=1)
    target_shifted = shifted[[label_col]]

    features_dataset = features_shifted.values
    target_dataset = target_shifted.values
    
    features_dataset = features_dataset.astype('float32')
    target_dataset = target_dataset.astype('float32')


    # normalize the dataset
    features_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0,1))

    tr_dataset = features_scaler.fit_transform(features_dataset)
    ts_dataset = target_scaler.fit_transform(target_dataset)
    # split into train and test sets
    test_size = int(len(shifted) * test_size_ratio)
    train_size = len(shifted) - test_size



    trainX, trainY = tr_dataset[:train_size, :], ts_dataset[:train_size, :]
    testX, testY = tr_dataset[train_size:, :], ts_dataset[train_size:, :]

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


    lstm_model=Sequential()
    lstm_model.add(LSTM(units=layer_1,return_sequences=True,input_shape=(trainX.shape[1], trainX.shape[2])))
    if no_layers == 2:
        lstm_model.add(LSTM(units=layer_2))
    else:
        lstm_model.add(LSTM(units=layer_2,return_sequences = True))
    if no_layers>2:
        i=2
        for each in l_colx:
            if i>=len(l_colx):
                lstm_model.add(LSTM(units=globals()['layer_'+str(i+1)]))
            else:
                lstm_model.add(LSTM(units=globals()['layer_'+str(i+1)],return_sequences = True))
            i+=1
    lstm_model.add(Dense(1))

    with st.expander('View model layer diagram'):
        plot_model(lstm_model,'sample_data/image.png',show_shapes=True)
        st.image('sample_data/image.png')
    with st.expander('View network architecture'):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')
        draw_neural_net(ax, .1, .9, .1, .9, layer_list, layer_desc)
        st.pyplot(fig)

    lstm_model.compile(loss=loss, optimizer=optimizer)
    with st.spinner('Training Model..'):
        lstm_model.fit(trainX, trainY, epochs=epochs, batch_size=batchsize,validation_data=(testX, testY), verbose=2)

        # make predictions
        trainPredict = lstm_model.predict(trainX)
        testPredict = lstm_model.predict(testX)
    
        # invert predictions
        trainPredict = target_scaler.inverse_transform(trainPredict)
        trainY = target_scaler.inverse_transform(trainY)

        
        testPredict = target_scaler.inverse_transform(testPredict)
        testY = target_scaler.inverse_transform(testY)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))


    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(target_shifted)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[:len(trainPredict), :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(target_shifted)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict):len(target_shifted), :] = testPredict

    result_df = pd.DataFrame(target_scaler.inverse_transform(ts_dataset),columns=[label_col])
    result_df['Prediction on training set']=trainPredictPlot
    result_df['Prediction on test set'] = testPredictPlot

   
    result_df.index = df[[label_col]].shift(-(period)).dropna().index
    st.title('Result')
    result1, result2 = st.columns(2)
    with result1: st.write('Train Score: %.2f RMSE' % (trainScore))
    with result2: st.write('Test Score: %.2f RMSE' % (testScore))
    
    with result1: st.subheader('Plot')
    with result2: scale_result = st.radio('Normalize result [Min-Max]', ['Yes','No'], index=1)

    if scale_result == 'Yes':
        nor_result = (result_df-result_df.min())/(result_df.max()-result_df.min())
    else:
        nor_result = result_df


    # st.line_chart(nor_result)
    fig_result = nor_result.iplot(asFigure=True)
    fig_result.update_layout(plot_bgcolor='rgba(17,17,17,0)',paper_bgcolor ='rgba(10,10,10,0)',legend_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_result)

    with st.expander('View result dataset'):
        st.write(nor_result)

    if file_name == None: file_name = 'air_pollution'
    st.download_button('Download result', nor_result.to_csv(), file_name=f'{file_name}_prediction_results.csv')
    st.download_button('Download trained Model', nor_result.to_csv(), file_name=f'{file_name}_prediction_results.csv')
    K.clear_session()
    del lstm_model
