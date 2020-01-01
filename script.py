from numpy import array
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
from IPython import get_ipython
from sklearn.metrics import mean_squared_error as mse


#signal = pd.read_csv("syg1.csv")
#
#input_feature= signal.iloc[:,[2,5]].values

get_ipython().run_line_magic('matplotlib', 'qt')
#l = 750 #sample length
#sLength = l/3
#t = 4444 #sample
def compare(true,prediction):
    x = mse(true,prediction)
    return x
def show_signal(input_feature):
    fig = plt.figure()
    one = fig.add_subplot(2,1,1)
    plt.plot(input_feature[:,0])
    plt.title("Signal X")
    two = fig.add_subplot(2, 1, 2)
    plt.plot(input_feature[:,1])
    plt.title("Signal Y")
    plt.show()
    
def scale_signal(input_feature):
    sc= MinMaxScaler(feature_range=(0,1))
    input_feature[:,:] = sc.fit_transform(input_feature[:,:])
    
   
def pick_sample(t,l,input_feature):
    sample = input_feature[t-int(2/3*l):t+int(1/3*l),:]
    return sample
def prep_sample(sample,sLength,l):
    X_one = sample[0:sLength,0]
    X_two = sample[sLength:sLength*l,0]
    X_prim =sample[2*sLength:3*sLength,0]
   
    Y_one = sample[0:sLength,1]
    Y_two = sample[sLength:2*sLength,1]
    Y_prim = sample[2*sLength:3*sLength,1]
    df = pd.DataFrame(list(zip(X_one, X_two, X_prim,Y_one,Y_two,Y_prim)), 
               columns =['X_one', 'X_two','X_prim','Y_one','Y_two','Y_prim'])
    
    #Predicting Signal 1
    train_data =  [df.iloc[:,0].values,df.iloc[:,3].values,df.iloc[:,4].values] #X1 Y1 Y2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,sLength,3)
    train_data_y = df.iloc[:,1].values #X2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,sLength)
    
    test_data = []
    test_data =  [df.iloc[:,1].values,df.iloc[:,4].values,df.iloc[:,5].values] #X2 Y2 Y3
    test_data = np.array(test_data)
    test_data= test_data.transpose()
    test_data = test_data.reshape(1,sLength,3)
    
    test_data_y = df.iloc[:,2].values #X3
    test_data_y = np.array(test_data_y)
    test_data_y = test_data_y.reshape(1,sLength) 
    input_x = []
    input_x.append(train_data)
    input_x.append(train_data_y)
    input_x.append(test_data)
    input_x.append(test_data_y)
    
    #Predicting Signal 2
    train_data = []
    train_data =  [df.iloc[:,3].values,df.iloc[:,0].values,df.iloc[:,1].values] #Y1 X1 X2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,sLength,3)
    train_data_y = df.iloc[:,4].values #Y2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,sLength)
    
    test_data = []
    test_data =  [df.iloc[:,4].values,df.iloc[:,1].values,df.iloc[:,3].values] #Y2 X2 X3
    test_data = np.array(test_data)
    test_data= test_data.transpose()
    test_data = test_data.reshape(1,sLength,3)
    
    test_data_y = df.iloc[:,5].values #Y3
    test_data_y = np.array(test_data_y)
    test_data_y = test_data_y.reshape(1,sLength) 
    
    input_y = []
    input_y.append(train_data)
    input_y.append(train_data_y)
    input_y.append(test_data)
    input_y.append(test_data_y)
    
    return input_x, input_y

    
def create_model_cnn(train_data_input,train_data_output,act,opt,los,ks,sLength):
    model_cnn= Sequential()
    model_cnn.add(Conv1D(filters=50, kernel_size=ks, activation=act, input_shape=(train_data_input.shape[1],3)))
    model_cnn.add(MaxPooling1D(pool_size=100))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(2*sLength, activation=act))
    model_cnn.add(Dense(sLength))
    model_cnn.compile(optimizer=opt, loss=los)
    history_cnn = model_cnn.fit(train_data_input,train_data_output , epochs=100)
    
    return model_cnn, history_cnn

def create_model_lstm(train_data_input,train_data_output,act,opt,los,u,sLength):
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=u, return_sequences= True, input_shape=(sLength,3)))
    #model_lstm.add(LSTM(units=u, return_sequences=True))
    model_lstm.add(LSTM(units=u))
    model_lstm.add(Dense(2*sLength, activation=act))
    model_lstm.add(Dense(units=sLength))
    model_lstm.summary()
    model_lstm.compile(optimizer=opt, loss=los)
    history_lstm = model_lstm.fit(train_data_input, train_data_output, epochs=50, batch_size=32)
    return model_lstm, history_lstm

def create_model_gru(train_data_input,train_data_output,act,opt,los,u,sLength):
    model_gru = Sequential()
    model_gru.add(GRU(units=u, return_sequences= True, input_shape=(train_data_input.shape[1],3)))
    #model_gru.add(GRU(units=u, return_sequences=True))
    model_gru.add(GRU(units=u))
    model_gru.add(Dense(2*sLength, activation=act))
    model_gru.add(Dense(units=sLength))
    model_gru.summary()
    model_gru.compile(optimizer=opt, loss=los)
    history_gru = model_gru.fit(train_data_input, train_data_output, epochs=50, batch_size=32)
    return model_gru, history_gru
def show(pred):
    plt.plot(pred)

    
#show_signal(input_feature)
#scale_signal(input_feature)
#sample = pick_sample(t,input_feature)
#show_signal(sample)
#x,y = prep_sample(sample)
#
#model_cnn, history_cnn = create_model_cnn(x[0],x[1])
#prediction_cnn = model_cnn.predict(x[2])
#prediction_cnn = np.transpose(prediction_cnn)
#plt.plot(prediction_cnn)
#plt.plot(np.transpose(x[3]))
#
#
#model_cnn, history_cnn = create_model_cnn(y[0],y[1])
#prediction_cnn = model_cnn.predict(y[2])
#prediction_cnn = np.transpose(prediction_cnn)
#plt.plot(prediction_cnn)
#plt.plot(np.transpose(y[3]))
#
#
#
#model_lstm, history_lstm = create_model_lstm(x[0],x[1])
#prediction_lstm = model_lstm.predict(x[2])
#prediction_lstm = np.transpose(prediction_lstm)
#plt.plot(prediction_lstm)
#
#model_gru, history_gru = create_model_gru(x[0],x[1])
#prediction_gru = model_gru.predict(x[2])
#prediction_gru = np.transpose(prediction_gru)
#plt.plot(prediction_gru)
#
#fig = plt.figure()
