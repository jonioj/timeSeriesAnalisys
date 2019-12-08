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


signal = pd.read_csv("syg1.csv")

input_feature= signal.iloc[:,[2,5]].values


l = 750
sLength = 250
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
    
   
def pick_sample(t,input_feature):
    sample = input_feature[t-int(2/3*l):t+int(1/3*l),:]
    return sample
def prep_sample(sample):
    X_one = sample[0:sLength,0]
    X_two = sample[sLength:sLength*l,0]
    X_prim =sample[2*sLength:3*sLength,0]
   
    Y_one = sample[0:sLength,1]
    Y_two = sample[sLength:2*sLength,1]
    Y_prim = sample[2*sLength:3*sLength,1]
    df = pd.DataFrame(list(zip(X_one, X_two, X_prim,Y_one,Y_two,Y_prim)), 
               columns =['X_one', 'X_two','X_prim','Y_one','Y_two','Y_prim'])
    
    
    train_data =  [df.iloc[:,0].values,df.iloc[:,3].values,df.iloc[:,4].values] #X1 Y1 Y2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,250,3)
    train_data_y = df.iloc[:,1].values #X2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,250)
    
    test_data = []
    test_data =  [df.iloc[:,1].values,df.iloc[:,4].values,df.iloc[:,5].values] #X2 Y2 Y3
    test_data = np.array(test_data)
    test_data= test_data.transpose()
    test_data = test_data.reshape(1,250,3)
    
    test_data_y = df.iloc[:,2].values #X3
    test_data_y = np.array(test_data_y)
    test_data_y = test_data_y.reshape(1,250) 
    input_x = []
    input_x.append(train_data)
    input_x.append(train_data_y)
    input_x.append(test_data)
    input_x.append(test_data_y)
    
    return input_x

def create_model_cnn(train_data_input,train_data_output,filters,kernel_size,activation,loss,epochs):
    model_cnn_x = Sequential()
    model_cnn_x.add(Conv1D(filters=50, kernel_size=100, activation='relu', input_shape=(train_data_input.shape[1],3)))
    model_cnn_x.add(MaxPooling1D(pool_size=100))
    model_cnn_x.add(Flatten())
    model_cnn_x.add(Dense(500, activation='relu'))
    model_cnn_x.add(Dense(250))
    model_cnn_x.compile(optimizer='adam', loss='mse')
    model_cnn_x.fit(train_data_input,train_data_output , epochs=100)
    
    return model_cnn_x


show_signal(input_feature)
scale_signal(input_feature)
sample = pick_sample(5324,input_feature)
show_signal(sample)
x = prep_sample(sample)
model = create_model_cnn(x[0],x[1])
prediction = model.predict(x[0])
prediction = np.transpose(prediction)
plt.plot(prediction)
