from numpy import array
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


number_of_samples = 4
length_of_sample = 750
sig = pd.read_csv("syg1.csv")
sig.head(10)
k = random.sample(range(int(2/3 * length_of_sample),int(len(sig)-1/3 * length_of_sample)), k =number_of_samples)
samples = []
input_feature= sig.iloc[:,[2,5]].values
sc= MinMaxScaler(feature_range=(0,1))
input_feature[:,:] = sc.fit_transform(input_feature[:,:])
for i in k:
    sample = input_feature[i-500:i+250,:]
    samples.append(sample)
num = 1
for sample in samples:
    fig = plt.figure()
    one = fig.add_subplot(2,1,1)
    plt.plot(sample[:,0])
    plt.title("Signal X")
    two = fig.add_subplot(2, 1, 2)
    plt.plot(sample[:,1])
    plt.title("Signal Y")
    fig.suptitle('Sample number %i' %num , fontsize=16)
    plt.show()
    num+= 1
    

#Forecasting using X and Y
l =int( 1/3 * len(sample))
input_data = []

for sample in samples:
    sample_conv = []
    X_one = sample[0:l,0]
    X_two = sample[l:2*l,0]
    X_prim =sample[2*l:3*l,0]
   
    Y_one = sample[0:l,1]
    Y_two = sample[l:2*l,1]
    Y_prim = sample[2*l:3*l,1]
    sample_conv = pd.DataFrame(list(zip(X_one, X_two, X_prim,Y_one,Y_two,Y_prim)), 
               columns =['X_one', 'X_two','X_prim','Y_one','Y_two','Y_prim'])
    input_data.append(sample_conv)

train_data_inputs = []
for input in input_data:
    train_data = []
    for i in range(len(X_one)):
        train_data.append(l)
        
    train_data =  [input.iloc[:,0].values,input.iloc[:,3].values,input.iloc[:,4].values]
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,250,3)
    train_data_y = input.iloc[:,1].values
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,250)
    sample = []
    sample.append(train_data)
    sample.append(train_data_y)
    train_data_inputs.append(sample)
    
    
#TEACHING THE MODEL
    
test_data_inputs = []

for input in input_data:
    test_data = []
    test_data =  [input.iloc[:,1].values,input.iloc[:,4].values,input.iloc[:,5].values]
    test_data = np.array(test_data)
    test_data= test_data.transpose()
    
    test_data = test_data.reshape(1,250,3)
    test_data_y = input.iloc[:,3].values
    test_data_y = np.array(test_data_y)
    test_data_y = test_data_y.reshape(1,250)
    sample = []
    sample.append(test_data)
    sample.append(test_data_y)
    test_data_inputs.append(sample)    

model = Sequential()
model.add(LSTM(units=30, return_sequences= True, input_shape=(train_data.shape[1],3)))
model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=30))
model.add(Dense(units=250))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

#Wybieramy na którym samplu będziemy uczyć
model.fit(train_data_inputs[0][0], train_data_inputs[0][1], epochs=50, batch_size=32)


#Testing the model
predicted_value_XY= model.predict(test_data_inputs[0][0])
pred = np.transpose(predicted_value_XY)

plt.plot(pred)

fitt = np.transpose(train_data_y)
plt.plot(fitt)

model_gru = Sequential()
model_gru.add(GRU(units=30, return_sequences= True, input_shape=(train_data.shape[1],3)))
model_gru.add(GRU(units=30, return_sequences=True))
model_gru.add(GRU(units=30))
model_gru.add(Dense(units=250))
model_gru.summary()


model_gru.compile(optimizer='adam', loss='mean_squared_error')

#Wybieramy na którym samplu będziemy uczyć
model_gru.fit(train_data_inputs[0][0], train_data_inputs[0][1], epochs=50, batch_size=32)


predicted_value= model_gru.predict(test_data_inputs[0][0])
pred = np.transpose(predicted_value)

plt.plot(pred)


input_feature= sig.iloc[0:30000,[1]].values
Y = input_feature[0:l,]

Y_prim = input_feature[l:2*l]



time_steps = 50

train_data_X = []
train_data_y = []
for i in range(len(X)-time_steps):
    t=[]
    g=[]
    for j in range(0,time_steps):
        
        t.append(X[[(i+j)], :])
        
    train_data_X.append(t)
    train_data_y.append(X_val[i])
    

train_data_X, train_data_y= np.array(train_data_X), np.array(train_data_y)

train_data_X = train_data_X.reshape(train_data_X.shape[0],time_steps, 1)
train_data_y = train_data_y.reshape(train_data_y.shape[0],1)


model = Sequential()
model.add(GRU(units=30, return_sequences= True, input_shape=(train_data_X.shape[1],1)))
model.add(GRU(units=30, return_sequences=True))
model.add(GRU(units=30))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(train_data_X, train_data_y, epochs=5, batch_size=32)

predicted_value_XY= model.predict(X_val)
plt.plot(predicted_value_XY[0:500])
plt.plot(y[0:500])

#Forecasting Using X

input_data = input_feature[:,1]


test_size=int(.3 * len(input_feature))
X=[]
y=[]
for i in range(len(input_feature)-time_steps-1):
    t=[]
    for j in range(0,time_steps):
        
        t.append(input_data[[(i+j)]])
    X.append(t)
    y.append(input_data[i+ time_steps,])
    
X, y= np.array(X), np.array(y)
X_test = X[:test_size+time_steps]

X = X.reshape(1,2500, 1)
Y = Y.reshape(1,2500)
X_test = X_test.reshape(X_test.shape[0],time_steps, 1)
print(X.shape)
print(X_test.shape)

model = Sequential()
model.add(GRU(units=30, return_sequences= True, input_shape=(X.shape[1],1)))
model.add(GRU(units=30))
model.add(Dense(units=2500))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=20, batch_size=10)

predicted_value_X= model.predict(X)
predicted_value_X = np.transpose(predicted_value_X)
plt.plot(predicted_value_X)

#Forecasting using Y'
input_data = input_feature[:,0:2]
time_steps = 1
test_size=int(.3 * len(input_feature))


test_size=int(.3 * len(input_feature))
X =input_data[:,0]
y = input_data[:,1]
plt.plot(X)
plt.show()
X,y = np.array(X), np.array(y)
X_test =X[:test_size+time_steps]

X = X.reshape(X.shape[0],time_steps, 1)
X_test = X_test.reshape(X_test.shape[0],time_steps, 1)
print(X.shape)
print(X_test.shape)


model = Sequential()
model.add(LSTM(units=30, return_sequences= True, input_shape=(X.shape[1],1)))
model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=30))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=2, batch_size=32)

predicted_value_Yprim= model.predict(X_test)



#SUMMARY
plt.plot(predicted_value_X, color= 'red')
#plt.plot(input_data[time_steps:test_size+(2*time_steps),], color='green')
plt.title("ECG signal prediction using X")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("ECG")
plt.show()
plt.plot(predicted_value_XY,color = 'blue')
#plt.plot(input_data[time_steps:test_size+(2*time_steps),], color='green')
plt.title("ECG signal prediction using XY")
plt.show()
plt.plot(predicted_value_Yprim, color= 'red')
plt.title("ECG signal prediction using Yprim")
plt.show()
