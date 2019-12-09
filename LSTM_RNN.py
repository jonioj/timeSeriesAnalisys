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



from patsy import dmatrices

#Wybieramy ilsc probek
number_of_samples = 4
#wybieramy długosc probek
length_of_sample = 750
#wczytujemy dane
sig = pd.read_csv("syg1.csv")
sig.head(10)
#Wybieramy próbki przypadkowo
k = random.sample(range(int(2/3 * length_of_sample),int(len(sig)-1/3 * length_of_sample)), k =number_of_samples)
samples = []
#skalujemy dane
input_feature= sig.iloc[:,[2,5]].values
sc= MinMaxScaler(feature_range=(0,1))
input_feature[:,:] = sc.fit_transform(input_feature[:,:])
for i in k:
    sample = input_feature[i-500:i+250,:]
    samples.append(sample)
num = 1


#Wizualizacja danych
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
def prep_sample(sample,l):
    X_one = sample[0:l,0]
    X_two = sample[l:2*l,0]
    X_prim =sample[2*l:3*l,0]
   
    Y_one = sample[0:l,1]
    Y_two = sample[l:2*l,1]
    Y_prim = sample[2*l:3*l,1]
    df = pd.DataFrame(list(zip(X_one, X_two, X_prim,Y_one,Y_two,Y_prim)), 
               columns =['X_one', 'X_two','X_prim','Y_one','Y_two','Y_prim'])
    train_data = []
    for i in range(len(X_one)):
        train_data.append(l)
        
    train_data =  [input.iloc[:,0].values,input.iloc[:,3].values,input.iloc[:,4].values] #X1 Y1 Y2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    train_data = train_data.reshape(1,250,3)
    train_data_y = input.iloc[:,1].values #X2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,250)
    train_data_x = []
    train_data_x.append(train_data)
    train_data_x.append(train_data_y)
    
    train_data = []
    for i in range(len(X_one)):
        train_data.append(l)
        
    train_data =  [input.iloc[:,3].values,input.iloc[:,0].values,input.iloc[:,1].values]#Y1 X1 X2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,250,3)
    train_data_y = input.iloc[:,4].values #Y2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,250)
    train_data_yy = []
    train_data_yy.append(train_data)
    train_data_yy.append(train_data_y)
    return train_data_x, train_data_yy
    


#podzial danych na x1 x2 x3 y1 y2 y3
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
    
l =int( 2/3 * length_of_sample)

var_data = []
for sample in samples:
    x_train = sample[:l,0].tolist()
    y_train = sample[:l,1].tolist()
    x_test = sample[l:len(sample),0].tolist()
    y_test = sample[l:len(sample),1].tolist()
    data=[]
    data.append(x_train)
    data.append(y_train)
    data.append(x_test)
    data.append(y_test)
    var_data.append(data)


#Dane do nauki predykcji X
train_data_inputs_x = []
for input in input_data:
    train_data = []
    for i in range(len(X_one)):
        train_data.append(l)
        
    train_data =  [input.iloc[:,0].values,input.iloc[:,3].values,input.iloc[:,4].values] #X1 Y1 Y2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,250,3)
    train_data_y = input.iloc[:,1].values #X2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,250)
    sample = []
    sample.append(train_data)
    sample.append(train_data_y)
    train_data_inputs_x.append(sample)


#Dane do nauki predykcji Y
train_data_inputs_y = []

for input in input_data:
    train_data = []
    for i in range(len(X_one)):
        train_data.append(l)
        
    train_data =  [input.iloc[:,3].values,input.iloc[:,0].values,input.iloc[:,1].values]#Y1 X1 X2
    train_data = np.array(train_data)
    train_data= train_data.transpose()
    
    train_data = train_data.reshape(1,250,3)
    train_data_y = input.iloc[:,4].values #Y2
    train_data_y = np.array(train_data_y)
    train_data_y = train_data_y.reshape(1,250)
    sample = []
    sample.append(train_data)
    sample.append(train_data_y)
    train_data_inputs_y.append(sample)
#Dane do predykcji X
test_data_inputs_x = []

for input in input_data:
    test_data = []
    test_data =  [input.iloc[:,1].values,input.iloc[:,4].values,input.iloc[:,5].values] #X2 Y2 Y3
    test_data = np.array(test_data)
    test_data= test_data.transpose()
    
    test_data = test_data.reshape(1,250,3)
    test_data_y = input.iloc[:,2].values #X3
    test_data_y = np.array(test_data_y)
    test_data_y = test_data_y.reshape(1,250)
    sample = []
    sample.append(test_data)
    sample.append(test_data_y)
    test_data_inputs_x.append(sample) 
    
#Dane do predykcji y  
test_data_inputs_y = []

for input in input_data:
    test_data = []
    test_data =  [input.iloc[:,4].values,input.iloc[:,1].values,input.iloc[:,2].values] #Y2 X2 X3
    test_data = np.array(test_data)
    test_data= test_data.transpose()
    
    test_data = test_data.reshape(1,250,3)
    test_data_y = input.iloc[:,5].values #Y3
    test_data_y = np.array(test_data_y)
    test_data_y = test_data_y.reshape(1,250)
    sample = []
    sample.append(test_data)
    sample.append(test_data_y)
    test_data_inputs_y.append(sample) 

n = 2 #wybór próbki
def create_model_cnn(train_data_input,train_data_output):
    model_cnn_x = Sequential()
    model_cnn_x.add(Conv1D(filters=50, kernel_size=100, activation='relu', input_shape=(train_data.shape[1],3)))
    model_cnn_x.add(MaxPooling1D(pool_size=100))
    model_cnn_x.add(Flatten())
    model_cnn_x.add(Dense(500, activation='relu'))
    model_cnn_x.add(Dense(250))
    model_cnn_x.compile(optimizer='adam', loss='mse')
    model_cnn_x.fit(train_data_input,train_data_output , epochs=100)
    return model_cnn_x

prediction_cnn_x = model_cnn_x.predict(test_data_inputs_x[n-1][0])
prediction_cnn_x = np.transpose(prediction_cnn_x)
plt.plot(prediction_cnn_x)
plt.plot(np.transpose(test_data_inputs_x[n-1][1]))
plt.legend(['Prediction', 'Original'])

train_data.shape[1]
#Model lstm do predykcji x
model_lstm_x = Sequential()
model_lstm_x.add(LSTM(units=30, return_sequences= True, input_shape=(train_data.shape[1],3)))
model_lstm_x.add(LSTM(units=30, return_sequences=True))
model_lstm_x.add(LSTM(units=30))
model_lstm_x.add(Dense(500, activation='relu'))
model_lstm_x.add(Dense(units=250))
model_lstm_x.summary()
model_lstm_x.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
history_lstm_x = model_lstm_x.fit(train_data_inputs_x[0][0], train_data_inputs_x[0][1], epochs=50, batch_size=32)

#Test modelu
fig2 = plt.figure()
predicted_value_lstm_x= model_lstm_x.predict(test_data_inputs_x[0][0])
pred_lstm_x = np.transpose(predicted_value_lstm_x)
test_data_inputs_x[0][1]
plt.plot(pred_lstm_x)
plt.plot(np.transpose(test_data_inputs_x[0][1]))
plt.legend(['Prediction', 'Original'])
fig2.show()

#Wizualizacja procesu uczenia

plt.plot(history_lstm_x.history['loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



#Model lstm do predykcji y
model_lstm_y = Sequential()
model_lstm_y.add(LSTM(units=30, return_sequences= True, input_shape=(train_data.shape[1],3)))
model_lstm_y.add(LSTM(units=30, return_sequences=True))
model_lstm_y.add(LSTM(units=30))
model_lstm_y.add(Dense(units=250))
model_lstm_y.summary()

model_lstm_y.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

history_lstm_y = model_lstm_y.fit(train_data_inputs_y[0][0], train_data_inputs_y[0][1], epochs=50, batch_size=32)

fig3 = plt.figure()
predicted_value_lstm_y= model_lstm_y.predict(test_data_inputs_y[0][0])
pred_lstm_y = np.transpose(predicted_value_lstm_y)
test_data_inputs_y[0][1]
plt.plot(pred_lstm_y)
plt.plot(np.transpose(test_data_inputs_y[0][1]))
plt.legend(['Prediction', 'Original'])
fig3.show()


#Model GRU do predykcji X
model_gru_x = Sequential()
model_gru_x.add(GRU(units=30, return_sequences= True, input_shape=(train_data.shape[1],3)))
model_gru_x.add(GRU(units=30, return_sequences=True))
model_gru_x.add(GRU(units=30))
model_gru_x.add(Dense(500, activation='relu'))
model_gru_x.add(Dense(units=250))
model_gru_x.summary()


model_gru_x.compile(optimizer='adam', loss='mean_squared_error')

#Wybieramy na którym samplu będziemy uczyć
history_gru_x = model_gru_x.fit(train_data_inputs_x[0][0], train_data_inputs_x[0][1], epochs=50, batch_size=32)


fig4 = plt.figure()
predicted_gru_x= model_gru_x.predict(test_data_inputs_x[0][0])
predicted_gru_x = np.transpose(predicted_gru_x)
plt.plot(predicted_gru_x)
plt.plot(np.transpose(test_data_inputs_x[0][1]))
plt.legend(['Prediction', 'Original'])
fig4.show()





#Model GRU do predykcji y
model_gru_y = Sequential()
model_gru_y.add(GRU(units=30, return_sequences= True, input_shape=(train_data.shape[1],3)))
model_gru_y.add(GRU(units=30, return_sequences=True))
model_gru_y.add(GRU(units=30))
model_gru_y.add(Dense(units=250))
model_gru_y.summary()


model_gru_y.compile(optimizer='adam', loss='mean_squared_error')

#Wybieramy na którym samplu będziemy uczyć
history_gru_y = model_gru_y.fit(train_data_inputs_y[0][0], train_data_inputs_y[0][1], epochs=50, batch_size=32)



fig5 = plt.figure()
predicted_gru_y= model_gru_y.predict(test_data_inputs_y[0][0])
predicted_gru_y = np.transpose(predicted_gru_y)
plt.plot(predicted_gru_y)
plt.plot(np.transpose(test_data_inputs_y[0][1]))
plt.legend(['Prediction', 'Original'])
fig5.show()



#PREDYKCJA VAR

def difference(dataset, interval):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

def inverse_difference(last_ob, value):
	return value + last_ob


x_train = sample[250:l,0].tolist()
y_train = sample[250:l,1].tolist()
y_test = sample[l:len(sample),1].tolist()
x_test = sample[l:len(sample),0].tolist()
plt.plot(x_train)
plt.plot(y_train)
diff = difference(y_train, 80)
plt.plot(diff)
inverted = [inverse_difference(y_train[i], diff[i]) for i in range(len(diff))]
plt.plot(inverted)
data = []
for i in range(len(x_train)):
    row = [x_train[i],y_train[i]]
    data.append(row)
data_exog = y_test

model = VARMAX(data, exog=data_exog, order=(1, 0))
model_fit = model.fit(disp=False)
# make prediction


print(yhat)
y_test= np.array(y_test).reshape(250,1)
forecast = model_fit.forecast(250,exog = y_test)
plt.plot(forecast)

from statsmodels.tsa.statespace.varma import VARMA
from statsmodels.tsa.vector_ar.var_model import VAR
data = x_train
data_exog = y_train
model = VAR(data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=50)
plt.plot(yhat)

dta = sm.datasets.webuse('lutkepohl2', 'https://www.stata-press.com/data/r12/')
dta.index = dta.qtr
endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]

exog = endog['dln_consump']
mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(2,0), trend='nc', exog=exog)
res = mod.fit(maxiter=1000, disp=False)
print(res.summary())





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
