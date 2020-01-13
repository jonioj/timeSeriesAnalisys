import tkinter as tk
from tkinter import *
from tkinter import filedialog
import script
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


root = tk.Tk()
root.title("Causality assessment")
root.geometry("1000x600")

def read_entry():
    print(e1.get())
# signal = loadmat(r'C:\Users\jonas\OneDrive\Dokumenty\Studia\INZ\Sygnaly\20.mat')
# signal = signal['data']
# input_feature = signal[:,0:2]   
    
# signal = pd.read_csv('syg1.csv',sep=r'\s*,\s*')



# input_feature= signal.iloc[:,[2,5]]
# input_feature.to_csv('signals.csv', index = False)
# input_feature = script.scale_signal(input_feature)
# script.show_signal(input_feature) 
# signal = pd.read_csv('signals.csv')
# x = signal.iloc[:,:].values
def opens():
    global path
    global signal
    global input_feature
    try:
        root.filename = filedialog.askopenfilename(initialdir  = "/",title="Signal")
        path = root.filename
        signal = pd.read_csv(path)
        input_feature= np.array(signal)
        input_feature = script.scale_signal(input_feature)
        script.show_signal(input_feature)
    except:
        print('Something is worng')
        

def sample():
    
        
        global l
        t = int(e2.get())
        l = int(((int(e1.get()))/3)*3)
        global s
        global inputs_x
        global inputs_y
        inputs_x = []
        inputs_y =[]
        
        
        for i in range(10):
            s = script.pick_sample(t-i*50,l,input_feature)
            #script.show_signal(s)
            #print(l)
            input_x, input_y = script.prep_sample(s,int(l/3),l)
            inputs_x.append(input_x)
            inputs_y.append(input_y)
        
        print("ready to analyze")
    
        
def analyse():
    global model
    global history
    global prediction_cnn
    global prediction_lstm
    global prediction_gru
    global pred
    global inpt
    
    
    if (var2.get() == 'Signal X'):
        inpt = inputs_x
    elif (var2.get() == 'Signal Y'):
        inpt = inputs_y
    
#    try:
    if (method.get() =='CNN'):
        
        model = script.create_model_cnn(inpt[0][0],inpt[0][1],activation.get(),optimizer.get(),loss.get(),int(e4.get()),int(l/3))
        for i in range(len(inputs_x)):
            model,history = script.train_model_cnn(model,inpt[i][0],inpt[i][1])
        # prediction_cnn = model.predict(inpt[2])
        # print(prediction_cnn)
        # pred = np.transpose(prediction_cnn)
        # fig3 = plt.figure()
        # plt.plot(pred)
        # fig3.show()
        
        
    elif(method.get() == 'LSTM'):
        model= script.create_model_lstm(inpt[0][0],inpt[0][1],activation.get(),optimizer.get(),loss.get(),int(e3.get()),int(l/3))
        for i in range(len(inputs_x)):
            model.history = script.train_model_lstm(model,inpt[i][0],inpt[i][1])
        # prediction_lstm = model.predict(inpt[2])
        # print(prediction_lstm)
        # pred = np.transpose(prediction_lstm)
        # fig4 = plt.figure()
        # plt.plot(pred)
        # fig4.show()
        
    elif(method.get() == 'GRU'):
        model,history = script.create_model_gru(inpt[0][0],inpt[0][1],activation.get(),optimizer.get(),loss.get(),int(e4.get()),int(l/3))
        for i in range(len(inputs_x)):
            model.history = script.train_model_gru(model,inpt[i][0],inpt[i][1])
        # prediction_gru = model.predict(inpt[2])
        # print(prediction_gru)
        # pred = np.transpose(prediction_gru)
        # fig5 = plt.figure()
        # plt.plot(pred)
        # fig5.show()
    else:
        print("Something is wrong")
#    except:
#        print("asda")
def predict():
    global prediction_cnn
    global prediction_lstm
    global prediction_gru
    global pred
    global inpt
    
    if (var2.get() == 'Signal X'):
        inpt = inputs_x
    elif (var2.get() == 'Signal Y'):
        inpt = inputs_y
    if (method.get() =='CNN'):
        prediction_cnn = model.predict(inpt[0][2])
        print(prediction_cnn)
        pred = np.transpose(prediction_cnn)
        fig3 = plt.figure()
        plt.plot(pred)
        fig3.show()
        
        
    elif(method.get() == 'LSTM'):
        prediction_lstm = model.predict(inpt[0][2])
        print(prediction_lstm)
        pred = np.transpose(prediction_lstm)
        fig4 = plt.figure()
        plt.plot(pred)
        fig4.show()
        
    elif(method.get() == 'GRU'):
        prediction_gru = model.predict(inpt[0][2])
        print(prediction_gru)
        pred = np.transpose(prediction_gru)
        fig5 = plt.figure()
        plt.plot(pred)
        fig5.show()
    else:
        print("Something is wrong")

def compare():
    if var3.get():
        try:
            print(script.compare(var3.get(),np.transpose(inpt[0][3]),pred))
            x = script.compare(var3.get(),np.transpose(inpt[0][3]),pred)
            tk.Label(root,text='Accuracy assessment').grid(row=7, column = 3)
            tk.Label(root,text=x).grid(row=7, column = 4)
            
        except:
            print("Something is wrong")
        #Tworzenie pliku raportu analizy
        

    
    
B = tk.Button(root,text='Open Signal',command = opens)
B.grid(row=0,column = 0)

tk.Label(root,text="Sample length").grid(row=1, column = 0)
e1 = tk.Entry(root)
e1.grid(row=1,column = 1)

tk.Label(root,text="Time T0").grid(row=2, column = 0)
e2 = tk.Entry(root)
e2.grid(row=2,column = 1)

tk.Label(root,text="Number of GRU units").grid(row=3, column = 0)
e4 = tk.Entry(root)
e4.grid(row=3,column = 1)


tk.Label(root,text="Number of LSTM units").grid(row=3, column = 2)
e3 = tk.Entry(root)
e3.grid(row=3,column = 3)

tk.Label(root,text="CNN kernel size").grid(row=3, column = 4)
e4 = tk.Entry(root)
e4.grid(row=3,column = 5)





B1 = tk.Button(root,text="show sample",command = sample).grid(row =2, column= 2)





tk.Label(root,text="Choose optimizer").grid(row = 4, column = 0)
optimizers = ['adam','adagrad','rmsprop']
optimizer = 'adam'
optimizer = tk.StringVar()

drop = tk.OptionMenu(root,optimizer,*optimizers)
drop.grid(row = 4,column = 1)

tk.Label(root,text="Choose loss function").grid(row = 4, column = 2)
losses = ['mse','mean_absolute_error','squared_hinge']
loss = 'mse'
loss = tk.StringVar()

drop = tk.OptionMenu(root,loss,*losses)
drop.grid(row = 4,column = 3)

tk.Label(root,text="Choose activation").grid(row = 4, column = 4)
activation = 'relu'
activations = ['relu','elu','tanh']

activation = tk.StringVar()
drop = tk.OptionMenu(root,activation,*activations)
drop.grid(row = 4,column = 5)

tk.Label(root,text="Choose method").grid(row = 5, column = 0)
methods = ['CNN','LSTM','GRU']
method = 'CNN'
method = tk.StringVar()

drop = tk.OptionMenu(root,method,*methods)
drop.grid(row = 5,column = 1)

tk.Label(root,text="Choose signal").grid(row = 5, column = 2)
signals = ['Signal X','Signal Y']
var2 = 'Signal X'
var2 = tk.StringVar()

drop2 = tk.OptionMenu(root,var2,*signals)
drop2.grid(row = 5,column = 3)

B2 = tk.Button(root,text='Analyse',command = analyse)
B2.grid(row=5,column = 4)


tk.Label(root,text="Choose evaluation method").grid(row = 6, column = 0)
var3 = 'MSE'
methods2 = ['MSE','R2','MAE']
var3 = tk.StringVar()
drop3 = tk.OptionMenu(root,var3,*methods2)
drop3.grid(row = 6,column = 2)

B3 = tk.Button(root,text='Compare',command = compare)
B3.grid(row=6,column = 1)
B4 = tk.Button(root,text='Predict',command = predict)
B4.grid(row=6,column = 4)



root.mainloop()
#




