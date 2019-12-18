import tkinter as tk
from tkinter import *
from tkinter import filedialog
import script
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = tk.Tk()
root.geometry("600x600")
def read_entry():
    print(e1.get())
    
def open():
    global path
    global signal
    global input_feature
    try:
        root.filename = filedialog.askopenfilename(initialdir  = "/",title="asds")
        path = root.filename
        signal = pd.read_csv(path)
        input_feature= signal.iloc[:,[2,5]].values
        script.show_signal(input_feature)
        
    except:
        print("Wrong file")
        

def sample():
    try:
        t = int(e2.get())
        l = int(e1.get())
        global s
        global input_x
        global input_y
        s = script.pick_sample(t,l,input_feature)
        script.show_signal(s)
        input_x, input_y = script.prep_sample(s,250,l)
        
        print("ready to analyze")
    except:
        print("Something went wrong")
        
def analyse():
    global model
    global history
    global prediction_cnn
    global pred
    global inpt
    if (var2.get() == 'Signal X'):
        inpt = input_x
    elif (var2.get() == 'Signal Y'):
        inpt = input_y
    
#    try:
    if (method.get() =='CNN'):
        model, history = script.create_model_cnn(inpt[0],inpt[1])
        prediction_cnn = model.predict(inpt[2])
        print(prediction_cnn)
        pred = np.transpose(prediction_cnn)
        fig2 = plt.figure()
        plt.plot(pred)
        fig2.show()
    elif(method.get() == 'GRU'):
        model,history = script.create_model_cnn(inpt[0])
#        prediction_cnn = np.transpose(prediction_cnn)
#        LSTM_RNN2.show(prediction_cnn)
    else:
        print("blad")
#    except:
#        print("asda")


def compare():
    if (var3.get() == 'MSE'):
        
        print(script.compare(np.transpose(inpt[3]),pred))

    
    
B = tk.Button(root,text='Open Signal',command = open)
B.grid(row=0,column = 0)

tk.Label(root,text="Sample length").grid(row=1, column = 0)
e1 = tk.Entry(root)
e1.grid(row=1,column = 1)

tk.Label(root,text="Time T0").grid(row=2, column = 0)
e2 = tk.Entry(root)
e2.grid(row=2,column = 1)

tk.Label(root,text="Activation function").grid(row=3, column = 0)
e3 = tk.Entry(root)
e3.grid(row=3,column = 1)





B1 = tk.Button(root,text="show sample",command = sample).grid(row =2, column= 2)

tk.Label(root,text="Choose method").grid(row = 4, column = 0)
methods = ['CNN','LSTM','GRU']
method = tk.StringVar()
drop = tk.OptionMenu(root,method,*methods)
drop.grid(row = 4,column = 1)

tk.Label(root,text="Choose signal").grid(row = 4, column = 2)
signals = ['Signal X','Signal Y']
var2 = tk.StringVar()
drop2 = tk.OptionMenu(root,var2,*signals)
drop2.grid(row = 4,column = 3)

B2 = tk.Button(root,text='Analyse',command = analyse)
B2.grid(row=4,column = 4)


tk.Label(root,text="Choose evaluation method").grid(row = 5, column = 0)
methods2 = ['MSE','MSA']
var3 = tk.StringVar()
drop3 = tk.OptionMenu(root,var3,*methods2)
drop3.grid(row = 5,column = 0)

B3 = tk.Button(root,text='Compare',command = compare)
B3.grid(row=5,column = 1)
root.mainloop()





