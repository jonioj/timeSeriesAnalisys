import tkinter as tk
from tkinter import *
from tkinter import filedialog
#import LSTM_RNN2
import pandas as pd

root = tk.Tk()
root.geometry("400x400")
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
        LSTM_RNN2.show_signal(input_feature)
    except:
        print("Błąd pliku")
def analyse():
    print("value is" + var1.get())
B = tk.Button(root,text='Open Signal',command = read_entry)
B.grid(row=0,column = 0)

tk.Label(root,text="Sample length").grid(row=1, column = 0)
e1 = tk.Entry(root)
e1.grid(row=1,column = 1)

tk.Label(root,text="Time T0").grid(row=2, column = 0)
e2 = tk.Entry(root)
e2.grid(row=2,column = 1)

tk.Label(root,text="Choose method").grid(row = 3, column = 0)
lst1 = ['CNN','LSTM','GRU']
var1 = tk.StringVar()
drop = tk.OptionMenu(root,var1,*lst1)
drop.grid(row = 3,column = 1)

B2 = tk.Button(root,text='Analyse',command = analyse)
B2.grid(row=3,column = 4)

root.mainloop()





