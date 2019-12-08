# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:54:34 2019

@author: jonas
"""

import tkinter
from tkinter import *
root = Tk()
root.mainloop()
from tkinter.filedialog import askopenfilename
filename = askopenfilename()
print(filename)



def helloCallBack():
   tkMessageBox.showinfo( "Hello Python", "Hello World")

B = Tkinter.Button(root, text ="Hello", command = askopenfil)

B.pack()
top.mainloop()