"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of initial value problem
of pendulum model
2025/09/04 Akihiro Nakatani (Osaka University)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import messagebox

g_over_l = 1.0; theta_0 = 0.0

root = tk.Tk()
root.title("Parameters")
root.geometry("360x120")
#
input_valone_label = tk.Label(text="c/m")
input_valone_label.grid(row=1, column=1, padx=10,)
input_valone = tk.Entry(width=40)
input_valone.grid(row=1, column=2)
#
input_valtwo_label = tk.Label(text="w0")
input_valtwo_label.grid(row=2, column=1, padx=10,)
input_valtwo = tk.Entry(width=40)
input_valtwo.grid(row=2, column=2)

def funcdxdt(t, x, g_over_l, c_over_m):
    dxdt = np.zeros(2)
    dxdt[0] = x[1]
    dxdt[1] = - g_over_l * math.sin(x[0]) - c_over_m * x[1]
    return dxdt

def mymain(c_over_m, omega_0):
    t0 = 0.0; t1 = 100.0; n = 1000
    t = np.linspace(t0, t1, num=n+1)
    x0 = theta_0, omega_0
#
    sol = solve_ivp(funcdxdt, (t0, t1), x0, method='RK45', \
         t_eval=t, args=(g_over_l, c_over_m))
    allsol = np.vstack((sol.t, sol.y)).T
#
    fall = open("output.txt", 'w')
    for i in range(len(allsol)):
        print (allsol[i][0], allsol[i][1], allsol[i][2], file=fall)
    fall.close()

def myplotgraph(c_over_m, omega_0):    
    gx=[]; gy=[]
    f = open("output.txt", 'rt')
    for line in f:
        data = line[:-1].split(' ')
        gx.append(float(data[0]))
        gy.append(float(data[1]))
    fig=plt.clf()
    plt.plot(gx, gy)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\theta$')
    gtitle = "$c/m$:" + str(c_over_m) + ','
    gtitle += "$\omega_0$:" + str(omega_0)
    plt.title(gtitle, loc='center')
    plt.grid(True)
    plt.show()

def button_click():
   c_over_m = float(input_valone.get())
   omega_0 = float(input_valtwo.get())
   mymain(c_over_m, omega_0)
   myplotgraph(c_over_m, omega_0)
def button_quit():
    root.destroy()
    plt.close("all")
button1 = tk.Button(root, text="ENTER", command=button_click)
button1.place(x=10, y=80)
button2 = tk.Button(root, text="Quit", command=button_quit)
button2.place(x=100, y=80)
root.mainloop()
