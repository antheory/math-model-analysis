"""An example of initial value problem
of pendulum model
2025/09/04 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- 未使用 import を整理
- open / close を with 文に変更
- 入力値の数値変換エラーを最低限処理
- 行内コメントを追加
- 計算の流れは元コードのまま維持
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import tkinter as tk
from tkinter import messagebox

g_over_l = 1.0
theta_0 = 0.0


# ----------------------------
# GUI for parameter input
# ----------------------------
root = tk.Tk()
root.title("Parameters")
root.geometry("360x120")

input_valone_label = tk.Label(text="c/m")
input_valone_label.grid(row=1, column=1, padx=10)

input_valone = tk.Entry(width=40)
input_valone.grid(row=1, column=2)

input_valtwo_label = tk.Label(text="w0")
input_valtwo_label.grid(row=2, column=1, padx=10)

input_valtwo = tk.Entry(width=40)
input_valtwo.grid(row=2, column=2)


def funcdxdt(t, x, g_over_l, c_over_m):
    """Return the first-order ODE system for the damped pendulum."""
    dxdt = np.zeros(2)
    dxdt[0] = x[1]  # d(theta)/dt = omega
    dxdt[1] = -g_over_l * math.sin(x[0]) - c_over_m * x[1]  # d(omega)/dt
    return dxdt


def mymain(c_over_m, omega_0):
    """Solve the ODE and save the time history to a text file."""
    t0 = 0.0
    t1 = 100.0
    n = 1000
    t = np.linspace(t0, t1, num=n + 1)

    # initial condition: (theta(0), omega(0))
    x0 = (theta_0, omega_0)

    sol = solve_ivp(
        funcdxdt,
        (t0, t1),
        x0,
        method="RK45",
        t_eval=t,
        args=(g_over_l, c_over_m),
    )

    # combine t, theta, omega into one array for output
    allsol = np.vstack((sol.t, sol.y)).T

    with open("output.txt", "w", encoding="utf-8") as fall:
        for i in range(len(allsol)):
            print(allsol[i][0], allsol[i][1], allsol[i][2], file=fall)


def myplotgraph(c_over_m, omega_0):
    """Read the output file and plot theta versus time."""
    gx = []
    gy = []

    with open("output.txt", "rt", encoding="utf-8") as f:
        for line in f:
            data = line.split()  # 空白数に依存しにくい分割
            gx.append(float(data[0]))
            gy.append(float(data[1]))

    plt.figure()
    plt.plot(gx, gy)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\theta$")

    gtitle = "$c/m$:" + str(c_over_m) + ", "
    gtitle += r"$\omega_0$:" + str(omega_0)
    plt.title(gtitle, loc="center")

    plt.grid(True)
    plt.show()


def button_click():
    """Read parameters from GUI, solve the ODE, and plot the result."""
    try:
        c_over_m = float(input_valone.get())
        omega_0 = float(input_valtwo.get())
    except ValueError:
        messagebox.showerror("Input error", "Please enter valid numbers.")
        return

    mymain(c_over_m, omega_0)
    myplotgraph(c_over_m, omega_0)


def button_quit():
    """Close the GUI and all matplotlib windows."""
    root.destroy()
    plt.close("all")


button1 = tk.Button(root, text="ENTER", command=button_click)
button1.place(x=10, y=80)

button2 = tk.Button(root, text="Quit", command=button_quit)
button2.place(x=100, y=80)

root.mainloop()
