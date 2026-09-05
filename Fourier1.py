"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of initial boundary value problem
of a thermal conduction equation
solution of Forier series expansion
2020/12/13 Akihiro Nakatani (Osaka University)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy import integrate

x0 = 0.0 # left bound
x1 = 1.0 # right bound
nx = 50 # the number of points
dx = (x1 - x0) / nx # space interval

l = x1 - x0

t0 = 0.0 # start time
t1 = 2.0 # stop time
nt = 200 # time steps
dt = (t1 - t0) / nt # time increment

nmode = 100

ckappa = 2.0e-2 # velocity of wave

# function of initial value

xa = 0.25
h = 1.0

def u_init(x):
    if x < xa:
        return 0
    elif x < (1-xa):
        return h
    else:
        return 0

# Fourier coefficient obtained using qadrature
def A_integrand(x, n):
    k = n * math.pi / l
    return u_init(x) * math.sin(k * x)

def A_fourier(n):
    val, _ = scipy.integrate.quad(A_integrand, x0, x1, args=(n))
    val *= 2 / l
    return val
        
# Solution of n-th mode

def u_phin(x, t, n):
    k = n * math.pi / l
    return An[n] * math.exp(- ckappa * k**2 * t) * math.sin(k * x)

An = np.zeros(nmode + 1)
for imode in range(1, nmode + 1): # 1 <= imode <= nmode
    An[imode] = A_fourier(imode)

t = 0.0
for imode in range(1, nmode + 1): # 1 <= imode <= nmode
    fmode = open("fmode1_{0:03d}.txt".format(imode),'w')
    for ix in range(0, nx + 1): # 0 <= ix <= nx
        x = x0 + ix * dx
        print(t, x, u_phin(x, t, imode), file=fmode)
    fmode.close()

u = np.zeros(nx +1)
fall = open("fourier1.txt",'w')
fenergy = open("fourier1_energy.txt",'w')
fevery = open("fourier1_every.txt",'w')

for it in range(0, nt + 1): # 0<= it <= nt
    t = t0 + it * dt

    for ix in range(0, nx + 1): # 0 <= ix <= nx
        x = x0 + ix * dx
        u[ix] = 0.0
        for imode in range(1, nmode + 1): # 1 <= imode <= nmode
            u[ix] += u_phin(x, t, imode)

    fone = open("fourier1_{0:03d}.txt".format(it),'w')
    print(t, (np.sum(u) - u[0]/2 - u[nx]/2) * dx, file=fenergy)
    for ix in range(0, nx + 1): # 0 <= ix <= nx
    	x = x0 + ix * dx
    	print(t, x, u[ix], file=fone)
    	print(t, x, u[ix], file=fall)
    	if it % 10 == 0:
            print(t, x, u[ix], file=fevery)
    fone.close()

    print(file=fall)
    if it % 10 == 0:
        print(file=fevery)

fevery.close()
fenergy.close()
fall.close()
