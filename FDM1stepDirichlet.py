"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of initial boundary value problem
of a thermal conduction equation
using finite difference method
2020/12/13 Akihiro Nakatani (Osaka University)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import math

x0 = 0.0 # left bound
x1 = 1.0 # right bound
nx = 50 # the number of points
dx = (x1 - x0) / nx # space interval

l = x1 - x0

t0 = 0.0 # start time

#ckappa = 2.0e-2 # velocity of wave 2.0e-2*50=1
#t1 = 2.0 # stop time
#nt = 200 # time steps

ckappa = 1.0 # velocity of wave
#t1 = 0.04 # stop time 2/50=0.04
#nt = 200 # time steps
#t1 = 0.05 # stop time 2/50=0.04
#nt = 250 # time steps
#t1 = 0.1 # stop time 2/50=0.04
#nt = 500 # time steps
t1 = 1.0 # stop time 2/50=0.04
nt = 10000 # time steps
dt = (t1 - t0) / nt # time increment

coef = ckappa * dt / dx**2  # square of Courant number

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

#def u_init(x):
#    return 1.0*math.sin(2.0*math.pi * x / (2.0*l)) + 0.2*math.sin(2.0*math.pi * 9.0 * x / (2.0*l))
#    return 1.0*math.sin(math.pi * x / l) + 0.2*math.sin(math.pi * 10 * x / l)

def intsimpson(u, nx, dx):
    sum1 = 0
    for kx in range(1, int(nx/2)):
        sum1 += u[2 * kx]
    sum2 = 0
    for kx in range(1, int(nx/2) + 1):
        sum2 += u[2 * kx - 1]
    return (dx / 3.0) * (u[0] + u[nx] + 2.0 * sum1 + 4.0 * sum2)

# setting of initial value

t = 0.0

u = np.zeros(nx + 1)
for ix in range(0, nx + 1): # 0<= ix <= nx
    u[ix] = u_init(x0 + ix * dx)

du = np.zeros(nx + 1)

# output initial value

fall = open("fdm1sD.txt",'w')
fenergy = open("fdm1sD_energy.txt",'w')
fevery = open("fdm1sD_every.txt",'w')

it = 0
fone = open("fdm1sD_{0:03d}.txt".format(it),'w')
for ix in range(0, nx + 1): # 0<= ix <= nx
    x = x0 + ix * dx
    print (x, t, u[ix], file=fall)
    print(x, t, u[ix], file=fone)
    print(x, t, u[ix], file=fevery)
fone.close()
print(t, (np.sum(u) - u[0]/2 - u[nx]/2) * dx, file=fenergy)
#print(t, intsimpson(u, nx, dx), file=fenergy)
print(file=fevery)
print (file=fall)

# temporal evolution

for it in range(1, nt + 1): # 1 <= ix <= nt - 1
    for ix in range(1, nx): # 1 <= ix <= nx - 1
        du[ix] = coef * (u[ix + 1] + u[ix - 1] - 2.0 * u[ix])
    for ix in range(1, nx): # 1 <= ix <= nx - 1
        u[ix] += du[ix]
    t += dt
    #print(t, (np.sum(u) - u[0]/2 - u[nx]/2) * dx, file=fenergy)
    print(t, intsimpson(u, nx, dx), file=fenergy)
    if it == 1 or it == 10 or it == 100 or it == 1000 or it == 10000:
        fone = open("fdm1sD_{0:03d}.txt".format(it),'w')
        for ix in range(0, nx + 1): # 0 <= ix <= nx
            x = x0 + ix * dx
            print(x, t, u[ix], file=fone)
            print(x, t, u[ix], file=fall)
            print(x, t, u[ix], file=fevery)
        fone.close()
        print(file=fall)
        print(file=fevery)

fevery.close()
fenergy.close()
fall.close()
