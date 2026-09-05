"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of boundary value problem
of Laplace equation using a finite difference method
2020/12/21 Akihiro Nakatani (Osaka University)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import math

x0 = 0.0 # left bound
x1 = 1.0 # right bound
lx = x1 - x0 # horizontal length
nx = 20 # the number of horizontal segments
dx = lx / nx # spatial interval
dxdx = dx**2

y0 = 0.0 # lower bound
y1 = 1.0 # lower bound
ly = y1 - y0 # vertical length
ny = 20 # the number of vertical segments
dy = ly / ny # spatial interval
dydy = dy**2

h = 1.0

npoin = (nx + 1) * (ny + 1)

u = np.zeros(npoin)

i_bd = np.zeros(npoin, dtype=np.int64)
v_bd = np.zeros(npoin)
     
def setbound1():
     iy = 0
     y = y0
     for ix in range(0, nx + 1):
          x = x0 + ix * dx
          ip = (nx + 1) * iy + ix
          i_bd[ip] = 1
          v_bd[ip] = h * math.sin(math.pi * x / lx)

     iy = ny
     y = y1
     for ix in range(0, nx + 1):
          x = x0 + ix * dx
          ip = (nx + 1) * iy + ix
          i_bd[ip] = 1
          v_bd[ip] = h * math.sin(math.pi * x / lx)

     ix = 0
     x = x0
     for iy in range(0, ny + 1):
          y = y0 + iy * dy
          ip = (nx + 1) * iy + ix
          i_bd[ip] = 1
          v_bd[ip] = - h * math.sin(math.pi * y / ly)

     ix = nx
     x = x1
     for iy in range(0, ny + 1):
          y = y0 + iy * dy
          ip = (nx + 1) * iy + ix
          i_bd[ip] = 1
          v_bd[ip] = - h * math.sin(math.pi * y / ly)

     nbd = 0
     for ip in range(0, npoin):
          if i_bd[ip] != 0:
               nbd += 1
     return(nbd)

def setbound2():
     ibd = 0
     for ip in range(0, npoin):
          if i_bd[ip] != 0:
               i_bd[ip] = ibd
               ip_bd[ibd] = ip
               ibd += 1

# main routine

nbd = setbound1()

ip_bd = np.zeros(nbd, dtype=np.int64)

setbound2()

A = np.zeros((npoin, npoin))
b = np.zeros(npoin)

for iy in range(0, ny + 1):
     for ix in range(0, nx + 1):
          ip = (nx + 1) * iy + ix
          if i_bd[ip] == 0:
               A[ip][ip] = - (2.0 / dxdx + 2.0 / dydy)
               ip_x_p = (nx + 1) * iy + ix + 1
               if i_bd[ip_x_p] == 0:
                    A[ip_x_p][ip] = 1.0 / dxdx
               else:
                    A[ip_x_p][ip] = (1.0 / dxdx) * 0.0
                    b[ip] += - 1.0 / dxdx * v_bd[ip_x_p]
               ip_x_m = (nx + 1) * iy + ix - 1
               if i_bd[ip_x_m] == 0:
                    A[ip_x_m][ip] = 1.0 / dxdx
               else:
                    A[ip_x_m][ip] = (1.0 / dxdx) * 0.0
                    b[ip] += - 1.0 / dxdx * v_bd[ip_x_m]
               ip_y_p = (nx + 1) * (iy + 1) + ix 
               if i_bd[ip_y_p] == 0:
                    A[ip_y_p][ip] = 1.0 / dydy
               else:
                    A[ip_y_p][ip] = (1.0 / dydy) * 0.0
                    b[ip] += - 1.0 / dydy * v_bd[ip_y_p]
               ip_y_m = (nx + 1) * (iy - 1) + ix
               if i_bd[ip_y_m] == 0:
                    A[ip_y_m][ip] = 1.0 / dydy
               else:
                    A[ip_y_m][ip] = (1.0 / dydy) * 0.0
                    b[ip] += - 1.0 / dydy * v_bd[ip_y_m]

for ibd in range(0, nbd):
     ip = ip_bd[ibd]
     A[ip][ip] = - (2.0 / dxdx + 2.0 / dydy)
     b[ip] = A[ip][ip] * v_bd[ip]

u = np.linalg.solve(A, b)  

fall = open("ffdmall.txt",'w')
for iy in range(0, ny + 1):
     y = y0 + iy * dy
     for ix in range(0, nx + 1):
          x = x0 + ix * dx
          ip = (nx + 1) * iy + ix
          print (x, y, u[ip], file=fall)
     print (file=fall)
fall.close()
