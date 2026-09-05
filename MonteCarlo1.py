"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of boundary value problem
of Laplace equation using Monte Carlo method
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

y0 = 0.0 # lower bound
y1 = 1.0 # lower bound
ly = y1 - y0 # vertical length
ny = 20 # the number of vertical segments
dy = ly / ny # spatial interval

h = 1.0

npoin = (nx + 1) * (ny + 1)
nwalk = 1000

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

def randomwalk(ix_start, iy_start):
     ix_walk = ix_start
     iy_walk = iy_start
     ip_walk = (nx + 1) * iy_walk + ix_walk
     while i_bd[ip_walk] == 0:
          id = np.random.randint(0, 4)
          if id == 0:
               ix_walk -= 1
          elif id == 1:
               ix_walk += 1
          elif id == 2:
               iy_walk -= 1
          else:
               iy_walk += 1
          ip_walk = (nx + 1) * iy_walk + ix_walk
     ibd = i_bd[ip_walk]
     return (ibd)

# main routine

nbd = setbound1()

ip_bd = np.zeros(nbd, dtype=np.int64)
ipr_bd = np.zeros(nbd, dtype=np.int64)

setbound2()

fall = open("fMonteCarloall.txt",'w')
for iy in range(0, ny + 1):
     y = y0 + iy * dy
     for ix in range(0, nx + 1):
          x = x0 + ix * dx
          ip = (nx + 1) * iy + ix
          for ibd in range(0, nbd):
               ipr_bd[ibd] = 0
          for iwalk in range(0, nwalk):
               ibd = randomwalk(ix,iy)
               ipr_bd[ibd] += 1
                    
          u[ip] = 0.0
          for ibd in range(0, nbd):
               u[ip] += float(ipr_bd[ibd]) / float(nwalk) * v_bd[ip_bd[ibd]]
          print (x, y, u[ip], file=fall)
     print (file=fall)
fall.close()

