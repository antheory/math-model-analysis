"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of boundary value problem
of Laplace equation using a finite element method
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

penalty = 1e8

npe = 3
ndim = 2

npoin = (nx + 1) * (ny + 1)
nelem = 2 * nx * ny

u = np.zeros(npoin)

xp = np.zeros((npoin, ndim))
icon = np.zeros((nelem, npe), dtype=np.int64)

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

def setnode():
     ip = 0
     for iy in range(0, ny + 1):
          for ix in range(0, nx + 1):
               xp[ip][0] = x0 + ix * dx
               xp[ip][1] = y0 + iy * dy
               ip += 1

def setelem():
     for iy in range(0, ny):
          for ix in range(0, nx):
               ielem = (2 * nx) * iy + 2 * ix
               icon[ielem][0] = (nx + 1) * iy + ix
               icon[ielem][1] = (nx + 1) * iy + ix + 1
               icon[ielem][2] = (nx + 1) * (iy + 1) + ix + 1
               ielem = (2 * nx) * iy + 2 * ix + 1
               icon[ielem][0] = (nx + 1) * iy + ix
               icon[ielem][1] = (nx + 1) * (iy + 1) + ix + 1
               icon[ielem][2] = (nx + 1) * (iy + 1) + ix

def dshape(xpe):
     fd = np.zeros((ndim, npe))
     x = 0
     y = 1
     ie1 = 0
     ie2 = 1
     ie3 = 2
     area2 = xpe[ie2][x] * xpe[ie3][y] - xpe[ie2][y] * xpe[ie3][x] + \
             xpe[ie3][x] * xpe[ie1][y] - xpe[ie3][y] * xpe[ie1][x] + \
             xpe[ie1][x] * xpe[ie2][y] - xpe[ie1][y] * xpe[ie2][x]
     fd[x][ie1] = (xpe[ie2][y] - xpe[ie3][y]) / area2
     fd[y][ie1] = (xpe[ie3][x] - xpe[ie2][x]) / area2
     fd[x][ie2] = (xpe[ie3][y] - xpe[ie1][y]) / area2
     fd[y][ie2] = (xpe[ie1][x] - xpe[ie3][x]) / area2
     fd[x][ie3] = (xpe[ie1][y] - xpe[ie2][y]) / area2
     fd[y][ie3] = (xpe[ie2][x] - xpe[ie1][x]) / area2
     area = area2 / 2.0
     return area, fd

# main routine

nbd = setbound1()

ip_bd = np.zeros(nbd, dtype=np.int64)

setbound2()

setnode()

setelem()

xpe = np.zeros((npe, ndim))
fd = np.zeros((ndim, npe))
Aelm = np.zeros((npe, npe))

A = np.zeros((npoin, npoin))
b = np.zeros(npoin)

for ielem in range(0, nelem):
     for ipe in range(0, npe):
          for idim in range(0, ndim):
               xpe[ipe][idim] = xp[icon[ielem][ipe]][idim]
     area, fd = dshape(xpe)
     Aelm = area * np.dot(np.transpose(fd), fd)
     for ipe2 in range(0, npe):
          for ipe1 in range(0, npe):
               ip1 = icon[ielem][ipe1]
               ip2 = icon[ielem][ipe2]
               A[ip1][ip2] += Aelm[ipe1][ipe2]

for ibd in range(0, nbd):
     ip = ip_bd[ibd]
     A[ip][ip] += penalty
     b[ip] = A[ip][ip] * v_bd[ip]

u = np.linalg.solve(A, b)  

fall = open("ffemall.txt",'w')
for iy in range(0, ny + 1):
     y = y0 + iy * dy
     for ix in range(0, nx + 1):
          x = x0 + ix * dx
          ip = (nx + 1) * iy + ix
          print (x, y, u[ip], file=fall)
     print (file=fall)
fall.close()
