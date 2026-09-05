"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
An example of boundary value problem
of Laplace equation using a boundary element method
2020/12/21 Akihiro Nakatani (Osaka University)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import math

x0 = 0.0 # left bound
x1 = 1.0 # right bound
lx = x1 - x0 # horizontal length
nx = 40 # the number of horizontal segments
dx = lx / nx # spatial interval
dxdx = dx**2

y0 = 0.0 # lower bound
y1 = 1.0 # lower bound
ly = y1 - y0 # vertical length
ny = 40 # the number of vertical segments
dy = ly / ny # spatial interval
dydy = dy**2

h = 1.0

npe = 2
ndim = 2

npoin = 2 * (nx + ny)
nelem = 2 * (nx + ny)

u = np.zeros(nelem)
un = np.zeros(nelem)

xe = np.zeros((nelem, ndim))
xp = np.zeros((npoin, ndim))
icon = np.zeros((nelem, npe), dtype=np.int64)

i_bd = np.zeros(nelem, dtype=np.int64)
v_bd = np.zeros(nelem)
     
ng = 4
xig = np.array([-0.86113631, -0.33998104, 0.33998104, 0.86113631])
wg = np.array([0.34785485, 0.65214515, 0.65214515, 0.34785485])

re = np.zeros(nelem)
unx = np.zeros((nelem, ndim))

def setbound1():
     for ie in range(0, nelem):
          i_bd[ie] = 1
          v_bd[ie] = 0.0
     
     iy = 0
     y = y0
     for ix in range(0, nx):
          x = x0 + (ix + 0.5) * dx
          ie = ix
          i_bd[ie] = 0
          v_bd[ie] = h * math.sin(math.pi * x / lx)

     ix = nx
     x = x1
     for iy in range(0, ny):
          y = y0 + (iy + 0.5) * dy
          ie = nx + iy
          i_bd[ie] = 0
          v_bd[ie] = - h * math.sin(math.pi * y / ly)

     iy = ny
     y = y1
     for ix in range(0, nx):
          x = x1 - (ix + 0.5) * dx
          ie = (nx + ny) + ix
          i_bd[ie] = 0
          v_bd[ie] = h * math.sin(math.pi * x / lx)

     ix = 0
     x = x0
     for iy in range(0, ny):
          y = y1 - (iy + 0.5) * dy
          ie = 2 * nx + ny + iy
          i_bd[ie] = 0
          v_bd[ie] = - h * math.sin(math.pi * y / ly)

     for ie in range(0, nelem):
          print (ie, i_bd[ie], v_bd[ie])

def setedgenode():
     ip = 0

     for ix in range(0, nx):
          xp[ip][0] = x0 + ix * dx
          xp[ip][1] = y0
          ip += 1

     for iy in range(0, ny):
          xp[ip][0] = x1
          xp[ip][1] = y0 + iy * dy
          ip += 1
          
     for ix in range(0, nx):
          xp[ip][0] = x1 - ix * dx
          xp[ip][1] = y1
          ip += 1

     for iy in range(0, ny):
          xp[ip][0] = x0
          xp[ip][1] = y1 - iy * dy
          ip += 1
          
def setelem():
     for ie in range(0, nelem):
          ip = ie
          icon[ie][0] = ip
          icon[ie][1] = (ip + 1) % npoin
     for ie in range(0, nelem):
          ip0 = icon[ie][0]
          ip1 = icon[ie][1]
          for idim in range(0, ndim):
               xe[ie][idim] = (xp[ip0][idim] + xp[ip1][idim]) * 0.5
          sx0 = xp[ip1][0] - xp[ip0][0]
          sx1 = xp[ip1][1] - xp[ip0][1]
          re[ie] = math.sqrt(sx0**2 + sx1**2)
          unx[ie][0] = sx1 / re[ie]
          unx[ie][1] = - sx0 / re[ie]

def setmatrixcomponents(ie, x0, x1):
     ip0 = icon[ie][0]
     ip1 = icon[ie][1]
     sx0 = xp[ip1][0] - xp[ip0][0]
     sx1 = xp[ip1][1] - xp[ip0][1]

     a = 0.0
     b = 0.0
     for ig in range(0, ng):
          xig0 = xe[ie][0] + sx0 / 2.0 * xig[ig]
          xig1 = xe[ie][1] + sx1 / 2.0 * xig[ig]
          rm = math.sqrt((xig0 - x0)**2 + (xig1 - x1)**2)
          a += - (unx[ie][0] * (xig0 - x0) + unx[ie][1] * (xig1 - x1)) \
               / rm**2 * wg[ig] * (re[ie] / 2.0)
          b += - math.log(rm) * wg[ig] * (re[ie] / 2.0)
     return a, b

def calcus(x0, x1):
     us_in = 0.0
     for ie in range(0, nelem):
          a, b = setmatrixcomponents(ie, x0, x1)
          us_in += un[ie] * b - u[ie] * a
     us_in /= (2.0 * math.pi)
     return us_in

# main routine

setbound1()

setedgenode()

setelem()

amat = np.zeros((nelem, nelem))
bmat = np.zeros((nelem, nelem))
rhs = np.zeros(nelem)

for ie1 in range(0, nelem):
     for ie2 in range(0, nelem):
          if ie1 == ie2:
               ie = ie2
               amat[ie][ie] = math.pi
               bmat[ie][ie] = re[ie] * (1.0 - math.log(re[ie] / 2.0))
          else:
               amat[ie1][ie2], bmat[ie1][ie2] = \
                    setmatrixcomponents(ie2, xe[ie1][0], xe[ie1][1])

#for ie1 in range(0, nelem):
#     wa2 = 0
#     for ie2 in range(0, nelem):
#          if ie1 != ie2:
#               wa2 += amat[ie1][ie2]
#     amat[ie1][ie1] = - wa2
               
for ie2 in range(0, nelem):
     if i_bd[ie2] == 0:
          for ie1 in range(0, nelem):
               work = bmat[ie1][ie2]
               bmat[ie1][ie2] = - amat[ie1][ie2]
               amat[ie1][ie2] = - work

for ie1 in range(0, nelem):
     rhs[ie1] = 0
     for ie2 in range(0, nelem):
          rhs[ie1] += bmat[ie1][ie2] * v_bd[ie2]

u = np.linalg.solve(amat, rhs)  

for ie in range(0, nelem):
     if i_bd[ie] == 0:
          un[ie] = u[ie]
          u[ie] = v_bd[ie]
     else:
          un[ie] = v_bd[ie]

fall = open("fbemall.txt",'w')
#for iy in range(0, ny + 1):
for iy in range(1, ny):
     y = y0 + iy * dy
#     for ix in range(0, nx + 1):
     for ix in range(1, nx):
          x = x0 + ix * dx
          us_in = calcus(x, y)
          print (x, y, us_in, file=fall)
     print (file=fall)
fall.close()
