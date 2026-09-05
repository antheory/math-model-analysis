"""An example of boundary value problem
of Laplace equation using a finite element method
2020/12/21 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- 誤記コメントを修正
- 未使用 import / 未使用変数を整理
- 授業用の行内コメントを追加
- 出力ファイルを with 文で安全に書き出し
"""

import numpy as np
import math

# ----------------------------
# basic settings
# ----------------------------
x0 = 0.0  # left bound
x1 = 1.0  # right bound
lx = x1 - x0  # horizontal length
nx = 20  # the number of horizontal segments
dx = lx / nx  # spatial interval in x

y0 = 0.0  # lower bound
y1 = 1.0  # upper bound
ly = y1 - y0  # vertical length
ny = 20  # the number of vertical segments
dy = ly / ny  # spatial interval in y

h = 1.0  # boundary value amplitude

penalty = 1e8  # penalty parameter for Dirichlet boundary condition

npe = 3   # nodes per element (linear triangle)
ndim = 2  # spatial dimension

npoin = (nx + 1) * (ny + 1)  # total number of nodes
nelem = 2 * nx * ny          # total number of triangular elements

u = np.zeros(npoin)  # nodal solution

xp = np.zeros((npoin, ndim))                   # node coordinates
icon = np.zeros((nelem, npe), dtype=np.int64)  # element connectivity

i_bd = np.zeros(npoin, dtype=np.int64)  # boundary flag (0: interior, nonzero: boundary)
v_bd = np.zeros(npoin)                  # prescribed boundary value


def setbound1():
    """Set Dirichlet boundary conditions on all outer boundary nodes."""
    iy = 0
    for ix in range(0, nx + 1):
        x = x0 + ix * dx
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = h * math.sin(math.pi * x / lx)

    iy = ny
    for ix in range(0, nx + 1):
        x = x0 + ix * dx
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = h * math.sin(math.pi * x / lx)

    ix = 0
    for iy in range(0, ny + 1):
        y = y0 + iy * dy
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = -h * math.sin(math.pi * y / ly)

    ix = nx
    for iy in range(0, ny + 1):
        y = y0 + iy * dy
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = -h * math.sin(math.pi * y / ly)

    # count boundary nodes
    nbd = 0
    for ip in range(0, npoin):
        if i_bd[ip] != 0:
            nbd += 1
    return nbd


def setbound2():
    """Store boundary node indices into ip_bd."""
    ibd = 0
    for ip in range(0, npoin):
        if i_bd[ip] != 0:
            i_bd[ip] = ibd
            ip_bd[ibd] = ip
            ibd += 1


def setnode():
    """Generate structured grid node coordinates."""
    ip = 0
    for iy in range(0, ny + 1):
        for ix in range(0, nx + 1):
            xp[ip][0] = x0 + ix * dx
            xp[ip][1] = y0 + iy * dy
            ip += 1


def setelem():
    """Divide each rectangle cell into two triangular elements."""
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
    """Return triangle area and derivatives of linear shape functions."""
    fd = np.zeros((ndim, npe))

    x = 0
    y = 1
    ie1 = 0
    ie2 = 1
    ie3 = 2

    # 2 * area of triangle
    area2 = (
        xpe[ie2][x] * xpe[ie3][y] - xpe[ie2][y] * xpe[ie3][x]
        + xpe[ie3][x] * xpe[ie1][y] - xpe[ie3][y] * xpe[ie1][x]
        + xpe[ie1][x] * xpe[ie2][y] - xpe[ie1][y] * xpe[ie2][x]
    )

    # derivatives of linear shape functions
    fd[x][ie1] = (xpe[ie2][y] - xpe[ie3][y]) / area2
    fd[y][ie1] = (xpe[ie3][x] - xpe[ie2][x]) / area2
    fd[x][ie2] = (xpe[ie3][y] - xpe[ie1][y]) / area2
    fd[y][ie2] = (xpe[ie1][x] - xpe[ie3][x]) / area2
    fd[x][ie3] = (xpe[ie1][y] - xpe[ie2][y]) / area2
    fd[y][ie3] = (xpe[ie2][x] - xpe[ie1][x]) / area2

    area = area2 / 2.0
    return area, fd


# ----------------------------
# main routine
# ----------------------------
nbd = setbound1()

ip_bd = np.zeros(nbd, dtype=np.int64)  # list of boundary node ids

setbound2()
setnode()
setelem()

xpe = np.zeros((npe, ndim))   # element node coordinates
fd = np.zeros((ndim, npe))    # derivatives of shape functions
Aelm = np.zeros((npe, npe))   # element stiffness matrix

A = np.zeros((npoin, npoin))  # global stiffness matrix
b = np.zeros(npoin)           # global right-hand side

# assemble global stiffness matrix
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

# impose Dirichlet boundary conditions by penalty method
for ibd in range(0, nbd):
    ip = ip_bd[ibd]
    A[ip][ip] += penalty
    b[ip] = A[ip][ip] * v_bd[ip]

# solve linear system
u = np.linalg.solve(A, b)

# save nodal solution to file
with open("ffemall.txt", "w", encoding="utf-8") as fall:
    for iy in range(0, ny + 1):
        y = y0 + iy * dy
        for ix in range(0, nx + 1):
            x = x0 + ix * dx
            ip = (nx + 1) * iy + ix
            print(x, y, u[ip], file=fall)
        print(file=fall)
