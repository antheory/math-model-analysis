"""An example of boundary value problem
of Laplace equation using a finite difference method
2020/12/21 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- 誤記コメントを修正
- 未使用 import を整理
- 授業用の行内コメントを追加
- 出力ファイルを with 文で安全に書き出し
- 計算の流れは元コードのまま維持
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
dxdx = dx**2  # dx squared

y0 = 0.0  # lower bound
y1 = 1.0  # upper bound
ly = y1 - y0  # vertical length
ny = 20  # the number of vertical segments
dy = ly / ny  # spatial interval in y
dydy = dy**2  # dy squared

h = 1.0  # boundary value amplitude

npoin = (nx + 1) * (ny + 1)  # total number of grid points

u = np.zeros(npoin)  # solution at grid points

i_bd = np.zeros(npoin, dtype=np.int64)  # boundary flag (0: interior, nonzero: boundary)
v_bd = np.zeros(npoin)  # prescribed boundary value


def setbound1():
    """Set Dirichlet boundary conditions on the outer boundary."""
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

    # count boundary points
    nbd = 0
    for ip in range(0, npoin):
        if i_bd[ip] != 0:
            nbd += 1
    return nbd


def setbound2():
    """Store boundary point indices into ip_bd."""
    ibd = 0
    for ip in range(0, npoin):
        if i_bd[ip] != 0:
            i_bd[ip] = ibd
            ip_bd[ibd] = ip
            ibd += 1


# ----------------------------
# main routine
# ----------------------------
nbd = setbound1()

ip_bd = np.zeros(nbd, dtype=np.int64)  # list of boundary point indices

setbound2()

A = np.zeros((npoin, npoin))  # coefficient matrix
b = np.zeros(npoin)  # right-hand side vector

# assemble finite difference equations
for iy in range(0, ny + 1):
    for ix in range(0, nx + 1):
        ip = (nx + 1) * iy + ix

        # only interior points use the 5-point Laplacian stencil
        if i_bd[ip] == 0:
            A[ip][ip] = -(2.0 / dxdx + 2.0 / dydy)

            # right neighbor
            ip_x_p = (nx + 1) * iy + ix + 1
            if i_bd[ip_x_p] == 0:
                A[ip_x_p][ip] = 1.0 / dxdx
            else:
                A[ip_x_p][ip] = 0.0
                b[ip] += -1.0 / dxdx * v_bd[ip_x_p]

            # left neighbor
            ip_x_m = (nx + 1) * iy + ix - 1
            if i_bd[ip_x_m] == 0:
                A[ip_x_m][ip] = 1.0 / dxdx
            else:
                A[ip_x_m][ip] = 0.0
                b[ip] += -1.0 / dxdx * v_bd[ip_x_m]

            # upper neighbor
            ip_y_p = (nx + 1) * (iy + 1) + ix
            if i_bd[ip_y_p] == 0:
                A[ip_y_p][ip] = 1.0 / dydy
            else:
                A[ip_y_p][ip] = 0.0
                b[ip] += -1.0 / dydy * v_bd[ip_y_p]

            # lower neighbor
            ip_y_m = (nx + 1) * (iy - 1) + ix
            if i_bd[ip_y_m] == 0:
                A[ip_y_m][ip] = 1.0 / dydy
            else:
                A[ip_y_m][ip] = 0.0
                b[ip] += -1.0 / dydy * v_bd[ip_y_m]

# impose Dirichlet conditions on boundary points
for ibd in range(0, nbd):
    ip = ip_bd[ibd]
    A[ip][ip] = -(2.0 / dxdx + 2.0 / dydy)
    b[ip] = A[ip][ip] * v_bd[ip]

# solve linear system
u = np.linalg.solve(A, b)

# save all grid values to file
with open("ffdmall.txt", "w", encoding="utf-8") as fall:
    for iy in range(0, ny + 1):
        y = y0 + iy * dy
        for ix in range(0, nx + 1):
            x = x0 + ix * dx
            ip = (nx + 1) * iy + ix
            print(x, y, u[ip], file=fall)
        print(file=fall)
