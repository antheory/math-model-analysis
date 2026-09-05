"""An example of boundary value problem
of Laplace equation using Monte Carlo method
2020/12/21 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- 誤記コメントを修正
- 未使用 import / 未使用代入を整理
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

y0 = 0.0  # lower bound
y1 = 1.0  # upper bound
ly = y1 - y0  # vertical length
ny = 20  # the number of vertical segments
dy = ly / ny  # spatial interval in y

h = 1.0       # boundary value amplitude
nwalk = 1000  # number of random walks per grid point

npoin = (nx + 1) * (ny + 1)  # total number of grid points

u = np.zeros(npoin)  # solution at grid points

i_bd = np.zeros(npoin, dtype=np.int64)  # boundary flag / boundary index after relabeling
v_bd = np.zeros(npoin)                  # prescribed boundary value


def setbound1():
    """Set Dirichlet boundary conditions on the outer boundary."""
    # lower boundary
    iy = 0
    for ix in range(0, nx + 1):
        x = x0 + ix * dx
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = h * math.sin(math.pi * x / lx)

    # upper boundary
    iy = ny
    for ix in range(0, nx + 1):
        x = x0 + ix * dx
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = h * math.sin(math.pi * x / lx)

    # left boundary
    ix = 0
    for iy in range(0, ny + 1):
        y = y0 + iy * dy
        ip = (nx + 1) * iy + ix
        i_bd[ip] = 1
        v_bd[ip] = -h * math.sin(math.pi * y / ly)

    # right boundary
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
    """Store boundary point indices into ip_bd and relabel boundary flags."""
    ibd = 0
    for ip in range(0, npoin):
        if i_bd[ip] != 0:
            i_bd[ip] = ibd
            ip_bd[ibd] = ip
            ibd += 1


def randomwalk(ix_start, iy_start):
    """Random walk from an interior point until it reaches the boundary."""
    ix_walk = ix_start
    iy_walk = iy_start
    ip_walk = (nx + 1) * iy_walk + ix_walk

    # continue walking while the point is still interior
    while i_bd[ip_walk] == 0:
        direction = np.random.randint(0, 4)

        if direction == 0:
            ix_walk -= 1
        elif direction == 1:
            ix_walk += 1
        elif direction == 2:
            iy_walk -= 1
        else:
            iy_walk += 1

        ip_walk = (nx + 1) * iy_walk + ix_walk

    # return the boundary index finally reached
    ibd = i_bd[ip_walk]
    return ibd


# ----------------------------
# main routine
# ----------------------------
nbd = setbound1()

ip_bd = np.zeros(nbd, dtype=np.int64)   # list of boundary point indices
ipr_bd = np.zeros(nbd, dtype=np.int64)  # hit count for each boundary point

setbound2()

# compute solution at every grid point by Monte Carlo sampling
with open("fMonteCarloall.txt", "w", encoding="utf-8") as fall:
    for iy in range(0, ny + 1):
        y = y0 + iy * dy
        for ix in range(0, nx + 1):
            x = x0 + ix * dx
            ip = (nx + 1) * iy + ix

            # reset boundary hit counts
            for ibd in range(0, nbd):
                ipr_bd[ibd] = 0

            # perform random walks from the current point
            for iwalk in range(0, nwalk):
                ibd = randomwalk(ix, iy)
                ipr_bd[ibd] += 1

            # estimate solution as expected boundary value
            u[ip] = 0.0
            for ibd in range(0, nbd):
                u[ip] += float(ipr_bd[ibd]) / float(nwalk) * v_bd[ip_bd[ibd]]

            print(x, y, u[ip], file=fall)
        print(file=fall)
