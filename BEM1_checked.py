"""An example of boundary value problem
of Laplace equation using a boundary element method
2020/12/21 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- 誤記コメントを修正
- 未使用 import / 未使用変数を整理
- 授業用の行内コメントを追加
- デバッグ出力を切替式に変更
- 出力ファイルを with 文で安全に書き出し
"""

import math
import numpy as np

# ----------------------------
# basic settings
# ----------------------------
DEBUG = False  # True にすると境界条件の割付けを表示

x0 = 0.0  # left bound
x1 = 1.0  # right bound
lx = x1 - x0  # horizontal length
nx = 40  # the number of horizontal segments
dx = lx / nx  # spatial interval in x

y0 = 0.0  # lower bound
y1 = 1.0  # upper bound
ly = y1 - y0  # vertical length
ny = 40  # the number of vertical segments
dy = ly / ny  # spatial interval in y

h = 1.0  # boundary value amplitude

npe = 2  # nodes per element (constant/linear edge handling用の接続情報)
ndim = 2  # spatial dimension

npoin = 2 * (nx + ny)  # number of boundary nodes
nelem = 2 * (nx + ny)  # number of boundary elements

u = np.zeros(nelem)   # boundary potential
un = np.zeros(nelem)  # normal derivative of potential

xe = np.zeros((nelem, ndim))                  # element center coordinates
xp = np.zeros((npoin, ndim))                  # boundary node coordinates
icon = np.zeros((nelem, npe), dtype=np.int64) # element connectivity

i_bd = np.zeros(nelem, dtype=np.int64)  # boundary condition type: 0=Dirichlet, 1=Neumann
v_bd = np.zeros(nelem)                  # prescribed boundary value

# 4-point Gauss quadrature on [-1, 1]
ng = 4
xig = np.array([-0.86113631, -0.33998104, 0.33998104, 0.86113631])
wg = np.array([0.34785485, 0.65214515, 0.65214515, 0.34785485])

re = np.zeros(nelem)           # element length
unx = np.zeros((nelem, ndim))  # outward normal vector of each element


def setbound1():
    """Assign boundary conditions on each boundary element."""
    # まず全境界を Neumann=0 として初期化
    for ie in range(nelem):
        i_bd[ie] = 1
        v_bd[ie] = 0.0

    # lower edge
    for ix in range(nx):
        x = x0 + (ix + 0.5) * dx  # 要素中央座標
        ie = ix
        i_bd[ie] = 0
        v_bd[ie] = h * math.sin(math.pi * x / lx)

    # right edge
    for iy in range(ny):
        y = y0 + (iy + 0.5) * dy
        ie = nx + iy
        i_bd[ie] = 0
        v_bd[ie] = -h * math.sin(math.pi * y / ly)

    # upper edge
    for ix in range(nx):
        x = x1 - (ix + 0.5) * dx
        ie = (nx + ny) + ix
        i_bd[ie] = 0
        v_bd[ie] = h * math.sin(math.pi * x / lx)

    # left edge
    for iy in range(ny):
        y = y1 - (iy + 0.5) * dy
        ie = 2 * nx + ny + iy
        i_bd[ie] = 0
        v_bd[ie] = -h * math.sin(math.pi * y / ly)

    if DEBUG:
        for ie in range(nelem):
            print(ie, i_bd[ie], v_bd[ie])


def setedgenode():
    """Generate boundary nodes counterclockwise along the square."""
    ip = 0

    # lower edge: (x0, y0) -> (x1, y0)
    for ix in range(nx):
        xp[ip][0] = x0 + ix * dx
        xp[ip][1] = y0
        ip += 1

    # right edge: (x1, y0) -> (x1, y1)
    for iy in range(ny):
        xp[ip][0] = x1
        xp[ip][1] = y0 + iy * dy
        ip += 1

    # upper edge: (x1, y1) -> (x0, y1)
    for ix in range(nx):
        xp[ip][0] = x1 - ix * dx
        xp[ip][1] = y1
        ip += 1

    # left edge: (x0, y1) -> (x0, y0)
    for iy in range(ny):
        xp[ip][0] = x0
        xp[ip][1] = y1 - iy * dy
        ip += 1


def setelem():
    """Build element connectivity, center position, length, and normal vector."""
    for ie in range(nelem):
        ip = ie
        icon[ie][0] = ip
        icon[ie][1] = (ip + 1) % npoin

    for ie in range(nelem):
        ip0 = icon[ie][0]
        ip1 = icon[ie][1]

        # element center
        for idim in range(ndim):
            xe[ie][idim] = (xp[ip0][idim] + xp[ip1][idim]) * 0.5

        # tangent vector of the boundary element
        sx0 = xp[ip1][0] - xp[ip0][0]
        sx1 = xp[ip1][1] - xp[ip0][1]

        # element length and outward normal
        re[ie] = math.sqrt(sx0**2 + sx1**2)
        unx[ie][0] = sx1 / re[ie]
        unx[ie][1] = -sx0 / re[ie]


def setmatrixcomponents(ie, px, py):
    """Compute influence coefficients a, b for element ie at collocation point (px, py)."""
    ip0 = icon[ie][0]
    ip1 = icon[ie][1]
    sx0 = xp[ip1][0] - xp[ip0][0]
    sx1 = xp[ip1][1] - xp[ip0][1]

    a = 0.0
    b = 0.0
    for ig in range(ng):
        # Gaussian point on the boundary element
        xg = xe[ie][0] + sx0 / 2.0 * xig[ig]
        yg = xe[ie][1] + sx1 / 2.0 * xig[ig]
        rm = math.sqrt((xg - px) ** 2 + (yg - py) ** 2)

        # a: coefficient for u, b: coefficient for un
        a += -(
            unx[ie][0] * (xg - px) + unx[ie][1] * (yg - py)
        ) / rm**2 * wg[ig] * (re[ie] / 2.0)
        b += -math.log(rm) * wg[ig] * (re[ie] / 2.0)

    return a, b


def calcus(px, py):
    """Evaluate the interior solution at point (px, py)."""
    us_in = 0.0
    for ie in range(nelem):
        a, b = setmatrixcomponents(ie, px, py)
        us_in += un[ie] * b - u[ie] * a
    us_in /= (2.0 * math.pi)
    return us_in


# ----------------------------
# main routine
# ----------------------------
setbound1()
setedgenode()
setelem()

amat = np.zeros((nelem, nelem))
bmat = np.zeros((nelem, nelem))
rhs = np.zeros(nelem)

# build influence matrices at collocation points
for ie1 in range(nelem):
    for ie2 in range(nelem):
        if ie1 == ie2:
            # diagonal term (singular integral treated analytically)
            ie = ie2
            amat[ie][ie] = math.pi
            bmat[ie][ie] = re[ie] * (1.0 - math.log(re[ie] / 2.0))
        else:
            amat[ie1][ie2], bmat[ie1][ie2] = setmatrixcomponents(
                ie2, xe[ie1][0], xe[ie1][1]
            )

# Dirichlet 条件の列では、未知量を u から un に入れ替える
for ie2 in range(nelem):
    if i_bd[ie2] == 0:
        for ie1 in range(nelem):
            work = bmat[ie1][ie2]
            bmat[ie1][ie2] = -amat[ie1][ie2]
            amat[ie1][ie2] = -work

# right-hand side vector
for ie1 in range(nelem):
    rhs[ie1] = 0.0
    for ie2 in range(nelem):
        rhs[ie1] += bmat[ie1][ie2] * v_bd[ie2]

# solve boundary unknowns
u = np.linalg.solve(amat, rhs)

# reconstruct (u, un) in physical meaning on the boundary
for ie in range(nelem):
    if i_bd[ie] == 0:
        un[ie] = u[ie]
        u[ie] = v_bd[ie]
    else:
        un[ie] = v_bd[ie]

# evaluate interior points and save to file
with open("fbemall.txt", "w", encoding="utf-8") as fall:
    for iy in range(1, ny):
        y = y0 + iy * dy
        for ix in range(1, nx):
            x = x0 + ix * dx
            us_in = calcus(x, y)
            print(x, y, us_in, file=fall)
        print(file=fall)