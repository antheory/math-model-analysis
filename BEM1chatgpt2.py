To: 
Subject: 
From: Akihiro Nakatani <nakatani@mech.eng.osaka-u.ac.jp>
Fcc: +inbox
X-Mailer: Mew version 6.9 on Emacs 29.4



"""
Laplace方程式の境界値問題を境界要素法(BEM)で解く例
  - 境界：単位正方形 [0,1]x[0,1]
  - 境界要素：一定要素（各辺を等分し、要素中心でコロケーション）
  - 積分：4点Gauss積分
  - 出力：内部格子点(境界を除く)でのポテンシャル u を fbemall.txt に保存

Original: 2020/12/21 Akihiro Nakatani (Osaka University)
Refined (readability/educational): 2026/03/01
"""

import numpy as np
import math

# -----------------------------
# 1) 計算領域（正方形）と分割数
# -----------------------------
x0, x1 = 0.0, 1.0   # x方向の下限・上限
y0, y1 = 0.0, 1.0   # y方向の下限・上限
lx, ly = x1 - x0, y1 - y0

nx, ny = 40, 40     # 各辺の分割数（要素数に相当）
dx, dy = lx / nx, ly / ny

h = 1.0             # 境界値の振幅

# 境界要素は「境界上の線分」：合計 nelem
# 境界節点は角を重複させないので npoin = nelem
npe = 2      # 1要素あたりの節点数（線分なので2）
ndim = 2     # 2次元

npoin = 2 * (nx + ny)
nelem = 2 * (nx + ny)

# -----------------------------
# 2) 境界上の未知量
#   u  : 境界上のポテンシャル
#   un : 境界上の法線微分 du/dn
# -----------------------------
u = np.zeros(nelem)
un = np.zeros(nelem)

# 要素中心座標 xe[ie]、節点座標 xp[ip]、要素の接続 icon[ie] = (ip0,ip1)
xe = np.zeros((nelem, ndim))
xp = np.zeros((npoin, ndim))
icon = np.zeros((nelem, npe), dtype=np.int64)

# 境界条件の指定
# i_bd[ie] = 0: Dirichlet（uが既知）
# i_bd[ie] = 1: Neumann  （unが既知）
i_bd = np.zeros(nelem, dtype=np.int64)
v_bd = np.zeros(nelem)  # 既知値（Dirichletならu、Neumannならun）

# -----------------------------
# 3) ガウス積分（4点）
# -----------------------------
ng = 4
xig = np.array([-0.86113631, -0.33998104, 0.33998104, 0.86113631])
wg  = np.array([ 0.34785485,  0.65214515, 0.65214515, 0.34785485])

# 要素長 re[ie] と 外向き単位法線 unx[ie]
re = np.zeros(nelem)
unx = np.zeros((nelem, ndim))


# =========================================================
# 境界条件：ここでは全境界 Dirichlet（u既知）を与える
#   下辺 y=y0 : u =  h sin(pi x / lx)
#   右辺 x=x1 : u = -h sin(pi y / ly)
#   上辺 y=y1 : u =  h sin(pi x / lx)
#   左辺 x=x0 : u = -h sin(pi y / ly)
# =========================================================
def set_boundary_conditions():
    # デフォルトは Neumann(un=0) として初期化（例として残す）
    for ie in range(nelem):
        i_bd[ie] = 1
        v_bd[ie] = 0.0

    # bottom（要素番号 ie = 0..nx-1）
    for ix in range(nx):
        x = x0 + (ix + 0.5) * dx  # 要素中心
        ie = ix
        i_bd[ie] = 0
        v_bd[ie] = h * math.sin(math.pi * x / lx)

    # right（ie = nx..nx+ny-1）
    for iy in range(ny):
        y = y0 + (iy + 0.5) * dy
        ie = nx + iy
        i_bd[ie] = 0
        v_bd[ie] = -h * math.sin(math.pi * y / ly)

    # top（ie = nx+ny..nx+ny+nx-1）
    for ix in range(nx):
        x = x1 - (ix + 0.5) * dx
        ie = (nx + ny) + ix
        i_bd[ie] = 0
        v_bd[ie] = h * math.sin(math.pi * x / lx)

    # left（ie = 2*nx+ny..2*nx+2*ny-1）
    for iy in range(ny):
        y = y1 - (iy + 0.5) * dy
        ie = 2 * nx + ny + iy
        i_bd[ie] = 0
        v_bd[ie] = -h * math.sin(math.pi * y / ly)


# =========================================================
# 境界節点 xp を正方形の周に反時計回りに並べる
# 角は重複させない（npoin = 2(nx+ny)）
# =========================================================
def build_boundary_nodes():
    ip = 0

    # bottom: (x0,y0) -> (x1,y0)
    for ix in range(nx):
        xp[ip, 0] = x0 + ix * dx
        xp[ip, 1] = y0
        ip += 1

    # right: (x1,y0) -> (x1,y1)
    for iy in range(ny):
        xp[ip, 0] = x1
        xp[ip, 1] = y0 + iy * dy
        ip += 1

    # top: (x1,y1) -> (x0,y1)
    for ix in range(nx):
        xp[ip, 0] = x1 - ix * dx
        xp[ip, 1] = y1
        ip += 1

    # left: (x0,y1) -> (x0,y0)
    for iy in range(ny):
        xp[ip, 0] = x0
        xp[ip, 1] = y1 - iy * dy
        ip += 1


# =========================================================
# 要素（線分）の接続 icon と幾何（中心 xe, 長さ re, 法線 unx）を作る
# =========================================================
def build_elements():
    # connectivity: ie番要素は節点 ie と ie+1 を結ぶ（最後は0に戻る）
    for ie in range(nelem):
        icon[ie, 0] = ie
        icon[ie, 1] = (ie + 1) % npoin

    # geometry
    for ie in range(nelem):
        ip0, ip1 = icon[ie, 0], icon[ie, 1]

        # 要素中心
        xe[ie, 0] = 0.5 * (xp[ip0, 0] + xp[ip1, 0])
        xe[ie, 1] = 0.5 * (xp[ip0, 1] + xp[ip1, 1])

        # 接線ベクトル（p0->p1）
        sx0 = xp[ip1, 0] - xp[ip0, 0]
        sx1 = xp[ip1, 1] - xp[ip0, 1]

        # 要素長
        re[ie] = math.sqrt(sx0**2 + sx1**2)

        # 外向き単位法線（境界を反時計回りに並べたときの外向き）
        unx[ie, 0] =  sx1 / re[ie]
        unx[ie, 1] = -sx0 / re[ie]


# =========================================================
# 影響係数（A,B）を計算：
#   collocation点 (x_col, y_col) に対して、
#   source要素 ie の積分を行い a,b を返す
#   a: double-layer（法線方向）に対応
#   b: single-layer（log r）に対応
# =========================================================
def influence_coefficients(ie, x_col, y_col):
    ip0, ip1 = icon[ie, 0], icon[ie, 1]
    sx0 = xp[ip1, 0] - xp[ip0, 0]
    sx1 = xp[ip1, 1] - xp[ip0, 1]

    a = 0.0
    b = 0.0

    for ig in range(ng):
        # ガウス点（要素中心 ± 接線 * xi/2）
        xg = xe[ie, 0] + (sx0 / 2.0) * xig[ig]
        yg = xe[ie, 1] + (sx1 / 2.0) * xig[ig]

        rx = xg - x_col
        ry = yg - y_col
        r = math.sqrt(rx**2 + ry**2)

        # 元コードの式をそのまま使用
        a += -(unx[ie, 0] * rx + unx[ie, 1] * ry) / (r**2) * wg[ig] * (re[ie] / 2.0)
        b += -math.log(r) * wg[ig] * (re[ie] / 2.0)

    return a, b


# =========================================================
# 内点でのポテンシャル評価（境界積分表示）
# =========================================================
def potential_inside(x, y):
    val = 0.0
    for ie in range(nelem):
        a, b = influence_coefficients(ie, x, y)
        val += un[ie] * b - u[ie] * a
    return val / (2.0 * math.pi)


# -----------------------------
# main
# -----------------------------
set_boundary_conditions()
build_boundary_nodes()
build_elements()

# 連立方程式 A * unknown = rhs
amat = np.zeros((nelem, nelem))
bmat = np.zeros((nelem, nelem))
rhs  = np.zeros(nelem)

# A,B行列の組み立て
for i_col in range(nelem):
    for j_src in range(nelem):
        if i_col == j_src:
            # 自己項（解析的に処理）
            amat[i_col, j_src] = math.pi
            bmat[i_col, j_src] = re[j_src] * (1.0 - math.log(re[j_src] / 2.0))
        else:
            a, b = influence_coefficients(j_src, xe[i_col, 0], xe[i_col, 1])
            amat[i_col, j_src] = a
            bmat[i_col, j_src] = b

# Dirichlet境界（u既知）の列を「未知量がunになる」ように入れ替える
# （元コードの列交換ロジックをそのまま）
for j in range(nelem):
    if i_bd[j] == 0:  # Dirichlet
        for i in range(nelem):
            work = bmat[i, j]
            bmat[i, j] = -amat[i, j]
            amat[i, j] = -work

# rhs = B * 既知値
for i in range(nelem):
    rhs[i] = 0.0
    for j in range(nelem):
        rhs[i] += bmat[i, j] * v_bd[j]

# 解く
sol = np.linalg.solve(amat, rhs)

# 解ベクトル sol から u, un を復元
for ie in range(nelem):
    if i_bd[ie] == 0:      # Dirichlet: u既知、unが未知
        un[ie] = sol[ie]
        u[ie]  = v_bd[ie]
    else:                 # Neumann: un既知、uが未知
        un[ie] = v_bd[ie]
        u[ie]  = sol[ie]

# 内点の値を出力（境界を除く格子点）
with open("fbemall.txt", "w") as f:
    for iy in range(1, ny):
        y = y0 + iy * dy
        for ix in range(1, nx):
            x = x0 + ix * dx
            ui = potential_inside(x, y)
            print(x, y, ui, file=f)
        print(file=f)

#-----------------------------------------------------------------------------
# =========================================================
# 可視化：fbemall.txt を読み込んで等高線図を描く
# =========================================================
import matplotlib.pyplot as plt

# 内点格子のサイズ
nx_in = nx - 1
ny_in = ny - 1

# データ格納用
x_vals = np.zeros(nx_in)
y_vals = np.zeros(ny_in)
u_vals = np.zeros((ny_in, nx_in))

# ファイル読み込み
with open("fbemall.txt", "r") as f:
    iy = 0
    ix = 0
    for line in f:
        if line.strip() == "":
            iy += 1
            ix = 0
            continue

        x, y, uin = map(float, line.split())
        if iy == 0:
            x_vals[ix] = x
        if ix == 0:
            y_vals[iy] = y

        u_vals[iy, ix] = uin
        ix += 1

# メッシュ作成
X, Y = np.meshgrid(x_vals, y_vals)

# 等高線図
plt.figure(figsize=(6, 5))
cont = plt.contourf(X, Y, u_vals, levels=20, cmap="viridis")
plt.colorbar(cont, label="Potential u")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Laplace equation (BEM solution)")

plt.axis("equal")
plt.tight_layout()
plt.show()
