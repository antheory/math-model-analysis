"""An example of initial boundary value problem
of a thermal conduction equation
using finite difference method
2020/12/13 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- 未使用 import を整理
- open / close を with 文に変更
- 行内コメントを追加
- 変数名を少しだけ読みやすく調整
- 計算の流れは元コードのまま維持
"""

import numpy as np
import math

# ----------------------------
# basic settings
# ----------------------------
x0 = 0.0  # left bound
x1 = 1.0  # right bound
nx = 50  # the number of spatial intervals
dx = (x1 - x0) / nx  # space interval

l = x1 - x0  # domain length

t0 = 0.0  # start time

# ckappa = 2.0e-2
# t1 = 2.0
# nt = 200

ckappa = 1.0  # thermal diffusivity
t1 = 1.0  # stop time
nt = 10000  # time steps
dt = (t1 - t0) / nt  # time increment

coef = ckappa * dt / dx**2  # diffusion number: kappa * dt / dx^2

# ----------------------------
# initial condition
# ----------------------------
xa = 0.25
h = 1.0


def u_init(x):
    """Initial temperature distribution."""
    if x < xa:
        return 0.0
    elif x < (1.0 - xa):
        return h
    else:
        return 0.0


# def u_init(x):
#     return 1.0 * math.sin(2.0 * math.pi * x / (2.0 * l)) \
#          + 0.2 * math.sin(2.0 * math.pi * 9.0 * x / (2.0 * l))


def intsimpson(u, nx, dx):
    """Integral of u over the domain by Simpson's rule."""
    sum_even = 0.0
    for kx in range(1, int(nx / 2)):
        sum_even += u[2 * kx]

    sum_odd = 0.0
    for kx in range(1, int(nx / 2) + 1):
        sum_odd += u[2 * kx - 1]

    return (dx / 3.0) * (u[0] + u[nx] + 2.0 * sum_even + 4.0 * sum_odd)


# ----------------------------
# setting of initial value
# ----------------------------
t = 0.0

u = np.zeros(nx + 1)
for ix in range(0, nx + 1):  # 0 <= ix <= nx
    x = x0 + ix * dx
    u[ix] = u_init(x)

du = np.zeros(nx + 1)  # increment at the next step

# Dirichlet boundary condition is implicitly:
# u[0] = 0, u[nx] = 0
# because only interior points (1 ... nx-1) are updated.

# ----------------------------
# output initial value and time history
# ----------------------------
with open("fdm1sD.txt", "w", encoding="utf-8") as fall, \
     open("fdm1sD_energy.txt", "w", encoding="utf-8") as fenergy, \
     open("fdm1sD_every.txt", "w", encoding="utf-8") as fevery:

    it = 0
    filename = "fdm1sD_{0:03d}.txt".format(it)
    with open(filename, "w", encoding="utf-8") as fone:
        for ix in range(0, nx + 1):  # 0 <= ix <= nx
            x = x0 + ix * dx
            print(x, t, u[ix], file=fall)
            print(x, t, u[ix], file=fone)
            print(x, t, u[ix], file=fevery)

    # total heat (energy-like quantity)
    print(t, (np.sum(u) - u[0] / 2.0 - u[nx] / 2.0) * dx, file=fenergy)
    # print(t, intsimpson(u, nx, dx), file=fenergy)

    print(file=fevery)
    print(file=fall)

    # ----------------------------
    # temporal evolution
    # ----------------------------
    for it in range(1, nt + 1):  # 1 <= it <= nt
        # explicit finite difference update for interior points
        for ix in range(1, nx):  # 1 <= ix <= nx - 1
            du[ix] = coef * (u[ix + 1] + u[ix - 1] - 2.0 * u[ix])

        for ix in range(1, nx):  # 1 <= ix <= nx - 1
            u[ix] += du[ix]

        t += dt

        # integral of temperature distribution
        print(t, intsimpson(u, nx, dx), file=fenergy)

        # output selected time steps
        if it == 1 or it == 10 or it == 100 or it == 1000 or it == 10000:
            filename = "fdm1sD_{0:03d}.txt".format(it)
            with open(filename, "w", encoding="utf-8") as fone:
                for ix in range(0, nx + 1):  # 0 <= ix <= nx
                    x = x0 + ix * dx
                    print(x, t, u[ix], file=fone)
                    print(x, t, u[ix], file=fall)
                    print(x, t, u[ix], file=fevery)

            print(file=fall)
            print(file=fevery)
