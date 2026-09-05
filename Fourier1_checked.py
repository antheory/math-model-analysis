"""An example of initial boundary value problem
of a thermal conduction equation
solution of Fourier series expansion
2020/12/13 Akihiro Nakatani (Osaka University)

最低限のチェックを入れた版：
- タイトル中の Forier -> Fourier を修正
- 未使用 import を整理
- scipy.integrate.quad の呼び出しを読みやすく修正
- open / close を with 文に変更
- インデントの乱れを修正
- 授業用の短い行内コメントを追加
"""

import numpy as np
import math
import scipy

# ----------------------------
# basic settings
# ----------------------------
x0 = 0.0  # left bound
x1 = 1.0  # right bound
nx = 50  # the number of spatial intervals
dx = (x1 - x0) / nx  # space interval

l = x1 - x0  # domain length

t0 = 0.0  # start time
t1 = 2.0  # stop time
nt = 200  # number of time steps
dt = (t1 - t0) / nt  # time increment

nmode = 100  # number of Fourier modes

ckappa = 2.0e-2  # thermal diffusivity

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


def A_integrand(x, n):
    """Integrand for the n-th Fourier coefficient."""
    k = n * math.pi / l
    return u_init(x) * math.sin(k * x)


def A_fourier(n):
    """Compute the n-th Fourier sine-series coefficient by numerical quadrature."""
    val, _ = scipy.integrate.quad(A_integrand, x0, x1, args=(n,))
    val *= 2.0 / l
    return val


def u_phin(x, t, n):
    """Contribution of the n-th Fourier mode at position x and time t."""
    k = n * math.pi / l
    return An[n] * math.exp(-ckappa * k**2 * t) * math.sin(k * x)


# ----------------------------
# precompute Fourier coefficients
# ----------------------------
An = np.zeros(nmode + 1)
for imode in range(1, nmode + 1):  # 1 <= imode <= nmode
    An[imode] = A_fourier(imode)

# ----------------------------
# output each mode at t = 0
# ----------------------------
t = 0.0
for imode in range(1, nmode + 1):
    filename = "fmode1_{0:03d}.txt".format(imode)
    with open(filename, "w", encoding="utf-8") as fmode:
        for ix in range(0, nx + 1):
            x = x0 + ix * dx
            print(t, x, u_phin(x, t, imode), file=fmode)

# ----------------------------
# time evolution of total solution
# ----------------------------
u = np.zeros(nx + 1)

with open("fourier1.txt", "w", encoding="utf-8") as fall, \
     open("fourier1_energy.txt", "w", encoding="utf-8") as fenergy, \
     open("fourier1_every.txt", "w", encoding="utf-8") as fevery:

    for it in range(0, nt + 1):  # 0 <= it <= nt
        t = t0 + it * dt

        # reconstruct solution by summing Fourier modes
        for ix in range(0, nx + 1):
            x = x0 + ix * dx
            u[ix] = 0.0
            for imode in range(1, nmode + 1):
                u[ix] += u_phin(x, t, imode)

        filename = "fourier1_{0:03d}.txt".format(it)
        with open(filename, "w", encoding="utf-8") as fone:
            # trapezoidal-rule estimate of total heat
            energy = (np.sum(u) - u[0] / 2.0 - u[nx] / 2.0) * dx
            print(t, energy, file=fenergy)

            for ix in range(0, nx + 1):
                x = x0 + ix * dx
                print(t, x, u[ix], file=fone)
                print(t, x, u[ix], file=fall)

                if it % 10 == 0:
                    print(t, x, u[ix], file=fevery)

        print(file=fall)
        if it % 10 == 0:
            print(file=fevery)
