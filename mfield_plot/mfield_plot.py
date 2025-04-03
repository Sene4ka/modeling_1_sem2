import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.constants import mu_0

D = float(input("Введите диаметр соленоида D(м): "))
R = D / 2
L = float(input("Введите длину соленоида L(м): "))
N = int(input("Введите число витков соленоида N: "))
I = float(input("Введите силу тока в соленоиде I(A): "))
M = float(input("Введите масштаб отрисовки M(max(L, D) * M == max(|x|) == max(|y|)): "))
K = int(input("Введите число разбиений пространства по осям K: "))
n = int(input("Введите число разбиений кольца n: "))

print("Вычисляем...")

k_ld = L / D
k_dl = D / L

t_positions = np.linspace(-L/2, L/2, N)

if L > D:
    x = np.linspace(-M * L, M * L, K)
    y = np.linspace(-M * k_ld * D, M * k_ld * D, K)
else:
    x = np.linspace(-M * k_dl * L, M * k_dl * L, K)
    y = np.linspace(-M * D, M * D, K)
X, Y = np.meshgrid(x, y)

@njit()
def magnetic_field_ring(x_0, y_0, z_0, R, I, x_ring, n=100):
    dphi = 2 * np.pi / n
    Bx, By, Bz = 0, 0, 0
    
    for i in range(0, n):
        phi = i * dphi
        x_elem = x_ring
        y_elem = R * np.cos(phi)
        z_elem = R * np.sin(phi)

        phi_n = (i + 1) * dphi
        x_elem_n = x_ring
        y_elem_n = R * np.cos(phi_n)
        z_elem_n = R * np.sin(phi_n)

        dl = [x_elem_n - x_elem, y_elem_n - y_elem, z_elem_n - z_elem]
        
        rx = x_0 - x_elem
        ry = y_0 - y_elem
        rz = z_0 - z_elem

        r_v = [rx, ry, rz]
        
        r = np.sqrt(rx**2 + ry**2 + rz**2)

        dl_x_r = np.cross(dl, r_v)
        
        dB = (mu_0 * I * dl_x_r) / (4 * np.pi * r**3)
        
        Bx += dB[0]
        By += dB[1]
        Bz += dB[2]

    return Bx, By, Bz

@njit()
def calc_mfield(X, Y, R, I, n, t_positions):
    Bx, By = np.zeros_like(X), np.zeros_like(Y)
    for x_ring in t_positions:
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                bx, by, bz = magnetic_field_ring(X[i, j], Y[i, j], 0, R, I, x_ring, n=n)
                Bx[i, j] += bx
                By[i, j] += by
    return Bx, By

Bx, By = calc_mfield(X, Y, R, I, n, t_positions)

mask = (((Y >= -1.05 * R) & (Y <= -0.95 * R)) | ((Y >= 0.95 * R) & (Y <= 1.05 * R))) & ((X >= -L/2) & (X <= L/2))
Bx[mask] = np.nan
By[mask] = np.nan

plt.figure(figsize=(8, 6))

B = np.sqrt(Bx**2 + By**2)

strmplt = plt.streamplot(X, Y, Bx, By, color=B, cmap='plasma', linewidth=0.8)

cbar = plt.colorbar(strmplt.lines)
cbar.set_label("B(Тл)")

plt.plot([-L/2, L/2], [R, R], color='black')
plt.plot([-L/2, L/2], [-R, -R], color='black')
plt.plot([min(x), max(x)], [0, 0], color='black', linestyle='--')

plt.plot([-L/2, -L/2], [R, -R], color='black', linestyle='--')
plt.plot([L/2, L/2], [R, -R], color='black', linestyle='--') 

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Магнитное поле соленоида в плоскости (x,y)")
plt.show()





