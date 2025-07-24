import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as sp

def sinc(x):
    return 1.0 if np.abs(x) < 1e-8 else np.sin(np.pi * x) / (np.pi * x)

def bessel_i0(x):
    sum = term = k = 1.0
    
    while term > 1e-10: 
        term *= (x / (2.0 * k)) * (x / (2.0 * k))
        sum += term
        k += 1.0

    return sum

def kaiser_weights(x, z, ix0, iz0, dx, dz, beta):

    N = 5

    weights = np.zeros((N,N))

    r_max = 2.0*np.sqrt(dx*dx + dz*dz)

    I0_beta = bessel_i0(beta)

    sum = 0.0

    for i in range(N):

        zi = (iz0 + i - 2) * dz
        dzr = (z - zi) / dz
        
        for j in range(N):
        
            xj = (ix0 + j - 2) * dx
            dxr = (x - xj) / dx

            rz = z - zi
            rx = x - xj

            r = np.sqrt(rx * rx + rz * rz)
            
            rnorm = 2.0 * r / r_max

            wij = 0.0
            if rnorm <= 1.0: 
                arg = beta * np.sqrt(1.0 - rnorm * rnorm)
                wij = bessel_i0(arg) / I0_beta

            sinc_term = sinc(dxr) * sinc(dzr)

            weights[i][j] = sinc_term * wij

            sum += weights[i][j]

    for i in range(N):
        for j in range(N):
            weights[i][j] /= sum

    return weights

def gaussian_weights(x, z, ix0, iz0, dx, dz):
    
    N = 5

    weights = np.zeros((N,N))

    r_max = 2.0*np.sqrt(dx*dx + dz*dz)

    sum = 0.0

    for i in range(N):

        zi = (iz0 + i - 2) * dz

        for j in range(N):
        
            xj = (ix0 + j - 2) * dx

            rz = z - zi
            rx = x - xj

            r = np.sqrt(rx * rx + rz * rz) / r_max

            weights[i][j] = 1.0/np.sqrt(2.0*np.pi)*np.exp(-0.5*r*r)

            sum += weights[i][j]

    for i in range(N):
        for j in range(N):
            weights[i][j] /= sum

    return weights

dx = dz = 10

x = 5000
z = 500

ix0 = int((x + 0.5*dx) / dx)
iz0 = int((z + 0.5*dz) / dz)

gaussian = gaussian_weights(x,z,ix0,iz0,dx,dz)
kaiser = kaiser_weights(x,z,ix0,iz0,dx,dz,5.0)

plt.subplot(131)
plt.imshow(kaiser)

plt.subplot(132)
plt.imshow(gaussian)

plt.subplot(133)
plt.imshow(gaussian + 0.5*np.max(np.abs(gaussian))/np.max(np.abs(kaiser))*kaiser)

plt.show()