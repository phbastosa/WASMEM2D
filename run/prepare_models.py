import numpy as np

nx = 1001
nz = 301

dx = 10.0
dz = 10.0

ns = 1
nr = 901

SPS = np.zeros((ns, 2))
RPS = np.zeros((nr, 2))
XPS = np.zeros((ns, 3))

SPS[:,0] = 5008.2
SPS[:,1] = 502.7 

RPS[:,0] = np.linspace(500, 9500, nr)
RPS[:,1] = 2500 

XPS[:,0] = np.arange(ns)
XPS[:,1] = np.zeros(ns) 
XPS[:,2] = np.zeros(ns) + nr 

np.savetxt("../inputs/geometry/precision_test_SPS.txt", SPS, fmt = "%.2f", delimiter = ",")
np.savetxt("../inputs/geometry/precision_test_RPS.txt", RPS, fmt = "%.2f", delimiter = ",")
np.savetxt("../inputs/geometry/precision_test_XPS.txt", XPS, fmt = "%.0f", delimiter = ",")

vp = np.array([1500])
vs = np.array([ 0.0])
ro = np.array([1000])
z = np.array([])

E = np.array([0.0])
D = np.array([0.0])

tht = np.array([0.0]) * np.pi/180.0

Vp = np.zeros((nz, nx))
Ro = np.zeros((nz, nx))

C11 = np.zeros_like(Vp)
C13 = np.zeros_like(Vp)
C15 = np.zeros_like(Vp)
C33 = np.zeros_like(Vp)
C35 = np.zeros_like(Vp)
C55 = np.zeros_like(Vp)

C = np.zeros((3,3))
M = np.zeros((3,3))

c11 = c13 = c15 = 0
c33 = c35 = 0
c55 = 0

SI = 1e9

for i in range(len(vp)):
    
    layer = int(np.sum(z[:i])/dz)

    c33 = ro[i]*vp[i]**2 / SI
    c55 = ro[i]*vs[i]**2 / SI

    c11 = c33*(1.0 + 2.0*E[i])

    c13 = np.sqrt((c33 - c55)**2 + 2.0*D[i]*c33*(c33 - c55)) - c55

    C[0,0] = c11; C[0,1] = c13; C[0,2] = c15;  
    C[1,0] = c13; C[1,1] = c33; C[1,2] = c35  
    C[2,0] = c15; C[2,1] = c35; C[2,2] = c55; 

    c = np.cos(tht[i])
    s = np.sin(tht[i])

    sin2 = np.sin(2.0*tht[i])
    cos2 = np.cos(2.0*tht[i])

    M = np.array([[     c**2,     s**2, sin2],
                  [     s**2,     c**2,-sin2],
                  [-0.5*sin2, 0.5*sin2, cos2]])

    Cr = (M @ C @ M.T) * SI

    Vp[layer:] = vp[i]
    Ro[layer:] = ro[i]

    C11[layer:] = Cr[0,0]; C13[layer:] = Cr[0,1]; C15[layer:] = Cr[0,2] 
    C33[layer:] = Cr[1,1]; C35[layer:] = Cr[1,2]; C55[layer:] = Cr[2,2]

Vp.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_vp.bin")
Ro.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_ro.bin")

C11.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_C11.bin")
C13.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_C13.bin")
C15.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_C15.bin")
C33.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_C33.bin")
C35.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_C35.bin")
C55.flatten("F").astype(np.float32, order = "F").tofile("../inputs/models/precision_test_C55.bin")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (12,5))

ax.imshow(Vp, cmap = "jet", extent = [0, (nx-1)*dx, (nz-1)*dz, 0])

ax.plot(SPS[:,0], SPS[:,1], "og")
ax.plot(RPS[:,0], RPS[:,1], "or")

ax.set_xlabel("X [m]", fontsize = 15)
ax.set_ylabel("Z [m]", fontsize = 15)

fig.tight_layout()
plt.show()