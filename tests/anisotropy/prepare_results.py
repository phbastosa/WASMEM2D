import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

nx = 1001
nz = 1001
dh = 10.0

SPS = np.loadtxt("../inputs/geometry/anisotropy_test_SPS.txt", delimiter = ",", dtype = float)
RPS = np.loadtxt("../inputs/geometry/anisotropy_test_RPS.txt", delimiter = ",", dtype = float)
XPS = np.loadtxt("../inputs/geometry/anisotropy_test_XPS.txt", delimiter = ",", dtype = int)

dt = 1e-3
tId = 2000

ncols = 3
nrows = 2

ani = ["F","T"]

title = [[r"$\epsilon = 0$, $\delta = 0$, $\theta = 0$", r"$\epsilon = 0.1$, $\delta = 0$, $\theta = 0$", r"$\epsilon = 0.1$, $\delta = 0$, $\theta = 20°$"], 
         [r"$\epsilon = 0$, $\delta = 0.1$, $\theta = 0$", r"$\epsilon = 0.1$, $\delta = 0.1$, $\theta = 0$", r"$\epsilon = 0.1$, $\delta = 0.1$, $\theta = 20°$"]]

xloc = np.linspace(0, nx-1, 5)
zloc = np.linspace(0, nz-1, 5)

xlab = np.linspace(0, 1e-3*(nx-1)*dh, 5)
zlab = np.linspace(0, 1e-3*(nz-1)*dh, 5)

fig, ax = plt.subplots(ncols = ncols, nrows = nrows, figsize = (12,8))

for i in range(nrows):
    for j in range(ncols):

        d = ani[i]    
        e = ani[0] if j == 0 else ani[1]
        tht = ani[0] if j <= 1 else ani[1] 
        
        eikonal = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/anisotropy_eikonal_e{e}_d{d}_tht{tht}.bin")
        snapshot = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/anisotropy_snapshot_e{e}_d{d}_tht{tht}.bin")

        ax[i,j].imshow(snapshot, cmap = "Greys")
        ax[i,j].contour(eikonal, levels = [tId*dt], colors = "red")
        ax[i,j].plot(SPS[0]/dh, SPS[1]/dh, ".g", label = "Source")    
        ax[i,j].set_title(title[i][j])
        ax[i,j].set_xlabel("X axis", fontsize = 15)
        ax[i,j].set_ylabel("Z axis", fontsize = 15)
        ax[i,j].legend(loc = "upper right", fontsize = 10)
        ax[i,j].set_xticks(xloc)
        ax[i,j].set_yticks(zloc)
        ax[i,j].set_xticklabels(xlab)
        ax[i,j].set_yticklabels(zlab)

fig.tight_layout()
plt.show()


