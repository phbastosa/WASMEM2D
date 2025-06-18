import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

nx = 1001
nz = 1001
dh = 10.0

SPS = np.loadtxt("../inputs/geometry/homogeneous_test_SPS.txt", delimiter = ",", dtype = float)
RPS = np.loadtxt("../inputs/geometry/homogeneous_test_RPS.txt", delimiter = ",", dtype = float)
XPS = np.loadtxt("../inputs/geometry/homogeneous_test_XPS.txt", delimiter = ",", dtype = int)

vp = pyf.read_binary_matrix(nz, nx, "../inputs/models/homogeneous_test_vp.bin")

fig, ax = plt.subplots(figsize = (6,6))

ax.imshow(vp, aspect = "auto", cmap = "Greys")
ax.plot(SPS[0]/dh, SPS[1]/dh, "og")
ax.plot(RPS[:,0]/dh, RPS[:,1]/dh, "og")

fig.tight_layout()
plt.show()

dt = 1e-3
tId = 2000

eikonal_iso = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_iso_eikonal_{nz}x{nx}_shot_1.bin")
snapshot_iso = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_iso_snapshot_step{tId}_{nz}x{nx}_shot_1.bin")

eikonal_ani = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_ani_eikonal_{nz}x{nx}_shot_1.bin")
snapshot_ani = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_ani_snapshot_step{tId}_{nz}x{nx}_shot_1.bin")

scale = np.std(snapshot_iso)

fig, ax = plt.subplots(figsize = (10,5), ncols = 2)

ax[0].imshow(snapshot_iso, cmap = "Greys", vmin = -scale, vmax = scale)
ax[0].contour(eikonal_iso, levels = [tId*dt], colors = "red")

ax[1].imshow(snapshot_ani, cmap = "Greys", vmin = -scale, vmax = scale)
ax[1].contour(eikonal_ani, levels = [tId*dt], colors = "red")

fig.tight_layout()
plt.show()