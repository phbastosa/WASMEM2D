import numpy as np
import matplotlib.pyplot as plt

nx = 1001
nz = 251

dh = 10.0

SPS = np.loadtxt("../inputs/geometry/precision_test_SPS.txt", delimiter = ",", dtype = np.float32)
RPS = np.loadtxt("../inputs/geometry/precision_test_RPS.txt", delimiter = ",", dtype = np.float32)

vp = np.fromfile("../inputs/models/precision_test_vp.bin", dtype = np.float32, count = nx*nz).reshape([nz,nx], order = "F")

fig, ax = plt.subplots(figsize = (15,5))

im = ax.imshow(vp, cmap = "jet", vmin = 1400, vmax = 2000, extent = [0, (nx-1)*dh, (nz-1)*dh, 0])

ax.plot(SPS[0], SPS[1], "ok")
ax.plot(RPS[:,0], RPS[:,1], "or")

ax.set_xlabel("Distance [m]", fontsize = 15)
ax.set_ylabel("Depth [m]", fontsize = 15)

fig.tight_layout()
plt.show()




nt = 14001
dt = 5e-4

nr = 17
dr = 600

path = "../outputs/seismograms/elastic_iso_nStations17_nSamples14001_shot_1"
path2 = "../outputs/seismograms/elastic_ani_nStations17_nSamples14001_shot_1.bin"

seism1 = np.fromfile(path + "_5m_raw.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seism2 = np.fromfile(path + "_10m_fixed.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seism3 = np.fromfile(path + "_10m_raw.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seism4 = np.fromfile(path2, count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

fig, ax = plt.subplots(ncols = 3, figsize = (15, 7))

ax[0].imshow(seism1, aspect = "auto", cmap = "Greys", extent = [200, (nr-1)*dr, (nt-1)*dt, 0])
ax[0].set_title("dh = 5 m", fontsize = 15)
ax[0].set_xlabel("Offset [m]", fontsize = 15)
ax[0].set_ylabel("Time [s]", fontsize = 15)

ax[1].imshow(seism2, aspect = "auto", cmap = "Greys", extent = [200, (nr-1)*dr, (nt-1)*dt, 0])
ax[1].set_title("dh = 10 m compensed", fontsize = 15)
ax[1].set_xlabel("Offset [m]", fontsize = 15)
ax[1].set_ylabel("Time [s]", fontsize = 15)

ax[2].imshow(seism3, aspect = "auto", cmap = "Greys", extent = [200, (nr-1)*dr, (nt-1)*dt, 0])
ax[2].set_title("dh = 10 m uncompensed", fontsize = 15)
ax[2].set_xlabel("Offset [m]", fontsize = 15)
ax[2].set_ylabel("Time [s]", fontsize = 15)

fig.tight_layout()

plt.show()

time = np.arange(nt)*dt

trace1 = seism1[:,1] / np.max(seism1[:,1])
trace2 = seism2[:,1] / np.max(seism2[:,1])
trace3 = seism3[:,1] / np.max(seism3[:,1])
trace4 = seism4[:,1] / np.max(seism4[:,1])

fig, ax = plt.subplots(figsize = (4,8))

ax.plot(trace1, time, "k")
ax.plot(trace2, time, "g")
ax.plot(trace3, time, "r")

ax.set_ylim([1,1.7])

ax.invert_yaxis()
fig.tight_layout()
plt.show()

time0 = np.sqrt((SPS[0] - RPS[1,0])**2 + (SPS[1] - RPS[1,1])**2) / 1500

time1 = np.argmax(trace1)*dt
time2 = np.argmax(trace2)*dt
time3 = np.argmax(trace3)*dt
time4 = np.argmax(trace4)*dt

print(time0, time1, time2, time3, time4)

