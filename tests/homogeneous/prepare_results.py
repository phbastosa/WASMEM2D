import sys; sys.path.append("../src/")

import numpy as np
import matplotlib.pyplot as plt
import functions as pyf

from matplotlib.gridspec import GridSpec

nx = 1001
nz = 1001
dh = 10.0

SPS = np.loadtxt("../inputs/geometry/homogeneous_test_SPS.txt", delimiter = ",", dtype = float)
RPS = np.loadtxt("../inputs/geometry/homogeneous_test_RPS.txt", delimiter = ",", dtype = float)
XPS = np.loadtxt("../inputs/geometry/homogeneous_test_XPS.txt", delimiter = ",", dtype = int)

dt = 1e-3
tId = 2000

eikonal_iso = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_iso_eikonal_{nz}x{nx}_shot_1.bin")
snapshot_iso = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_iso_snapshot_step{tId}_{nz}x{nx}_shot_1.bin")

eikonal_ani = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_ani_eikonal_{nz}x{nx}_shot_1.bin")
snapshot_ani = pyf.read_binary_matrix(nz, nx, f"../outputs/snapshots/elastic_ani_snapshot_step{tId}_{nz}x{nx}_shot_1.bin")

x = np.arange(nx)*dh
trace = np.zeros(nx) + 2500

trace_iso = snapshot_iso[int(2500/dh),:]
trace_ani = snapshot_ani[int(2500/dh),:]

trace_iso *= 1.0 / np.max(np.abs(trace_ani))
trace_ani *= 1.0 / np.max(np.abs(trace_ani))

xloc = np.linspace(0, nx-1, 5)
zloc = np.linspace(0, nz-1, 5)

xlab = np.linspace(0, 1e-3*(nx-1)*dh, 5)
zlab = np.linspace(0, 1e-3*(nz-1)*dh, 5)

scale = np.std(snapshot_iso)

fig = plt.figure(figsize=(10, 8))

gs = GridSpec(3, 4, figure = fig)

ax1 = fig.add_subplot(gs[:2,:2]) 
ax1.imshow(snapshot_iso, cmap = "Greys", vmin = -scale, vmax = scale)
ax1.contour(eikonal_iso, levels = [tId*dt], colors = "red")
ax1.plot(x/dh, trace/dh, "--k", label = "Trace")
ax1.plot(RPS[:,0]/dh, RPS[:,1]/dh, ".b", label = "Receivers")
ax1.plot(SPS[0]/dh, SPS[1]/dh, ".g", label = "Source")
ax1.legend(loc = "lower right", fontsize = 10)
ax1.set_title("Elastic ISO", fontsize = 15)
ax1.set_xlabel("X axis [km]", fontsize = 15)
ax1.set_ylabel("Z axis [km]", fontsize = 15)
ax1.set_xticks(xloc)
ax1.set_yticks(zloc)
ax1.set_xticklabels(xlab)
ax1.set_yticklabels(zlab)

ax2 = fig.add_subplot(gs[:2,2:]) 
ax2.imshow(snapshot_ani, cmap = "Greys", vmin = -scale, vmax = scale)
ax2.contour(eikonal_ani, levels = [tId*dt], colors = "red")
ax2.plot(x/dh, trace/dh, "--k", label = "Trace")
ax2.plot(RPS[:,0]/dh, RPS[:,1]/dh, ".b", label = "Receivers")
ax2.plot(SPS[0]/dh, SPS[1]/dh, ".g", label = "Source")
ax2.legend(loc = "lower right", fontsize = 10)
ax2.set_title("Elastic ANI", fontsize = 15)
ax2.set_xlabel("X axis [km]", fontsize = 15)
ax2.set_ylabel("Z axis [km]", fontsize = 15)
ax2.set_xticks(xloc)
ax2.set_yticks(zloc)
ax2.set_xticklabels(xlab)
ax2.set_yticklabels(zlab)

ax3 = fig.add_subplot(gs[2:,:]) 
ax3.plot(x/dh, trace_iso, label = "Trace ISO")
ax3.plot(x/dh, trace_ani, label = "Trace ANI")
ax3.legend(loc = "lower right", fontsize = 10)
ax3.set_title("Trace comparison", fontsize = 15)
ax3.set_xlabel("X axis [km]", fontsize = 15)
ax3.set_ylabel("Norm. Amp.", fontsize = 15)
ax3.set_xlim([0, x[-1]/dh])
ax3.set_xticks(xloc)
ax3.set_xticklabels(xlab)

fig.tight_layout() 
plt.show()

nt = 5001
nr = len(RPS)

xloc = np.linspace(0, nr-1, 5)
xlab = np.linspace(0, nr, 5, dtype = int)

tloc = np.linspace(0, nt-1, 11)
tlab = np.linspace(0, (nt-1)*dt, 11)

seismogram_iso = pyf.read_binary_matrix(nt, nr, f"../outputs/seismograms/elastic_iso_Ps_nStations{nr}_nSamples{nt}_shot_1.bin")
seismogram_ani = pyf.read_binary_matrix(nt, nr, f"../outputs/seismograms/elastic_ani_Ps_nStations{nr}_nSamples{nt}_shot_1.bin")

trace_iso = seismogram_iso[:,int(0.5*nr)].copy()
trace_ani = seismogram_ani[:,int(0.5*nr)].copy()

trace_iso *= 1.0 / np.max(trace_ani)
trace_ani *= 1.0 / np.max(trace_ani)

fft_iso = np.fft.fft(trace_iso)
fft_ani = np.fft.fft(trace_ani)

freqs = np.fft.fftfreq(nt,dt)
mask = freqs >= 0

rectangle = np.array([[40, 2.4], [40, 3.4], [60, 3.4], [60, 2.4], [40, 2.4]])

t = np.arange(nt)*dt

scale = np.std(seismogram_iso)

fig = plt.figure(figsize=(15, 8))

gs = GridSpec(1, 6, figure = fig)

ax1 = fig.add_subplot(gs[:,:2]) 
ax1.imshow(seismogram_iso, aspect = "auto", cmap = "Greys")
ax1.plot(rectangle[:,0], rectangle[:,1]/dt, "--k")
ax1.set_title("Elastic ISO", fontsize = 15)
ax1.set_xlabel("Trace Index", fontsize = 15)
ax1.set_ylabel("Time [s]", fontsize = 15)
ax1.set_xticks(xloc)
ax1.set_yticks(tloc)
ax1.set_xticklabels(xlab)
ax1.set_yticklabels(tlab)

ax2 = fig.add_subplot(gs[:,2:4]) 
ax2.imshow(seismogram_ani, aspect = "auto", cmap = "Greys")
ax2.plot(rectangle[:,0], rectangle[:,1]/dt, "--k")
ax2.set_title("Elastic ANI", fontsize = 15)
ax2.set_xlabel("Trace Index", fontsize = 15)
ax2.set_ylabel("Time [s]", fontsize = 15)
ax2.set_xticks(xloc)
ax2.set_yticks(tloc)
ax2.set_xticklabels(xlab)
ax2.set_yticklabels(tlab)

ts = slice(int(2.4/dt), int(3.4/dt))

ax3 = fig.add_subplot(gs[:,4:5])
ax3.plot(trace_iso[ts], t[ts], label = "Trace ISO")
ax3.plot(trace_ani[ts], t[ts], label = "Trace ANI")
ax3.legend(loc = "lower right", fontsize = 10)
ax3.set_ylim([2.4, 3.4])
ax3.invert_yaxis()
ax3.set_title("Trace", fontsize = 15)
ax3.set_xlabel("Norm. Amp.", fontsize = 15)
ax3.set_ylabel("Time [s]", fontsize = 15)

ax4 = fig.add_subplot(gs[:,5:]) 
ax4.plot(np.abs(fft_iso[mask]), freqs[mask], label = "Trace ISO")
ax4.plot(np.abs(fft_ani[mask]), freqs[mask], label = "Trace ANI")
ax4.legend(loc = "lower right", fontsize = 10)
ax4.set_ylim([0, 16])
ax4.invert_yaxis()
ax4.set_title("Amp. Spectra", fontsize = 15)
ax4.set_xlabel("Norm. Amp.", fontsize = 15)
ax4.set_ylabel("Frequency [Hz]", fontsize = 15)

fig.tight_layout()
plt.show()