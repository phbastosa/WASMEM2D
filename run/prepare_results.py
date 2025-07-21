import numpy as np
import matplotlib.pyplot as plt

nr = 901
nt = 4001
dt = 1e-3

SPS = np.loadtxt("../inputs/geometry/precision_test_SPS.txt", delimiter = ",", dtype = np.float32)
RPS = np.loadtxt("../inputs/geometry/precision_test_RPS.txt", delimiter = ",", dtype = np.float32)

prefix = "../outputs/seismograms/elastic_iso"
suffix = "nStations901_nSamples4001_shot_1"

seismPs_5m_g = np.fromfile(f"{prefix}_Ps_{suffix}_" + "5m_g.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seismVx_5m_g = np.fromfile(f"{prefix}_Vx_{suffix}_" + "5m_g.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seismVz_5m_g = np.fromfile(f"{prefix}_Vz_{suffix}_" + "5m_g.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

# seismPs_10m_g = np.fromfile(f"{prefix}_Ps_{suffix}_" + "10m_g.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
# seismVx_10m_g = np.fromfile(f"{prefix}_Vx_{suffix}_" + "10m_g.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
# seismVz_10m_g = np.fromfile(f"{prefix}_Vz_{suffix}_" + "10m_g.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

# seismPs_10m_nsx = np.fromfile(f"{prefix}_Ps_{suffix}_" + "10m_nsx.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
# seismVx_10m_nsx = np.fromfile(f"{prefix}_Vx_{suffix}_" + "10m_nsx.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
# seismVz_10m_nsx = np.fromfile(f"{prefix}_Vz_{suffix}_" + "10m_nsx.bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

trace = int(0.2*nr)

time = np.arange(nt)*dt

analytical_time = np.sqrt((SPS[0] - RPS[trace,0])**2 + (SPS[1] - RPS[trace,1])**2) / 1500

tracePs_5m_g = seismPs_5m_g[:,trace] 
traceVx_5m_g = seismVx_5m_g[:,trace] 
traceVz_5m_g = seismVz_5m_g[:,trace] 

# tracePs_10m_g = seismPs_10m_g[:,trace] / np.max(np.abs(seismPs_10m_g[:,trace]))
# traceVx_10m_g = seismVx_10m_g[:,trace] / np.max(np.abs(seismVx_10m_g[:,trace]))
# traceVz_10m_g = seismVz_10m_g[:,trace] / np.max(np.abs(seismVz_10m_g[:,trace]))

# tracePs_10m_nsx = seismPs_10m_nsx[:,trace] / np.max(np.abs(seismPs_10m_nsx[:,trace]))
# traceVx_10m_nsx = seismVx_10m_nsx[:,trace] / np.max(np.abs(seismVx_10m_nsx[:,trace]))
# traceVz_10m_nsx = seismVz_10m_nsx[:,trace] / np.max(np.abs(seismVz_10m_nsx[:,trace]))

n = 101
amp = np.linspace(-1.5, 1.5, n)
att = np.zeros(n) + analytical_time

attPs_5m_g = np.zeros(n) + np.argmax(np.abs(tracePs_5m_g))*dt
attVx_5m_g = np.zeros(n) + np.argmax(np.abs(traceVx_5m_g))*dt
attVz_5m_g = np.zeros(n) + np.argmax(np.abs(traceVz_5m_g))*dt

# attPs_10m_g = np.zeros(n) + np.argmax(np.abs(tracePs_10m_g))*dt
# attVx_10m_g = np.zeros(n) + np.argmax(np.abs(traceVx_10m_g))*dt
# attVz_10m_g = np.zeros(n) + np.argmax(np.abs(traceVz_10m_g))*dt

# attPs_10m_nsx = np.zeros(n) + np.argmax(np.abs(tracePs_10m_nsx))*dt
# attVx_10m_nsx = np.zeros(n) + np.argmax(np.abs(traceVx_10m_nsx))*dt
# attVz_10m_nsx = np.zeros(n) + np.argmax(np.abs(traceVz_10m_nsx))*dt

fig, ax = plt.subplots(ncols = 3, figsize = (8,9))

ax[0].plot(amp, att, "--k")
ax[0].plot(amp, attPs_5m_g, "--", color = "blue")
# ax[0].plot(amp, attPs_10m_g, "--", color = "green")
# ax[0].plot(amp, attPs_10m_nsx, "--", color = "red")
ax[0].plot(tracePs_5m_g, time, color = "blue")
# ax[0].plot(tracePs_10m_g, time, color = "green")
# ax[0].plot(tracePs_10m_nsx, time, color = "red")
ax[0].set_title("Pressure", fontsize = 15)
ax[0].set_ylabel("Time [s]", fontsize = 15)
ax[0].set_xlabel("Amplitude", fontsize = 15)
# ax[0].set_ylim([1.9, 2.1])
ax[0].invert_yaxis()

ax[1].plot(amp, att, "--k")
ax[1].plot(amp, attVx_5m_g, "--", color = "blue")
# ax[1].plot(amp, attVx_10m_g, "--", color = "green")
# ax[1].plot(amp, attVx_10m_nsx, "--", color = "red")
ax[1].plot(traceVx_5m_g, time, color = "blue")
# ax[1].plot(traceVx_10m_g, time, color = "green")
# ax[1].plot(traceVx_10m_nsx, time, color = "red")
ax[1].set_title("Vx", fontsize = 15)
ax[1].set_ylabel("Time [s]", fontsize = 15)
ax[1].set_xlabel("Amplitude", fontsize = 15)
# ax[1].set_ylim([1.9, 2.1])
ax[1].invert_yaxis()

ax[2].plot(amp, att, "--k")
ax[2].plot(amp, attVz_5m_g, "--", color = "blue")
# ax[2].plot(amp, attVz_10m_g, "--", color = "green")
# ax[2].plot(amp, attVz_10m_nsx, "--", color = "red")
ax[2].plot(traceVz_5m_g, time, color = "blue")
# ax[2].plot(traceVz_10m_g, time, color = "green")
# ax[2].plot(traceVz_10m_nsx, time, color = "red")
ax[2].set_title("Vz", fontsize = 15)
ax[2].set_ylabel("Time [s]", fontsize = 15)
ax[2].set_xlabel("Amplitude", fontsize = 15)
# ax[2].set_ylim([1.9, 2.1])
ax[2].invert_yaxis()

fig.tight_layout()
plt.show()

print(f"{att[0] - attPs_5m_g[0]:.4f}", f"{att[0] - attVx_5m_g[0]:.4f}", f"{att[0] - attVz_5m_g[0]:.4f}")
# print(f"{att[0] - attPs_10m_g[0]:.4f}", f"{att[0] - attVx_10m_g[0]:.4f}", f"{att[0] - attVz_10m_g[0]:.4f}")
# print(f"{att[0] - attPs_10m_nsx[0]:.4f}", f"{att[0] - attVx_10m_nsx[0]:.4f}", f"{att[0] - attVz_10m_nsx[0]:.4f}")

