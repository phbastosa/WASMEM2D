import numpy as np
import matplotlib.pyplot as plt

nr = 901
nt = 4001
dt = 1e-3

SPS = np.loadtxt("../inputs/geometry/precision_test_SPS.txt", delimiter = ",", dtype = np.float32)
RPS = np.loadtxt("../inputs/geometry/precision_test_RPS.txt", delimiter = ",", dtype = np.float32)

prefix = "../outputs/seismograms/elastic"
suffix = "nStations901_nSamples4001_shot_1"

seismPs_r = np.fromfile(f"{prefix}_iso_Ps_{suffix}" + ".bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seismVx_r = np.fromfile(f"{prefix}_iso_Vx_{suffix}" + ".bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seismVz_r = np.fromfile(f"{prefix}_iso_Vz_{suffix}" + ".bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

seismPs_c = np.fromfile(f"{prefix}_ani_Ps_{suffix}" + ".bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seismVx_c = np.fromfile(f"{prefix}_ani_Vx_{suffix}" + ".bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")
seismVz_c = np.fromfile(f"{prefix}_ani_Vz_{suffix}" + ".bin", count = nt*nr, dtype = np.float32).reshape([nt,nr], order = "F")

trace = int(0.1*nr)

time = np.arange(nt)*dt

fig, ax = plt.subplots(ncols = 3, figsize = (12,6))

ax[0].imshow(seismPs_r, cmap = "Greys", aspect = "auto", extent = [0, nr-1, (nt-1)*dt, 0])
ax[0].plot(np.ones(nt)*trace, time)
ax[0].set_title("Pressure", fontsize = 15)
ax[0].set_ylabel("Time [s]", fontsize = 15)
ax[0].set_xlabel("Receiver Index [s]", fontsize = 15)

ax[1].imshow(seismVx_r, cmap = "Greys", aspect = "auto", extent = [0, nr-1, (nt-1)*dt, 0])
ax[1].plot(np.ones(nt)*trace, time)
ax[1].set_title("Vx", fontsize = 15)
ax[1].set_ylabel("Time [s]", fontsize = 15)
ax[1].set_xlabel("Receiver Index [s]", fontsize = 15)

ax[2].imshow(seismVz_r, cmap = "Greys", aspect = "auto", extent = [0, nr-1, (nt-1)*dt, 0])
ax[2].plot(np.ones(nt)*trace, time)
ax[2].set_title("Vz", fontsize = 15)
ax[2].set_ylabel("Time [s]", fontsize = 15)
ax[2].set_xlabel("Receiver Index [s]", fontsize = 15)

fig.tight_layout()
plt.show()


analytical_time = np.sqrt((SPS[0] - RPS[trace,0])**2 + (SPS[1] - RPS[trace,1])**2) / 1500

tracePs_r = seismPs_r[:,trace] #/ np.max(np.abs(seismPs_r[:,trace]))
traceVx_r = seismVx_r[:,trace] #/ np.max(np.abs(seismVx_r[:,trace]))
traceVz_r = seismVz_r[:,trace] #/ np.max(np.abs(seismVz_r[:,trace]))

tracePs_c = seismPs_c[:,trace] #/ np.max(np.abs(seismPs_c[:,trace]))
traceVx_c = seismVx_c[:,trace] #/ np.max(np.abs(seismVx_c[:,trace]))
traceVz_c = seismVz_c[:,trace] #/ np.max(np.abs(seismVz_c[:,trace]))

n = 101
amp = np.linspace(-1.5, 1.5, n)
att = np.zeros(n) + analytical_time

attPs_r = np.zeros(n) + np.argmax(np.abs(tracePs_r))*dt
attVx_r = np.zeros(n) + np.argmax(np.abs(traceVx_r))*dt
attVz_r = np.zeros(n) + np.argmax(np.abs(traceVz_r))*dt

attPs_c = np.zeros(n) + np.argmax(np.abs(tracePs_c))*dt
attVx_c = np.zeros(n) + np.argmax(np.abs(traceVx_c))*dt
attVz_c = np.zeros(n) + np.argmax(np.abs(traceVz_c))*dt

fig, ax = plt.subplots(ncols = 3, figsize = (8,9))

# ax[0].plot(amp, att, "--k", label = "Analytical")
# ax[0].plot(amp, attPs_r, "--", color = "blue", label = "SSG")
# ax[0].plot(amp, attPs_c, "--", color = "green", label = "RSG")
ax[0].plot(tracePs_r, time, color = "blue", label = "SSG")
ax[0].plot(tracePs_c, time, color = "green", label = "RSG")
ax[0].set_title("Pressure", fontsize = 15)
ax[0].set_ylabel("Time [s]", fontsize = 15)
ax[0].set_xlabel("Amplitude", fontsize = 15)
ax[0].set_ylim([2.5, 3.0])
ax[0].invert_yaxis()
ax[0].legend(loc = "upper right", fontsize = 12)

# ax[1].plot(amp, att, "--k", label = "Analytical")
# ax[1].plot(amp, attVx_r, "--", color = "blue", label = "SSG")
# ax[1].plot(amp, attVx_c, "--", color = "green", label = "RSG")
ax[1].plot(traceVx_r, time, color = "blue", label = "SSG")
ax[1].plot(traceVx_c, time, color = "green", label = "RSG")
ax[1].set_title("Vx", fontsize = 15)
ax[1].set_ylabel("Time [s]", fontsize = 15)
ax[1].set_xlabel("Amplitude", fontsize = 15)
ax[1].set_ylim([2.5, 3.0])
ax[1].invert_yaxis()
ax[1].legend(loc = "upper right", fontsize = 12)

# ax[2].plot(amp, att, "--k", label = "Analytical")
# ax[2].plot(amp, attVz_r, "--", color = "blue", label = "SSG")
# ax[2].plot(amp, attVz_c, "--", color = "green", label = "RSG")
ax[2].plot(traceVz_r, time, color = "blue", label = "SSG")
ax[2].plot(traceVz_c, time, color = "green", label = "RSG")
ax[2].set_title("Vz", fontsize = 15)
ax[2].set_ylabel("Time [s]", fontsize = 15)
ax[2].set_xlabel("Amplitude", fontsize = 15)
ax[2].set_ylim([2.5, 3.0])
ax[2].invert_yaxis()
ax[2].legend(loc = "upper right", fontsize = 12)

fig.tight_layout()
plt.show()

print(f"{att[0] - attPs_r[0]:.4f}", f"{att[0] - attVx_r[0]:.4f}", f"{att[0] - attVz_r[0]:.4f}")
print(f"{att[0] - attPs_c[0]:.4f}", f"{att[0] - attVx_c[0]:.4f}", f"{att[0] - attVz_c[0]:.4f}")

