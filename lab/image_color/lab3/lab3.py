import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Load dữ liệu từ file .mat
data = sio.loadmat("lab/image_color/lab3/colorMatchingFunction.mat")   # đổi tên file theo bạn
cmf = data['CMFs']   

# Tách dữ liệu
wavelength = cmf[:,0]
r = cmf[:,1]
g = cmf[:,2]
b = cmf[:,3]

# Vẽ figure
plt.figure(figsize=(10,6))

plt.plot(wavelength, r, 'r', label='r(λ)')
plt.plot(wavelength, g, 'g', label='g(λ)')
plt.plot(wavelength, b, 'b', label='b(λ)')

plt.xlabel("Wavelength λ (nm)", fontsize=12)
plt.ylabel("Tristimulus values", fontsize=12)
plt.title("RGB Color Matching Functions", fontsize=14)
plt.legend()
plt.grid(True)

# --- tìm cực đại ---
idx_r = np.argmax(r)
idx_g = np.argmax(g)
idx_b = np.argmax(b)

peak_r = (wavelength[idx_r], r[idx_r])
peak_g = (wavelength[idx_g], g[idx_g])
peak_b = (wavelength[idx_b], b[idx_b])

# --- vẽ đường thẳng & annotate ---
plt.axvline(peak_r[0], color='red', linestyle='-', alpha=0.6)
plt.text(peak_r[0], -0.4, f"{peak_r[0]:.1f} nm", color='red', ha='center')

plt.axvline(peak_g[0], color='green', linestyle='-', alpha=0.6)
plt.text(peak_g[0], -0.4, f"{peak_g[0]:.1f} nm", color='green', ha='center')

plt.axvline(peak_b[0], color='blue', linestyle='-', alpha=0.6)
plt.text(peak_b[0], -0.4, f"{peak_b[0]:.1f} nm", color='blue', ha='center')

plt.ylim(-0.5, 3.5)
plt.xlim(400, 750)

plt.show()
