import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Load dữ liệu từ file .mat
data = sio.loadmat("lab/image_color/lab2/coneFundamentals.mat")
coneFund = data['coneFundamentals']   # shape (4401,4)

# Tách dữ liệu
wavelength = coneFund[:,0]   # nm
L = coneFund[:,1]
M = coneFund[:,2]
S = coneFund[:,3]

# Vẽ figure
plt.figure(figsize=(10,6))

plt.plot(wavelength, L, 'r', label='L: $S_R(λ)$')
plt.plot(wavelength, M, 'g', label='M: $S_G(λ)$')
plt.plot(wavelength, S, 'b', label='S: $S_B(λ)$')

plt.xlabel("Wavelength λ (nm)", fontsize=12)
plt.ylabel("Sensitivity", fontsize=12)
plt.title("Absorption of light in the cones of the human retina", fontsize=14)
plt.legend()
plt.grid(True)

# --- tìm cực đại ---
idx_S = np.argmax(S)
idx_M = np.argmax(M)
idx_L = np.argmax(L)

peak_S = (wavelength[idx_S], S[idx_S])
peak_M = (wavelength[idx_M], M[idx_M])
peak_L = (wavelength[idx_L], L[idx_L])

# --- vẽ đường thẳng & chú thích ---
plt.axvline(peak_S[0], color='blue', linestyle='--', alpha=0.6)
plt.text(peak_S[0], 1.02, f"{peak_S[0]:.0f} nm", color='blue', ha='center')

plt.axvline(peak_M[0], color='green', linestyle='--', alpha=0.6)
plt.text(peak_M[0], 1.02, f"{peak_M[0]:.0f} nm", color='green', ha='center')

plt.axvline(peak_L[0], color='red', linestyle='--', alpha=0.6)
plt.text(peak_L[0], 1.02, f"{peak_L[0]:.0f} nm", color='red', ha='center')

plt.ylim(0,1.1)
plt.xlim(400,750)

plt.show()
