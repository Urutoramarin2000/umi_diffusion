from scipy.interpolate import make_interp_spline  # 或用 PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt
No_mem =    np.array([0.22, 0.17, 0.052, 0.01, 0.005, 0.005, 0.004, 0.003, 0.004, 0.004, 0.003, 0.0035])
One_mem =   np.array([0.2, 0.07, 0.032, 0.009, 0.003, 0.002, 0.002, 0.001, 0.001, 0.0009, 0.0006, 0.0006])
Two_mem =   np.array([0.14, 0.02, 0.01, 0.005, 0.0007, 0.0003, 0.0001, 0.00004, 0.00002, 0.00004, 0.00001, 0.00001])
Three_mem = np.array([0.019, 0.01, 0.007, 0.002, 0.0009, 0.0003, 0.00004, 0.00002, 0.00003, 0.00001, 0.00002,0.00001])
Four_mem =  np.array([0.029, 0.01, 0.007, 0.007, 0.005, 0.002, 0.002, 0.001, 0.002, 0.001, 0.001, 0.002])
Epoch =     np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])



x_smooth = np.linspace(Epoch.min(), Epoch.max(), 24)  # 生成更多点
def smooth(x, y):
    spline = make_interp_spline(x, y)
    return spline(x_smooth)

plt.figure(figsize=(10, 6))
plt.plot(x_smooth, smooth(Epoch, No_mem), label="memory_size = 0")
plt.plot(x_smooth, smooth(Epoch, One_mem), label="memory_size = 1")
plt.plot(x_smooth, smooth(Epoch, Two_mem), label="memory_size = 2")
plt.plot(x_smooth, smooth(Epoch, Three_mem), label="memory_size = 3")
plt.plot(x_smooth, smooth(Epoch, Four_mem), label="memory_size = 4")

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.ylim(0, 0.02)
plt.title("Validation Loss Per Epoch Across Memory Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("smooth_memory_loss.png", dpi=300)
print("[✓] Saved smoothed plot to: smooth_memory_loss.png")

