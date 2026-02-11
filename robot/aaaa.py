import matplotlib.pyplot as plt
import numpy as np

# Data from the pre-lab table (Volume NaOH added (mL), pH)
V = np.array([0.00, 10.00, 20.00, 30.00, 40.00, 49.00, 49.90, 50.00, 50.10, 51.00,
              60.00, 70.00, 80.00, 90.00, 100.00])
pH = np.array([2.87, 4.14, 4.56, 4.92, 5.34, 6.44, 7.45, 8.72, 10.00, 11.00,
               11.96, 12.22, 12.36, 12.46, 12.52])

Ve = 50.00            # equivalence point from the steep jump in the data
Vhalf = Ve / 2.0      # half-equivalence point

# Interpolate pH at half-equivalence (since 25.00 mL isn't an explicit data point)
pH_half = np.interp(Vhalf, V, pH)
pH_eq = np.interp(Ve, V, pH)

plt.figure(figsize=(8,5))
plt.plot(V, pH, marker='o')
plt.xlabel("Volume of 0.1000 M NaOH added (mL)")
plt.ylabel("pH")
plt.title("pH Titration Curve: Unknown Monoprotic Acid (50.00 mL sample; 0.3000 g)\n"
          "Titrated with 0.1000 M NaOH  |  Cole Herber")
plt.grid(True)

plt.annotate(f"Equivalence point\nVe ≈ {Ve:.2f} mL, pH ≈ {pH_eq:.2f}",
             xy=(Ve, pH_eq), xytext=(Ve+8, pH_eq-2.0),
             arrowprops=dict(arrowstyle="->"))

plt.annotate(f"Half-equivalence point\nVe/2 = {Vhalf:.2f} mL, pH ≈ {pH_half:.2f} (= pKa)",
             xy=(Vhalf, pH_half), xytext=(Vhalf+10, pH_half-1.5),
             arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig("titration_curve.png", dpi=200)
plt.show()