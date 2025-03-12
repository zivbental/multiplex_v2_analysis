import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# Raw data (from the image, organized for clarity)
data = {
    'Dop': [10, 1, 1, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 10, 10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 10],
    'Vh': [-80, -20, -20, -20, -20, -80, -80, -20, -20, -80, -80, -20, -20, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80, -80],
    'Response': [-175, -50, 0, -250, -280, -2070, -173, -217, -380, -1760, -1400, -100, -140, -1990, 0, -650, -720, -370, -380, -840, -380, -50, -275, 0, -70, -70, 0, -112]
}

df = pd.DataFrame(data)

# Separate data by holding potential
df_20 = df[df['Vh'] == -20]
df_80 = df[df['Vh'] == -80]

# Normalization factor for -20mV (average of max concentration)
norm_factor_20 = df_20[df_20['Dop'] == 1000]['Response'].mean()

# Normalization factor for -80mV (average of max concentration, last two values)
norm_factor_80 = df_80[df_80['Dop'] == 1000]['Response'].iloc[-2:].mean()

# Normalize the data (divide by the absolute value of the normalization factor)
df_20['Normalized_Response'] = df_20['Response'] / norm_factor_20
df_80['Normalized_Response'] = df_80['Response'] / norm_factor_80

# Calculate Mean and SEM (for the normalized responses)
grouped_20 = df_20.groupby('Dop')['Normalized_Response'].agg(['mean', 'sem']).reset_index()
grouped_80 = df_80.groupby('Dop')['Normalized_Response'].agg(['mean', 'sem']).reset_index()

# --- Sigmoid Function (Hill Coefficient = 1) ---
def sigmoid(dose, bottom, top, ec50):
    return bottom + (top - bottom) / (1 + (ec50 / dose))

# --- Fitting (-20mV) ---
p0_20 = [0, 1, 10]  # Guess: Bottom, Top, EC50 (Top guess is now 1)
params_20, covariance_20 = curve_fit(sigmoid, grouped_20['Dop'], grouped_20['mean'], p0=p0_20, sigma=grouped_20['sem'], absolute_sigma=True)
bottom_20, top_20, ec50_20 = params_20

# --- Fitting (-80mV) ---
p0_80 = [0, 1, 1]  # Guess for -80mV (Top guess is now 1)
params_80, covariance_80 = curve_fit(sigmoid, grouped_80['Dop'], grouped_80['mean'], p0=p0_80, sigma=grouped_80['sem'], absolute_sigma=True)
bottom_80, top_80, ec50_80 = params_80

# --- Generate points for the fitted curves ---
dose_fit = np.logspace(-4, 4, 500)
response_fit_20 = sigmoid(dose_fit, bottom_20, top_20, ec50_20)
response_fit_80 = sigmoid(dose_fit, bottom_80, top_80, ec50_80)

# --- Plotting ---
plt.figure(figsize=(8, 6))

# Plot -20mV data and fit
plt.errorbar(grouped_20['Dop'], grouped_20['mean'], yerr=grouped_20['sem'], fmt='o', color='blue', label='-20mV (Data)', capsize=4)
plt.plot(dose_fit, response_fit_20, color='blue', linestyle='-', label=f'-20mV (Fit, EC50={ec50_20:.2f} uM)')

# Plot -80mV data and fit
plt.errorbar(grouped_80['Dop'], grouped_80['mean'], yerr=grouped_80['sem'], fmt='s', color='red', label='-80mV (Data)', capsize=4)
plt.plot(dose_fit, response_fit_80, color='red', linestyle='-', label=f'-80mV (Fit, EC50={ec50_80:.2f} uM)')

plt.xscale('log')
plt.xlabel('Dopamine Concentration [uM]')
plt.ylabel('Normalized Response')  # Removed (%)
plt.title('Dose-Response Curves with Normalized Data and SEM')
plt.legend()
plt.grid(True)
plt.show()
# --- Print Results ---
print("--- -20mV Fit Results ---")
print(f"Bottom: {bottom_20:.4f}")
print(f"Top: {top_20:.4f}")
print(f"EC50: {ec50_20:.2f} uM")

print("\n--- -80mV Fit Results ---")
print(f"Bottom: {bottom_80:.4f}")
print(f"Top: {top_80:.4f}")
print(f"EC50: {ec50_80:.2f} uM")