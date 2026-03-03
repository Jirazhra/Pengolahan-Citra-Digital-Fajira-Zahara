import numpy as np
import matplotlib.pyplot as plt

def simulate_digitization(analog_signal, sampling_rate, quantization_levels):
    """
    analog_signal: fungsi kontinu f(x)
    sampling_rate: jumlah sampel
    quantization_levels: jumlah level kuantisasi
    """
    
    # ==========================
    # 1. SINYAL ANALOG (CONTINUOUS)
    # ==========================
    t_continuous = np.linspace(0, 1, 1000)  # waktu kontinu
    y_continuous = analog_signal(t_continuous)
    
    # ==========================
    # 2. SAMPLING
    # ==========================
    t_sampled = np.linspace(0, 1, sampling_rate)
    y_sampled = analog_signal(t_sampled)
    
    # ==========================
    # 3. QUANTIZATION
    # ==========================
    y_min = np.min(y_sampled)
    y_max = np.max(y_sampled)
    
    step_size = (y_max - y_min) / quantization_levels
    
    y_quantized = np.round((y_sampled - y_min) / step_size) * step_size + y_min
    
    # ==========================
    # 4. VISUALISASI
    # ==========================
    plt.figure(figsize=(12, 6))
    
    # Sinyal analog
    plt.plot(t_continuous, y_continuous, label="Analog Signal", linewidth=2)
    
    # Sampling
    plt.stem(t_sampled, y_sampled, linefmt='orange', 
             markerfmt='o', basefmt=" ", label="Sampled Signal")
    
    # Quantized
    plt.step(t_sampled, y_quantized, where='mid',
             label="Quantized Signal", linewidth=2)
    
    plt.title("Simulasi Digitalisasi (Sampling & Quantization)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        "sampling_rate": sampling_rate,
        "quantization_levels": quantization_levels,
        "step_size": step_size
    }


# ==================================
# CONTOH PENGGUNAAN
# ==================================

# Definisi sinyal analog (sinus)
def analog_signal(t):
    return np.sin(2 * np.pi * 5 * t)

result = simulate_digitization(
    analog_signal=analog_signal,
    sampling_rate=20,
    quantization_levels=8
)

print("Hasil Simulasi:")
print(result)
