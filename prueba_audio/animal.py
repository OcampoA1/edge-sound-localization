import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees
import threading

# === CONFIGURACIÓN ===
DURATION = 2  # segundos
SAMPLE_RATE = 44100
MIC_DISTANCE = 0.2  # en metros
SPEED_OF_SOUND = 343  # m/s

# === INDICES REALES DE LOS MICRÓFONOS ===
# Izquierda = device 10, Derecha = device 0 (según tu setup físico)
DEVICE_INDEX_LEFT = 10
DEVICE_INDEX_RIGHT = 0

# === VARIABLES GLOBALES PARA THREADS ===
mic_left_data = None
mic_right_data = None

def record_mic_thread(index, side):
    global mic_left_data, mic_right_data
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=index
    )
    sd.wait()
    if side == 'left':
        mic_left_data = audio.flatten()
    else:
        mic_right_data = audio.flatten()

# === GRABAR EN PARALELO ===
print("🎙️ Grabando en paralelo...")
thread_left = threading.Thread(target=record_mic_thread, args=(DEVICE_INDEX_LEFT, 'left'))
thread_right = threading.Thread(target=record_mic_thread, args=(DEVICE_INDEX_RIGHT, 'right'))

thread_left.start()
thread_right.start()

thread_left.join()
thread_right.join()
print("✅ Grabación finalizada.")

# === NORMALIZAR SEÑALES ===
mic_left = mic_left_data / np.max(np.abs(mic_left_data))
mic_right = mic_right_data / np.max(np.abs(mic_right_data))

# === CALCULAR TDoA (Δt) ===
corr = correlate(mic_right, mic_left, mode='full')
lags = np.arange(-len(mic_left)+1, len(mic_right))
lag = lags[np.argmax(corr)]
delta_t = lag / SAMPLE_RATE

# === ÁNGULO DE LLEGADA ===
x = (SPEED_OF_SOUND * delta_t) / MIC_DISTANCE
print(f"Δt = {delta_t:.6f} s")
print(f"Valor crudo para asin: {x:.4f}")
x_clipped = np.clip(x, -1.0, 1.0)
angle = degrees(asin(x_clipped))
print(f"🧭 Ángulo estimado: {angle:.2f}°")

# === GRAFICAR SEÑALES ===
time_axis = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
plt.figure(figsize=(10, 5))
plt.plot(time_axis, mic_left, label="Micrófono Izquierdo (device 10)", alpha=0.7)
plt.plot(time_axis, mic_right, label="Micrófono Derecho (device 0)", alpha=0.7)
plt.title("Comparación de señales - Localización de sonido")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud normalizada")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
