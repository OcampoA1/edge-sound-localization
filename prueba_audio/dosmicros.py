import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees

# === CONFIGURACIÓN ===
DURATION = 5  # segundos
SAMPLE_RATE = 44100
MIC_DISTANCE = 0.2  # distancia entre micrófonos en metros (ajusta según lo real)
DEVICE_INDEX_1 = 0  # Micrófono izquierdo
DEVICE_INDEX_2 = 10  # Micrófono derecho
SPEED_OF_SOUND = 343  # m/s

# === FUNCIÓN PARA GRABAR UN MICRÓFONO ===
def record_mic(index):
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=index
    )
    sd.wait()
    return audio.flatten()

# === GRABAR LOS DOS MICRÓFONOS (secuencial por ahora) ===
print("🎙️ Grabando micrófono 1...")
mic1 = record_mic(DEVICE_INDEX_1)
print("🎙️ Grabando micrófono 2...")
mic2 = record_mic(DEVICE_INDEX_2)

# === NORMALIZAR ===
mic1 = mic1 / np.max(np.abs(mic1))
mic2 = mic2 / np.max(np.abs(mic2))

# === CALCULAR LA DIFERENCIA DE TIEMPO (TDoA) CON CORRELACIÓN CRUZADA ===
corr = correlate(mic2, mic1, mode='full')
lags = np.arange(-len(mic1)+1, len(mic2))
lag = lags[np.argmax(corr)]
delta_t = lag / SAMPLE_RATE

# === CALCULAR ÁNGULO (θ) ===
try:
    angle = degrees(asin((SPEED_OF_SOUND * delta_t) / MIC_DISTANCE))
except ValueError:
    angle = None

print(f"Δt = {delta_t:.6f} s")
if angle is not None:
    print(f"🧭 Ángulo estimado: {angle:.2f}°")
else:
    print("⚠️ No se puede calcular el ángulo (asin fuera de rango)")

# === GRAFICAR LAS DOS SEÑALES PARA COMPARAR ===
time_axis = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))

plt.figure(figsize=(10, 5))
plt.plot(time_axis, mic1, label="Micrófono 1 (izquierda)", alpha=0.7)
plt.plot(time_axis, mic2, label="Micrófono 2 (derecha)", alpha=0.7)
plt.title("Forma de onda de ambos micrófonos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
