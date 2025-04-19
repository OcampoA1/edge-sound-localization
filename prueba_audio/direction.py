import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees
import threading

# === CONFIGURACIÓN GENERAL ===
DURATION = 2  # segundos
SAMPLE_RATE = 44100
MIC_DISTANCE = 0.2  # metros
SPEED_OF_SOUND = 343  # m/s

# === CONFIGURACIÓN DE MICRÓFONOS ===
DEVICE_INDEX_LEFT = 10  # Micrófono a la izquierda físicamente
DEVICE_INDEX_RIGHT = 0  # Micrófono a la derecha físicamente

# === VARIABLES PARA THREADS ===
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

# === GRABACIÓN SIMULTÁNEA ===
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

# === CALCULAR Δt Y ÁNGULO θ ===
corr = correlate(mic_right, mic_left, mode='full')
lags = np.arange(-len(mic_left)+1, len(mic_right))
lag = lags[np.argmax(corr)]
delta_t = lag / SAMPLE_RATE

x = (SPEED_OF_SOUND * delta_t) / MIC_DISTANCE
print(f"Δt = {delta_t:.6f} s")
print(f"Valor crudo para asin: {x:.4f}")
x_clipped = np.clip(x, -1.0, 1.0)
angle = degrees(asin(x_clipped))
print(f"🧭 Ángulo estimado: {angle:.2f}°")

# === GRAFICAR SEÑALES DE AUDIO ===
plt.figure("Forma de onda")
time_axis = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
plt.plot(time_axis, mic_left, label="Micrófono Izquierdo (device 10)", alpha=0.7)
plt.plot(time_axis, mic_right, label="Micrófono Derecho (device 0)", alpha=0.7)
plt.title("Forma de onda de ambos micrófonos")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === RADAR VISUAL ===
plt.figure("Radar de dirección")
ax = plt.subplot(111, polar=True)
ax.set_theta_zero_location("N")  # 0° hacia arriba
ax.set_theta_direction(-1)       # sentido horario

theta_radians = np.radians(angle)
ax.plot([theta_radians], [1], marker='o', markersize=10, color='red', label=f"{angle:.1f}°")

ax.set_rmax(1.2)
ax.set_rticks([])  # sin círculos interiores
ax.set_thetagrids(range(-90, 91, 30))
ax.set_title("Dirección estimada del sonido", va='bottom')
ax.legend(loc='upper right')
ax.grid(True)

# === MOSTRAR TODO ===
plt.show()
