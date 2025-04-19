import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# === CONFIGURACIÓN DEL MICRÓFONO ===
DURATION = 10               # Duración de la grabación en segundos
SAMPLE_RATE = 44100        # Frecuencia de muestreo en Hz
DEVICE_INDEX = 0           # Índice del micrófono USB
CHANNELS = 1               # Mono (el micrófono ML1-TRWCF es omnidireccional mono)
DTYPE = 'int16'            # Formato de grabación

# === GRABACIÓN ===
print("🎙️ Grabando audio...")
frames = int(DURATION * SAMPLE_RATE)
recording = sd.rec(
    frames,
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=DTYPE,
    device=DEVICE_INDEX
)
sd.wait()
print("✅ Grabación finalizada.")

# === GUARDAR ARCHIVO WAV ===
output_path = "output_mic1.wav"
wav.write(output_path, SAMPLE_RATE, recording)
print(f"💾 Audio guardado en: {output_path}")

# === PROCESAMIENTO PARA VISUALIZACIÓN ===
# Convertimos a float32 para análisis matemático
recording_float = recording.astype(np.float32) / np.max(np.abs(recording))

# Calcular energía RMS (nivel general del audio)
rms = np.sqrt(np.mean(recording_float**2))
print(f"📊 RMS (energía de la señal): {rms:.4f}")

# === GRAFICAR FORMA DE ONDA ===
time_axis = np.linspace(0, DURATION, num=frames)

plt.figure(figsize=(10, 4))
plt.plot(time_axis, recording_float, linewidth=1)
plt.title("Forma de onda del micrófono ML1-TRWCF")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud normalizada")
plt.grid(True)
plt.tight_layout()
plt.show()
