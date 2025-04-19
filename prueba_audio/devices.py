import sounddevice as sd

print("Micrófonos disponibles:\n")
print(sd.query_devices())
