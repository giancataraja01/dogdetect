import sounddevice as sd
import scipy.io.wavfile as wav

# Path to your 20kHz WAV file
filename = '12000.wav'

# Load WAV file
sample_rate, data = wav.read(filename)

# Check info
print(f"Sample Rate: {sample_rate} Hz")
print(f"Data Type: {data.dtype}")
print(f"Duration: {len(data) / sample_rate:.2f} seconds")

# Play audio
print("Playing...")
sd.play(data, sample_rate)
sd.wait()
print("Playback finished.")
