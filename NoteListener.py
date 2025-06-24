import pyaudio
import numpy as np
import aubio

# Parameters for audio stream
CHUNK = 4096  # Increased chunk size for better frequency resolution
FORMAT = pyaudio.paInt16  # Data type format (16-bit audio)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (samples per second)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream with the above settings
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Aubio setup for pitch detection
win_s = 8192  # Default FFT window size
hop_s = CHUNK  # Hop size should be equal to CHUNK
pitch_o = aubio.pitch("default", win_s, hop_s, RATE)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)  # Adjust sensitivity to ignore silence

# Define frequencies for standard notes
notes = {
    "E2": 82.41, "F2": 87.31, "F#2": 92.50, "G2": 98.00,
    "G#2": 103.83, "A2": 110.00, "A#2": 116.54, "B2": 123.47,
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56,
    "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00,
    "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00,
    "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25,
    "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99,
    "G#5": 830.61, "A5": 880.00, "A#5": 932.33, "B5": 987.77
}


# Function to map frequency to the closest note
def freq_to_note(freq):
    if freq == 0:  # No valid frequency detected
        return None
    min_diff = float("inf")
    closest_note = None
    for note, note_freq in notes.items():
        diff = abs(freq - note_freq)
        if diff < min_diff:
            min_diff = diff
            closest_note = note
    return closest_note


# Dynamically adjust the FFT window size based on the detected pitch
def adjust_window(freq):
    if freq < 100:  # Low frequencies (bass notes)
        pitch_o.set_tolerance(0.6)  # Allow larger tolerance for lower frequencies
        return 8192  # Large window size
    elif freq < 300:  # Mid-range frequencies (lower guitar strings)
        pitch_o.set_tolerance(0.5)
        return 4096  # Medium window size
    else:  # High frequencies (upper strings)
        pitch_o.set_tolerance(0.3)  # Higher precision for upper notes
        return 2048  # Smaller window size for higher notes


# Smoothing: Keep track of the last N pitch readings
smooth_pitch = []
SAMPLE_WINDOW = 5  # Increased sample window for better averaging

try:
    print("Recording with Dynamic Pitch Detection...")
    while True:
        # Read audio data in chunks
        data = stream.read(CHUNK)

        # Convert data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Aubio expects float32
        pitch = pitch_o(audio_data.astype(np.float32))[0]

        # Dynamically adjust FFT window based on the detected frequency
        win_s = adjust_window(pitch)

        # Smoothing by averaging over the last SAMPLE_WINDOW readings
        smooth_pitch.append(pitch)
        if len(smooth_pitch) > SAMPLE_WINDOW:
            smooth_pitch.pop(0)

        avg_pitch = sum(smooth_pitch) / len(smooth_pitch)

        # Convert the detected pitch to a musical note
        note = freq_to_note(avg_pitch)
        if note:
            print(f"Detected note: {note} (Frequency: {avg_pitch:.2f} Hz)")
        else:
            print("No valid pitch detected.")

except KeyboardInterrupt:
    print("Stopping recording")

finally:
    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
