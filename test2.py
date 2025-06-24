import numpy as np
import pyaudio
import scipy.fft
from scipy.signal import butter, filtfilt
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


class FrequencyDetector:
    def __init__(self, threshold=0.05, smoothing_window=5, buffer_size=2048, sample_rate=44100):
        self.threshold = threshold
        self.smoothing_window = smoothing_window
        self.frequency_buffer = deque(maxlen=smoothing_window)
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.stream = None  # Initialize stream variable for later use

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut):
        b, a = self.butter_bandpass(lowcut, highcut, self.sample_rate, order=6)
        return filtfilt(b, a, data)

    def _preprocess_audio(self, audio_data):
        audio_data = self.bandpass_filter(audio_data, 80, 1100)
        audio_data = audio_data - np.mean(audio_data)
        audio_data = np.where(np.abs(audio_data) < self.threshold, 0, audio_data)
        return audio_data

    def _estimate_frequency(self, audio_data):
        audio_data = self._preprocess_audio(audio_data)

        spectrum = np.abs(scipy.fft.fft(audio_data))
        freqs = scipy.fft.fftfreq(len(audio_data), 1 / self.sample_rate)

        positive_freqs = freqs[:len(freqs) // 2]
        positive_spectrum = spectrum[:len(spectrum) // 2]

        return positive_freqs, positive_spectrum

    def _find_closest_note(self, frequency):
        if frequency <= 0:
            return "N/A", 0

        A4_freq = 440.0
        semitones_from_A4 = 12 * np.log2(frequency / A4_freq)
        nearest_note_index = round(semitones_from_A4) % 12

        nearest_note_frequency = A4_freq * (2 ** (nearest_note_index / 12))
        return self.note_names[nearest_note_index], nearest_note_frequency

    def process_audio(self):
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True,
                             frames_per_buffer=self.buffer_size)

        try:
            while True:
                audio_data = np.frombuffer(self.stream.read(self.buffer_size), dtype=np.float32)
                frequency = self._estimate_frequency(audio_data)[0]

                smoothed_freq = np.mean(self.frequency_buffer)
                note, nearest_frequency = self._find_closest_note(smoothed_freq)

                print(f"Frequency: {smoothed_freq:.2f} Hz | Note: {note} | Nearest Frequency: {nearest_frequency:.2f} Hz")

                self.frequency_buffer.append(frequency)

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping audio processing...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            p.terminate()

    def run(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(0, self.sample_rate / 2)
        ax.set_ylim(0, 1)
        ax.set_title("Real-Time Frequency Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")

        freqs = np.linspace(0, self.sample_rate / 2, self.buffer_size // 2)

        def update_plot(frame):
            audio_data = np.frombuffer(self.stream.read(self.buffer_size), dtype=np.float32)

            positive_freqs, positive_spectrum = self._estimate_frequency(audio_data)

            line.set_data(positive_freqs, positive_spectrum)
            return [line]

        ani = FuncAnimation(fig, update_plot, blit=True, interval=50, cache_frame_data=False)

        plot_thread = threading.Thread(target=plt.show)
        plot_thread.start()

        self.process_audio()


# Create and run the detector
detector = FrequencyDetector()
detector.run()


