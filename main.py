import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os
from pathlib import Path


def load_audio(file_path):
    """Load audio file and normalize to [-1, 1]"""
    sample_rate, data = wavfile.read(file_path)
    
    # Convert to float and normalize
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    
    # Handle stereo by taking first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    
    return sample_rate, data


def add_watermark(audio, sample_rate, watermark_freq_start=20000, freq_spacing=500, watermark_width=50, amplitude_add=1000.0):
    """
    Add inaudible watermark to audio in frequency domain.
    Adds amplitude to specific frequencies while preserving phase.
    
    Args:
        audio: normalized audio signal
        sample_rate: sampling rate in Hz
        watermark_freq_start: start frequency for watermark (default 20 kHz)
        freq_spacing: spacing between watermark frequencies in Hz (default 500 Hz)
        watermark_width: width of each watermark peak in Hz (default 50 Hz)
        amplitude_add: amplitude to add to watermark frequencies (default 1000.0)
    
    Returns:
        watermarked audio signal
    """
    N = len(audio)
    
    # Compute FFT
    fft_data = np.fft.fft(audio)
    freqs = np.fft.fftfreq(N, d=1/sample_rate)
    
    # Get positive frequencies only
    positive_freq_mask = freqs > 0
    positive_freqs = freqs[positive_freq_mask]
    positive_indices = np.where(positive_freq_mask)[0]
    
    # Find which frequencies should get watermark
    freq_offset = positive_freqs - watermark_freq_start
    is_watermark = (freq_offset >= 0) & ((freq_offset % freq_spacing) < watermark_width)
    
    # Get indices of watermark frequencies
    watermark_indices = positive_indices[is_watermark]
    
    # Add amplitude while preserving phase (vectorized)
    magnitudes = np.abs(fft_data[watermark_indices])
    phases = np.angle(fft_data[watermark_indices])
    new_magnitudes = magnitudes + amplitude_add
    fft_data[watermark_indices] = new_magnitudes * np.exp(1j * phases)
    
    # Preserve conjugate symmetry for real signal (vectorized)
    negative_indices = N - watermark_indices
    fft_data[negative_indices] = np.conj(fft_data[watermark_indices])
    
    # Inverse FFT to get watermarked signal
    watermarked = np.fft.ifft(fft_data).real
    
    return watermarked


def add_bad_watermark(audio, sample_rate, watermark_freq_start=100, freq_spacing=500, watermark_width=50, amplitude_add=1000.0):
    """
    Add a BAD watermark that affects audible frequencies (DEMONSTRATION ONLY).
    This modifies frequencies starting from low frequencies, making it audible and easily detectable.
    
    Args:
        audio: normalized audio signal
        sample_rate: sampling rate in Hz
        watermark_freq_start: start frequency for watermark (default 100 Hz - AUDIBLE!)
        freq_spacing: spacing between watermark frequencies in Hz (default 500 Hz)
        watermark_width: width of each watermark peak in Hz (default 50 Hz)
        amplitude_add: amplitude to add to watermark frequencies (default 1000.0)
    
    Returns:
        badly watermarked audio signal
    """
    N = len(audio)
    
    # Compute FFT
    fft_data = np.fft.fft(audio)
    freqs = np.fft.fftfreq(N, d=1/sample_rate)
    
    # Get positive frequencies only (skip DC at index 0)
    positive_freq_mask = freqs > 0
    positive_freqs = freqs[positive_freq_mask]
    positive_indices = np.where(positive_freq_mask)[0]
    
    # Find which frequencies should get watermark
    freq_offset = positive_freqs - watermark_freq_start
    is_watermark = (freq_offset >= 0) & ((freq_offset % freq_spacing) < watermark_width)
    
    # Get indices of watermark frequencies
    watermark_indices = positive_indices[is_watermark]
    
    # Add amplitude while preserving phase (vectorized)
    magnitudes = np.abs(fft_data[watermark_indices])
    phases = np.angle(fft_data[watermark_indices])
    new_magnitudes = magnitudes + amplitude_add
    fft_data[watermark_indices] = new_magnitudes * np.exp(1j * phases)
    
    # Preserve conjugate symmetry for real signal (vectorized)
    negative_indices = N - watermark_indices
    fft_data[negative_indices] = np.conj(fft_data[watermark_indices])
    
    # Inverse FFT to get watermarked signal
    watermarked = np.fft.ifft(fft_data).real
    
    return watermarked


def plot_spectrogram(audio, sample_rate, title, save_path=None, max_freq=None):
    """
    Plot spectrogram (time-frequency representation) of audio signal.
    
    Args:
        audio: audio signal
        sample_rate: sampling rate in Hz
        title: plot title
        save_path: path to save the plot (optional)
        max_freq: maximum frequency to display (default: Nyquist/2)
    """
    plt.figure(figsize=(14, 8))
    
    # Use shorter window for better time resolution
    nperseg = 2048
    noverlap = nperseg // 2
    
    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        audio, 
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='spectrum'
    )
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Plot
    plt.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.title(title)
    
    # Set frequency limit
    if max_freq is None:
        max_freq = sample_rate / 2
    plt.ylim(0, max_freq)
    
    # Add colorbar
    plt.colorbar(label='Magnitude (dB)')
    
    # Add vertical lines every second
    max_time = times[-1]
    for t in range(1, int(max_time) + 1):
        plt.axvline(x=t, color='white', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add horizontal line at 20 kHz if visible
    if max_freq >= 20000:
        plt.axhline(y=20000, color='r', linestyle='--', alpha=0.7, linewidth=2, label='20 kHz (watermark start)')
        plt.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectrogram to {save_path}")
    
    plt.close()


def save_audio(file_path, sample_rate, data):
    """Save audio to file, converting from [-1, 1] to int16"""
    data_clipped = np.clip(data, -1.0, 1.0)
    data_int16 = (data_clipped * 32767).astype(np.int16)
    wavfile.write(file_path, sample_rate, data_int16)
    print(f"Saved audio to {file_path}")


def main():
    print("Audio Watermarking - Spectrogram Generation")
    print("=" * 60)
    
    # Task 1: Add watermark to task1.wav
    print("\nTASK 1: Adding Watermark to task1.wav")
    input_file = "Exercise Inputs-20251113/Task 1/task1.wav"
    output_file = "Exercise Inputs-20251113/Task 1/task1_watermarked.wav"
    spectrogram_file = "Exercise Inputs-20251113/Task 1/task1_spectrogram.png"
    
    sample_rate, audio = load_audio(input_file)
    print(f"Loaded {input_file}")
    print(f"Sample rate: {sample_rate} Hz, Duration: {len(audio) / sample_rate:.2f}s")
    
    # Add watermark
    watermarked = add_watermark(audio, sample_rate, watermark_freq_start=20000, 
                                freq_spacing=500, watermark_width=50, amplitude_add=1000.0)
    
    save_audio(output_file, sample_rate, watermarked)
    plot_spectrogram(watermarked, sample_rate, f"Task 1: Watermarked Audio", save_path=spectrogram_file)
    
    # Task 1b: Add BAD watermark (for comparison)
    print("\nTASK 1b: Adding BAD Watermark (affects audible frequencies)")
    bad_output_file = "Exercise Inputs-20251113/Task 1/task1_bad_watermarked.wav"
    bad_spectrogram_file = "Exercise Inputs-20251113/Task 1/task1_bad_spectrogram.png"
    
    # Add bad watermark starting from 100 Hz (audible range)
    bad_watermarked = add_bad_watermark(audio, sample_rate, watermark_freq_start=100, 
                                        freq_spacing=500, watermark_width=50, amplitude_add=1000.0)
    
    save_audio(bad_output_file, sample_rate, bad_watermarked)
    plot_spectrogram(bad_watermarked, sample_rate, f"Task 1: BAD Watermark (Audible)", save_path=bad_spectrogram_file)
    print("⚠️  Bad watermark affects audible frequencies - NOT recommended!")
    
    # Task 2: Generate spectrograms for all files
    print("\n" + "=" * 60)
    print("TASK 2: Generating Spectrograms")
    
    task2_dir = "Exercise Inputs-20251113/Task 2"
    spectrogram_dir = "Exercise Inputs-20251113/Task 2/spectrograms"
    os.makedirs(spectrogram_dir, exist_ok=True)
    
    wav_files = sorted([f for f in os.listdir(task2_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files)} files\n")
    
    for wav_file in wav_files:
        print(f"Processing {wav_file}...")
        input_path = os.path.join(task2_dir, wav_file)
        output_path = os.path.join(spectrogram_dir, f"{Path(wav_file).stem}_spectrogram.png")
        
        sample_rate, audio = load_audio(input_path)
        plot_spectrogram(audio, sample_rate, f"{wav_file}", save_path=output_path)
    
    print(f"\nAll spectrograms saved to {spectrogram_dir}/")
    
    # Task 3: Generate spectrograms for Task 3 files
    print("\n" + "=" * 60)
    print("TASK 3: Generating Spectrograms")
    
    task3_dir = "Exercise Inputs-20251113/Task 3"
    spectrogram3_dir = "Exercise Inputs-20251113/Task 3/spectrograms"
    os.makedirs(spectrogram3_dir, exist_ok=True)
    
    wav_files3 = sorted([f for f in os.listdir(task3_dir) if f.endswith('.wav')])
    print(f"Found {len(wav_files3)} files\n")
    
    for wav_file in wav_files3:
        print(f"Processing {wav_file}...")
        input_path = os.path.join(task3_dir, wav_file)
        output_path = os.path.join(spectrogram3_dir, f"{Path(wav_file).stem}_spectrogram.png")
        
        sample_rate, audio = load_audio(input_path)
        plot_spectrogram(audio, sample_rate, f"{wav_file}", save_path=output_path)
    
    print(f"\nAll Task 3 spectrograms saved to {spectrogram3_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
