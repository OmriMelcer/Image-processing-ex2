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


def detect_watermark_stft(audio, sample_rate, reference_freq=20000, search_freq_min=15000, search_freq_max=20000, nperseg=2048):
    """
    Detect time-varying watermark pattern using STFT - CORRECTED VERSION.
    
    The watermark is a sine wave in TIME where the frequency itself oscillates:
    f(t) = f_center + amplitude * sin(watermark_freq * t)
    
    Algorithm:
    1. Perform STFT to get time-frequency representation
    2. Track the peak (brightest) frequency in 15-20 kHz range at each time
    3. Find peaks in this trajectory (when it reaches >19.5 kHz)
    4. Count peaks to determine watermark frequency
    
    Args:
        audio: audio signal
        sample_rate: sampling rate in Hz
        reference_freq: not used in corrected version (kept for compatibility)
        search_freq_min: minimum frequency to search (default 15000 Hz)
        search_freq_max: maximum frequency to search (default 20000 Hz)
        nperseg: STFT window size (default 2048)
    
    Returns:
        dict with watermark analysis results
    """
    from scipy.signal import find_peaks
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    
    # Get magnitude spectrum
    magnitude = np.abs(Zxx)
    
    print(f"  STFT shape: {magnitude.shape[0]} frequencies x {magnitude.shape[1]} time windows")
    print(f"  Frequency resolution: {frequencies[1] - frequencies[0]:.2f} Hz")
    print(f"  Time resolution: {times[1] - times[0]:.4f} seconds")
    
    # Define search range for frequencies
    freq_mask = (frequencies >= search_freq_min) & (frequencies <= search_freq_max)
    search_freq_indices = np.where(freq_mask)[0]
    search_frequencies = frequencies[search_freq_indices]
    
    print(f"  Searching {len(search_frequencies)} frequencies from {search_frequencies[0]:.1f} to {search_frequencies[-1]:.1f} Hz")
    
    # Track the peak (brightest) frequency at each time window
    detected_freqs = []
    detected_times = []
    
    for window_idx in range(magnitude.shape[1]):
        window_mags = magnitude[search_freq_indices, window_idx]
        max_idx = np.argmax(window_mags)
        detected_freqs.append(search_frequencies[max_idx])
        detected_times.append(times[window_idx])
    
    detected_freqs = np.array(detected_freqs)
    detected_times = np.array(detected_times)
    
    print(f"  Tracked {len(detected_freqs)} time windows")
    
    # Analyze the frequency variation
    f_center = np.mean(detected_freqs)
    f_min = np.min(detected_freqs)
    f_max = np.max(detected_freqs)
    f_amplitude = (f_max - f_min) / 2
    
    print(f"  Center frequency: {f_center:.1f} Hz")
    print(f"  Frequency range: {f_min:.1f} - {f_max:.1f} Hz")
    print(f"  Frequency amplitude: ±{f_amplitude:.1f} Hz")
    
    # Find peaks in the frequency trajectory (when sine wave reaches high frequencies)
    # Use threshold of 19500 Hz which corresponds to ~11 peaks in 30 seconds
    peaks, _ = find_peaks(detected_freqs, height=19500, distance=30)
    
    print(f"  Number of peaks (>19500 Hz): {len(peaks)}")
    
    if len(peaks) > 1:
        # Calculate watermark frequency from peak spacing
        peak_times = detected_times[peaks]
        gaps = np.diff(peak_times)
        avg_gap = np.mean(gaps)
        watermark_freq_hz = 1.0 / avg_gap
        watermark_period_seconds = avg_gap
        
        print(f"  Average gap between peaks: {avg_gap:.4f} seconds")
        print(f"  Watermark frequency: {watermark_freq_hz:.4f} Hz")
        print(f"  Watermark period: {watermark_period_seconds:.4f} seconds")
        print(f"  Cycles in 30s: {30 * watermark_freq_hz:.2f}")
        
        return {
            'detected_freqs': detected_freqs,
            'detected_times': detected_times,
            'f_center': f_center,
            'f_amplitude': f_amplitude,
            'watermark_freq_hz': watermark_freq_hz,
            'watermark_period_seconds': watermark_period_seconds,
            'wavelength_samples': watermark_period_seconds * sample_rate,
            'num_detections': len(detected_freqs),
            'num_peaks': len(peaks),
            'peak_times': peak_times
        }
    else:
        print(f"  WARNING: Too few peaks to analyze pattern")
        return {
            'detected_freqs': detected_freqs,
            'detected_times': detected_times,
            'f_center': f_center,
            'f_amplitude': f_amplitude,
            'watermark_freq_hz': None,
            'num_detections': len(detected_freqs)
        }


def analyze_task2_watermarks():
    """
    Analyze watermarks in all Task 2 files and group results.
    Groups: 0-2, 3-5, 6-8
    """
    print("\n" + "=" * 80)
    print("WATERMARK DETECTION - TASK 2 (Peak Counting Method)")
    print("=" * 80)
    
    task2_dir = "Exercise Inputs-20251113/Task 2"
    wav_files = sorted([f for f in os.listdir(task2_dir) if f.endswith('_watermarked.wav') and not f.startswith('.')])
    
    # Store results by group
    groups = {
        'Group 1 (0-2)': [],
        'Group 2 (3-5)': [],
        'Group 3 (6-8)': []
    }
    
    results = []
    
    for wav_file in wav_files:
        file_path = os.path.join(task2_dir, wav_file)
        file_num = int(wav_file.split('_')[0])
        
        print(f"\n{'='*80}")
        print(f"File {file_num}: {wav_file}")
        print('='*80)
        
        sample_rate, audio = load_audio(file_path)
        print(f"Sample rate: {sample_rate} Hz, Duration: {len(audio)/sample_rate:.2f}s")
        
        # Detect watermark
        result = detect_watermark_stft(audio, sample_rate)
        
        if result:
            result['file'] = wav_file
            result['file_num'] = file_num
            results.append(result)
            
            # Assign to group
            if file_num <= 2:
                groups['Group 1 (0-2)'].append(result)
            elif file_num <= 5:
                groups['Group 2 (3-5)'].append(result)
            else:
                groups['Group 3 (6-8)'].append(result)
    
    # Print summary by group
    print("\n" + "=" * 80)
    print("SUMMARY BY GROUP")
    print("=" * 80)
    
    for group_name, group_results in groups.items():
        print(f"\n{group_name}:")
        print("-" * 80)
        
        valid_results = [r for r in group_results if r.get('watermark_freq_hz') is not None]
        
        if valid_results:
            # Calculate group statistics
            watermark_freqs = [r['watermark_freq_hz'] for r in valid_results]
            periods = [r['watermark_period_seconds'] for r in valid_results]
            f_centers = [r['f_center'] for r in valid_results]
            f_amplitudes = [r['f_amplitude'] for r in valid_results]
            
            print(f"Files: {[r['file_num'] for r in group_results]}")
            print(f"Watermark frequency: {np.mean(watermark_freqs):.4f} Hz (±{np.std(watermark_freqs):.4f})")
            print(f"Watermark period: {np.mean(periods):.4f} s (±{np.std(periods):.4f})")
            print(f"Center frequency: {np.mean(f_centers):.1f} Hz (±{np.std(f_centers):.1f})")
            print(f"Frequency amplitude: ±{np.mean(f_amplitudes):.1f} Hz (±{np.std(f_amplitudes):.1f})")
            
            # Show individual results
            print("\nIndividual files:")
            for r in group_results:
                if r.get('watermark_freq_hz') is not None:
                    print(f"  File {r['file_num']}: "
                          f"watermark_freq={r['watermark_freq_hz']:.4f} Hz, "
                          f"period={r['watermark_period_seconds']:.4f}s, "
                          f"f_center={r['f_center']:.1f} Hz, "
                          f"f_amp=±{r['f_amplitude']:.1f} Hz")
                else:
                    print(f"  File {r['file_num']}: No watermark detected")
        else:
            print("No valid watermarks detected in this group")
    
    print("\n" + "=" * 80)
    
    return results, groups


def detect_single_file_watermark(file_path, nperseg=2048):
    """
    Convenience function to detect watermark in a single file.
    
    Args:
        file_path: path to .wav file
        nperseg: STFT window size (default 2048)
    
    Returns:
        dict with watermark results
    """
    print(f"Analyzing: {file_path}")
    sample_rate, audio = load_audio(file_path)
    result = detect_watermark_stft(audio, sample_rate, nperseg=nperseg)
    return result


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
    
    # Watermark Detection for Task 2
    analyze_task2_watermarks()


if __name__ == "__main__":
    main()
