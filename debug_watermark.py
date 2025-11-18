import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

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


def visualize_watermark_detection_improved(file_path, nperseg=2048):
    """
    Visualize the watermark detection process step by step
    """
    sample_rate, audio = load_audio(file_path)
    filename = os.path.basename(file_path)
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {filename}")
    print('='*80)
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    magnitude = np.abs(Zxx)
    
    # Find reference frequency (20 kHz)
    ref_freq_idx = np.argmin(np.abs(frequencies - 20000))
    actual_ref_freq = frequencies[ref_freq_idx]
    
    # Find first non-zero window
    ref_magnitudes = magnitude[ref_freq_idx, :]
    nonzero_windows = np.where(ref_magnitudes > 0)[0]
    
    if len(nonzero_windows) == 0:
        print("No non-zero magnitude at reference frequency!")
        return
    
    first_window = nonzero_windows[0]
    reference_magnitude = ref_magnitudes[first_window]
    
    print(f"Reference frequency: {actual_ref_freq:.1f} Hz")
    print(f"Reference magnitude: {reference_magnitude:.6f}")
    print(f"First non-zero window: {first_window} at time {times[first_window]:.4f}s")
    
    # Search range
    freq_mask = (frequencies >= 15000) & (frequencies <= 20000)
    search_freq_indices = np.where(freq_mask)[0]
    search_frequencies = frequencies[search_freq_indices]
    
    # Track detected frequencies over time
    detected_freqs = []
    detected_mags = []
    detected_times = []
    tolerance = reference_magnitude * 0.05  # 5% tolerance
    
    for window_idx in range(magnitude.shape[1]):
        window_mags = magnitude[search_freq_indices, window_idx]
        matches = np.abs(window_mags - reference_magnitude) < tolerance
        
        if np.any(matches):
            closest_idx = search_freq_indices[matches][np.argmin(np.abs(window_mags[matches] - reference_magnitude))]
            detected_freqs.append(frequencies[closest_idx])
            detected_mags.append(magnitude[closest_idx, window_idx])
            detected_times.append(times[window_idx])
    
    detected_freqs = np.array(detected_freqs)
    detected_mags = np.array(detected_mags)
    detected_times = np.array(detected_times)
    
    print(f"Detected {len(detected_freqs)} time windows with matching magnitude")
    print(f"Tolerance: ±{tolerance:.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    # Plot 1: Full spectrogram with detected frequencies overlay
    ax1 = axes[0]
    Sxx_db = 10 * np.log10(magnitude + 1e-10)
    im1 = ax1.pcolormesh(times, frequencies/1000, Sxx_db, shading='gouraud', cmap='viridis')
    ax1.plot(detected_times, np.array(detected_freqs)/1000, 'r.', markersize=2, alpha=0.5, label='Detected watermark')
    ax1.set_ylim(15, 22)
    ax1.set_ylabel('Frequency (kHz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title(f'{filename} - Spectrogram with Detected Watermark')
    ax1.axhline(y=20, color='white', linestyle='--', alpha=0.7, linewidth=1, label='20 kHz reference')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    # Plot 2: Detected frequency vs time
    ax2 = axes[1]
    ax2.plot(detected_times, detected_freqs, 'b-', linewidth=1)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Detected Watermark Frequency Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency variation (remove DC component)
    ax3 = axes[2]
    f_center = np.mean(detected_freqs)
    freq_variation = detected_freqs - f_center
    ax3.plot(detected_times, freq_variation, 'g-', linewidth=1)
    ax3.set_ylabel('Frequency Variation (Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title(f'Frequency Variation (centered at {f_center:.1f} Hz)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 4: FFT of frequency variation to find watermark period
    ax4 = axes[3]
    if len(detected_times) > 10:
        # Perform FFT on the frequency variation
        dt = detected_times[1] - detected_times[0] if len(detected_times) > 1 else 0.023
        fft_variation = np.fft.fft(freq_variation)
        fft_freqs = np.fft.fftfreq(len(detected_times), d=dt)
        
        # Only positive frequencies
        positive_mask = fft_freqs > 0
        fft_magnitude = np.abs(fft_variation[positive_mask])
        fft_freqs_positive = fft_freqs[positive_mask]
        
        # Plot FFT
        ax4.plot(fft_freqs_positive, fft_magnitude, 'r-', linewidth=1)
        ax4.set_ylabel('FFT Magnitude')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_title('FFT of Frequency Variation (to find watermark period)')
        ax4.set_xlim(0, 20)  # Focus on low frequencies
        ax4.grid(True, alpha=0.3)
        
        # Find and mark peak
        peak_idx = np.argmax(fft_magnitude)
        watermark_freq_hz = fft_freqs_positive[peak_idx]
        ax4.axvline(x=watermark_freq_hz, color='blue', linestyle='--', linewidth=2, 
                   label=f'Peak at {watermark_freq_hz:.4f} Hz')
        ax4.legend()
        
        print(f"\nFFT Analysis:")
        print(f"  Time resolution: {dt:.4f} seconds")
        print(f"  Number of samples: {len(detected_times)}")
        print(f"  Detected watermark frequency: {watermark_freq_hz:.4f} Hz")
        print(f"  Period: {1/watermark_freq_hz:.4f} seconds")
        
        # Show top 5 peaks
        sorted_indices = np.argsort(fft_magnitude)[::-1][:5]
        print(f"\n  Top 5 FFT peaks:")
        for i, idx in enumerate(sorted_indices):
            print(f"    {i+1}. {fft_freqs_positive[idx]:.4f} Hz (magnitude: {fft_magnitude[idx]:.2f})")
    
    plt.tight_layout()
    
    # Save figure
    output_dir = "Exercise Inputs-20251113/Task 2/analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_debug.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved debug visualization to {output_file}")
    plt.close()
    
    return detected_freqs, detected_times


# Analyze files 0, 1, 2 from Group 1
task2_dir = "Exercise Inputs-20251113/Task 2"
for i in range(3):
    file_path = os.path.join(task2_dir, f"{i}_watermarked.wav")
    visualize_watermark_detection_improved(file_path)

print("\n" + "="*80)
print("Analysis complete! Check the 'analysis' folder for debug visualizations.")
print("="*80)
