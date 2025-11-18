import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

def load_audio(file_path):
    """Load audio file and normalize to [-1, 1]"""
    sample_rate, data = wavfile.read(file_path)
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    
    if len(data.shape) > 1:
        data = data[:, 0]
    
    return sample_rate, data


def analyze_file(file_path, nperseg=2048):
    """Analyze a single file and return detected frequency pattern"""
    sample_rate, audio = load_audio(file_path)
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    magnitude = np.abs(Zxx)
    
    # Find reference at 20 kHz
    ref_freq_idx = np.argmin(np.abs(frequencies - 20000))
    ref_magnitudes = magnitude[ref_freq_idx, :]
    max_magnitude = np.max(ref_magnitudes)
    reference_magnitude = max_magnitude * 0.8
    
    # Search range
    freq_mask = (frequencies >= 15000) & (frequencies <= 20000)
    search_freq_indices = np.where(freq_mask)[0]
    
    # Track detected frequencies
    detected_freqs = []
    detected_times = []
    tolerance = reference_magnitude * 0.15
    
    for window_idx in range(magnitude.shape[1]):
        window_mags = magnitude[search_freq_indices, window_idx]
        diff_from_ref = np.abs(window_mags - reference_magnitude)
        
        if np.min(diff_from_ref) < tolerance:
            closest_idx = search_freq_indices[np.argmin(diff_from_ref)]
            detected_freqs.append(frequencies[closest_idx])
            detected_times.append(times[window_idx])
    
    return {
        'frequencies': frequencies,
        'times': times,
        'magnitude': magnitude,
        'detected_freqs': np.array(detected_freqs),
        'detected_times': np.array(detected_times),
        'reference_magnitude': reference_magnitude,
        'sample_rate': sample_rate
    }


# Analyze files 0, 1, 2
task2_dir = "Exercise Inputs-20251113/Task 2"
results = []
for i in range(3):
    file_path = os.path.join(task2_dir, f"{i}_watermarked.wav")
    print(f"Analyzing file {i}...")
    result = analyze_file(file_path)
    result['file_num'] = i
    results.append(result)

# Create comprehensive comparison
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(5, 3, hspace=0.3, wspace=0.3)

# Row 1: Spectrograms
for i, result in enumerate(results):
    ax = fig.add_subplot(gs[0, i])
    Sxx_db = 10 * np.log10(result['magnitude'] + 1e-10)
    im = ax.pcolormesh(result['times'], result['frequencies']/1000, Sxx_db, 
                       shading='gouraud', cmap='viridis', vmin=-100, vmax=-20)
    ax.plot(result['detected_times'], result['detected_freqs']/1000, 
            'r-', linewidth=2, alpha=0.7, label='Detected')
    ax.set_ylim(15, 22)
    ax.set_ylabel('Frequency (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'File {i} - Spectrogram')
    ax.axhline(y=20, color='white', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(im, ax=ax, label='dB')

# Row 2: Detected frequency over time
for i, result in enumerate(results):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(result['detected_times'], result['detected_freqs'], 'b-', linewidth=1)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'File {i} - Detected Frequency vs Time')
    ax.grid(True, alpha=0.3)
    f_mean = np.mean(result['detected_freqs'])
    ax.axhline(y=f_mean, color='r', linestyle='--', alpha=0.5)
    ax.set_ylim(14500, 20500)

# Row 3: Frequency variation (centered)
for i, result in enumerate(results):
    ax = fig.add_subplot(gs[2, i])
    f_center = np.mean(result['detected_freqs'])
    freq_variation = result['detected_freqs'] - f_center
    ax.plot(result['detected_times'], freq_variation, 'g-', linewidth=1)
    ax.set_ylabel('Freq Variation (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'File {i} - Variation (center={f_center:.1f} Hz)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    f_amp = (np.max(result['detected_freqs']) - np.min(result['detected_freqs'])) / 2
    ax.set_ylim(-3000, 3000)
    ax.text(0.02, 0.98, f'Amplitude: ±{f_amp:.1f} Hz', 
            transform=ax.transAxes, va='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 4: FFT of frequency variation
for i, result in enumerate(results):
    ax = fig.add_subplot(gs[3, i])
    f_center = np.mean(result['detected_freqs'])
    freq_variation = result['detected_freqs'] - f_center
    
    if len(result['detected_times']) > 1:
        dt = result['detected_times'][1] - result['detected_times'][0]
        fft_variation = np.fft.fft(freq_variation)
        fft_freqs = np.fft.fftfreq(len(result['detected_times']), d=dt)
        
        positive_mask = fft_freqs > 0
        fft_magnitude = np.abs(fft_variation[positive_mask])
        fft_freqs_positive = fft_freqs[positive_mask]
        
        ax.plot(fft_freqs_positive, fft_magnitude, 'r-', linewidth=1)
        ax.set_ylabel('FFT Magnitude')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(f'File {i} - FFT of Variation')
        ax.set_xlim(0, 3)
        ax.grid(True, alpha=0.3)
        
        # Find peak
        peak_idx = np.argmax(fft_magnitude)
        watermark_freq = fft_freqs_positive[peak_idx]
        ax.axvline(x=watermark_freq, color='blue', linestyle='--', linewidth=2, 
                   label=f'{watermark_freq:.4f} Hz')
        ax.legend()

# Row 5: Side-by-side comparison
ax_compare = fig.add_subplot(gs[4, :])
colors = ['blue', 'green', 'red']
labels = ['File 0', 'File 1', 'File 2']

for i, result in enumerate(results):
    # Normalize time to percentage of total duration
    max_time = result['detected_times'][-1] if len(result['detected_times']) > 0 else 30
    normalized_times = result['detected_times'] / max_time * 100
    ax_compare.plot(normalized_times, result['detected_freqs'], 
                   color=colors[i], linewidth=1.5, alpha=0.7, label=labels[i])

ax_compare.set_ylabel('Frequency (Hz)')
ax_compare.set_xlabel('Time (%)')
ax_compare.set_title('Overlay Comparison (Files 0, 1, 2)')
ax_compare.grid(True, alpha=0.3)
ax_compare.legend()
ax_compare.set_ylim(14500, 20500)

plt.suptitle('Group 1 Watermark Comparison (Files 0, 1, 2)', fontsize=16, fontweight='bold')

# Save
output_dir = "Exercise Inputs-20251113/Task 2/analysis"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "group1_comparison.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved comparison to {output_file}")

# Print numerical comparison
print("\n" + "="*80)
print("NUMERICAL COMPARISON - GROUP 1")
print("="*80)

for i, result in enumerate(results):
    f_center = np.mean(result['detected_freqs'])
    f_min = np.min(result['detected_freqs'])
    f_max = np.max(result['detected_freqs'])
    f_amp = (f_max - f_min) / 2
    freq_variation = result['detected_freqs'] - f_center
    
    if len(result['detected_times']) > 1:
        dt = result['detected_times'][1] - result['detected_times'][0]
        fft_variation = np.fft.fft(freq_variation)
        fft_freqs = np.fft.fftfreq(len(result['detected_times']), d=dt)
        positive_mask = fft_freqs > 0
        fft_magnitude = np.abs(fft_variation[positive_mask])
        fft_freqs_positive = fft_freqs[positive_mask]
        peak_idx = np.argmax(fft_magnitude)
        watermark_freq = fft_freqs_positive[peak_idx]
        period = 1 / watermark_freq
        
        # Show top 3 peaks
        sorted_indices = np.argsort(fft_magnitude)[::-1][:3]
        
        print(f"\nFile {i}:")
        print(f"  Center frequency: {f_center:.1f} Hz")
        print(f"  Frequency range: {f_min:.1f} - {f_max:.1f} Hz")
        print(f"  Amplitude: ±{f_amp:.1f} Hz")
        print(f"  Number of detections: {len(result['detected_freqs'])}")
        print(f"  Time resolution: {dt:.4f} seconds")
        print(f"  Primary watermark frequency: {watermark_freq:.4f} Hz ({period:.4f} s period)")
        print(f"  Top 3 FFT peaks:")
        for j, idx in enumerate(sorted_indices):
            freq = fft_freqs_positive[idx]
            mag = fft_magnitude[idx]
            print(f"    {j+1}. {freq:.4f} Hz (magnitude: {mag:.1f})")

print("\n" + "="*80)
plt.show()
