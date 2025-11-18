import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

def load_audio(file_path):
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

def track_peak_frequency(file_path, nperseg=2048):
    """Simply track the PEAK frequency in 15-20kHz range for each time window"""
    sample_rate, audio = load_audio(file_path)
    
    frequencies, times, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    magnitude = np.abs(Zxx)
    
    # Define search range
    freq_mask = (frequencies >= 15000) & (frequencies <= 20000)
    search_freq_indices = np.where(freq_mask)[0]
    search_frequencies = frequencies[search_freq_indices]
    
    # For each time window, find the PEAK (maximum) frequency
    detected_freqs = []
    detected_mags = []
    
    for window_idx in range(magnitude.shape[1]):
        window_mags = magnitude[search_freq_indices, window_idx]
        max_idx = np.argmax(window_mags)
        detected_freqs.append(search_frequencies[max_idx])
        detected_mags.append(window_mags[max_idx])
    
    detected_freqs = np.array(detected_freqs)
    detected_mags = np.array(detected_mags)
    
    return times, detected_freqs, detected_mags

# Analyze files
task2_dir = "Exercise Inputs-20251113/Task 2"
results = []

print("="*80)
print("PEAK FREQUENCY TRACKING (15-20 kHz range)")
print("="*80)

for i in range(3):
    file_path = os.path.join(task2_dir, f"{i}_watermarked.wav")
    times, freqs, mags = track_peak_frequency(file_path)
    
    f_center = np.mean(freqs)
    f_amp = (np.max(freqs) - np.min(freqs)) / 2
    freq_variation = freqs - f_center
    
    # FFT analysis
    dt = times[1] - times[0]
    fft_variation = np.fft.fft(freq_variation)
    fft_freqs = np.fft.fftfreq(len(times), d=dt)
    positive_mask = fft_freqs > 0
    fft_magnitude = np.abs(fft_variation[positive_mask])
    fft_freqs_positive = fft_freqs[positive_mask]
    peak_idx = np.argmax(fft_magnitude)
    watermark_freq = fft_freqs_positive[peak_idx]
    period = 1 / watermark_freq
    
    results.append({
        'file': i,
        'times': times,
        'freqs': freqs,
        'mags': mags,
        'f_center': f_center,
        'f_amp': f_amp,
        'watermark_freq': watermark_freq,
        'period': period
    })
    
    print(f"\nFile {i}:")
    print(f"  Center frequency: {f_center:.1f} Hz")
    print(f"  Amplitude: ±{f_amp:.1f} Hz")
    print(f"  Watermark frequency: {watermark_freq:.4f} Hz")
    print(f"  Period: {period:.4f} seconds")
    print(f"  Number of cycles in 30s: {30 * watermark_freq:.2f}")

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# Plot 1: All three overlaid (raw frequencies)
ax = axes[0]
colors = ['blue', 'green', 'red']
for i, result in enumerate(results):
    ax.plot(result['times'], result['freqs'], color=colors[i], linewidth=1, alpha=0.7, label=f'File {i}')
ax.set_ylabel('Peak Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('Peak Frequency Over Time (15-20 kHz range) - Overlay')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(14500, 20500)

# Plot 2: All three normalized to percentage
ax = axes[1]
for i, result in enumerate(results):
    # Normalize time to 0-100%
    norm_time = result['times'] / result['times'][-1] * 100
    ax.plot(norm_time, result['freqs'], color=colors[i], linewidth=1, alpha=0.7, label=f'File {i}')
ax.set_ylabel('Peak Frequency (Hz)')
ax.set_xlabel('Time (%)')
ax.set_title('Peak Frequency - Normalized Time Scale')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(14500, 20500)

# Plot 3: Frequency variations (centered)
ax = axes[2]
for i, result in enumerate(results):
    freq_variation = result['freqs'] - result['f_center']
    ax.plot(result['times'], freq_variation, color=colors[i], linewidth=1, alpha=0.7, label=f'File {i}')
ax.set_ylabel('Frequency Variation (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('Frequency Variation (centered)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# Plot 4: First 10 seconds only (detailed view)
ax = axes[3]
for i, result in enumerate(results):
    mask = result['times'] <= 10
    ax.plot(result['times'][mask], result['freqs'][mask], color=colors[i], linewidth=2, alpha=0.7, label=f'File {i}', marker='o', markersize=3)
ax.set_ylabel('Peak Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('Peak Frequency - First 10 Seconds (Detailed)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(14500, 20500)

plt.tight_layout()

output_dir = "Exercise Inputs-20251113/Task 2/analysis"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "group1_peak_tracking.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to {output_file}")

print("\n" + "="*80)
plt.show()
