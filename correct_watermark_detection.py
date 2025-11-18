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

def detect_watermark_correct(file_path, nperseg=2048):
    """
    CORRECT watermark detection:
    1. Find max magnitude at 20kHz → this is the anchor
    2. For each time window, find which frequency has this anchor magnitude
    3. Track the frequency over time
    """
    sample_rate, audio = load_audio(file_path)
    filename = os.path.basename(file_path)
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {filename}")
    print('='*80)
    
    # Compute STFT
    frequencies, times, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2)
    magnitude = np.abs(Zxx)
    
    print(f"STFT shape: {magnitude.shape[0]} frequencies x {magnitude.shape[1]} time windows")
    
    # Find index of 20 kHz
    ref_freq_idx = np.argmin(np.abs(frequencies - 20000))
    actual_ref_freq = frequencies[ref_freq_idx]
    
    # Get magnitude at 20kHz for all time windows
    ref_magnitudes = magnitude[ref_freq_idx, :]
    
    # Find MAXIMUM magnitude at 20kHz → this is our anchor
    anchor_magnitude = np.max(ref_magnitudes)
    anchor_window = np.argmax(ref_magnitudes)
    
    print(f"Reference frequency: {actual_ref_freq:.1f} Hz")
    print(f"Anchor magnitude: {anchor_magnitude:.8f}")
    print(f"Found at window {anchor_window} (time: {times[anchor_window]:.2f}s)")
    
    # Define search range
    freq_mask = (frequencies >= 15000) & (frequencies <= 20000)
    search_freq_indices = np.where(freq_mask)[0]
    search_frequencies = frequencies[search_freq_indices]
    
    print(f"Searching {len(search_frequencies)} frequencies: {search_frequencies[0]:.1f} - {search_frequencies[-1]:.1f} Hz")
    
    # For each time window, find the frequency that has magnitude closest to anchor
    detected_freqs = []
    detected_times = []
    detected_mags = []
    tolerance = anchor_magnitude * 0.2  # 20% tolerance
    
    for window_idx in range(magnitude.shape[1]):
        # Get magnitudes in search range for this time window
        window_mags = magnitude[search_freq_indices, window_idx]
        
        # Find frequency with magnitude closest to anchor
        diff_from_anchor = np.abs(window_mags - anchor_magnitude)
        min_diff = np.min(diff_from_anchor)
        
        # Only include if within tolerance
        if min_diff < tolerance:
            best_idx = search_freq_indices[np.argmin(diff_from_anchor)]
            detected_freqs.append(frequencies[best_idx])
            detected_times.append(times[window_idx])
            detected_mags.append(magnitude[best_idx, window_idx])
    
    detected_freqs = np.array(detected_freqs)
    detected_times = np.array(detected_times)
    detected_mags = np.array(detected_mags)
    
    print(f"Detected {len(detected_freqs)} time windows with anchor magnitude (tolerance: ±{tolerance:.8f})")
    
    if len(detected_freqs) < 10:
        print("WARNING: Too few detections!")
        return None
    
    # Analyze frequency variation
    f_center = np.mean(detected_freqs)
    f_min = np.min(detected_freqs)
    f_max = np.max(detected_freqs)
    f_amp = (f_max - f_min) / 2
    freq_variation = detected_freqs - f_center
    
    print(f"Center frequency: {f_center:.1f} Hz")
    print(f"Frequency range: {f_min:.1f} - {f_max:.1f} Hz")
    print(f"Amplitude: ±{f_amp:.1f} Hz")
    
    # FFT to find watermark frequency
    if len(detected_times) > 1:
        dt = detected_times[1] - detected_times[0]
        fft_variation = np.fft.fft(freq_variation)
        fft_freqs = np.fft.fftfreq(len(detected_times), d=dt)
        
        positive_mask = fft_freqs > 0
        fft_magnitude = np.abs(fft_variation[positive_mask])
        fft_freqs_positive = fft_freqs[positive_mask]
        
        peak_idx = np.argmax(fft_magnitude)
        watermark_freq = fft_freqs_positive[peak_idx]
        period = 1 / watermark_freq
        
        print(f"Watermark frequency: {watermark_freq:.4f} Hz")
        print(f"Period: {period:.4f} seconds")
        print(f"Cycles in 30s: {30 * watermark_freq:.2f}")
        
        return {
            'filename': filename,
            'times': times,
            'magnitude': magnitude,
            'frequencies': frequencies,
            'anchor_magnitude': anchor_magnitude,
            'detected_freqs': detected_freqs,
            'detected_times': detected_times,
            'detected_mags': detected_mags,
            'f_center': f_center,
            'f_amp': f_amp,
            'watermark_freq': watermark_freq,
            'period': period
        }
    
    return None


# Analyze files 0, 1, 2
task2_dir = "Exercise Inputs-20251113/Task 2"
results = []

for i in range(3):
    file_path = os.path.join(task2_dir, f"{i}_watermarked.wav")
    result = detect_watermark_correct(file_path)
    if result:
        result['file_num'] = i
        results.append(result)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for result in results:
    print(f"\nFile {result['file_num']}:")
    print(f"  Watermark frequency: {result['watermark_freq']:.4f} Hz ({result['period']:.4f}s period)")
    print(f"  Center frequency: {result['f_center']:.1f} Hz")
    print(f"  Amplitude: ±{result['f_amp']:.1f} Hz")
    print(f"  Detections: {len(result['detected_freqs'])}")

# Create comparison visualization
fig, axes = plt.subplots(5, 1, figsize=(18, 16))

colors = ['blue', 'green', 'red']
labels = ['File 0', 'File 1', 'File 2']

# Plot 1: Spectrograms side by side with detected overlay
for i, result in enumerate(results):
    ax = fig.add_subplot(3, 3, i+1)
    Sxx_db = 10 * np.log10(result['magnitude'] + 1e-10)
    im = ax.pcolormesh(result['times'], result['frequencies']/1000, Sxx_db, 
                       shading='gouraud', cmap='viridis', vmin=-100, vmax=-20)
    ax.plot(result['detected_times'], result['detected_freqs']/1000, 
            'r-', linewidth=2, alpha=0.8)
    ax.set_ylim(15, 21)
    ax.set_ylabel('Freq (kHz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'File {result["file_num"]}')
    ax.axhline(y=20, color='white', linestyle='--', alpha=0.5)

# Plot 2: Detected frequencies overlay (absolute)
ax = axes[0]
for i, result in enumerate(results):
    ax.plot(result['detected_times'], result['detected_freqs'], 
            color=colors[i], linewidth=1.5, alpha=0.7, label=labels[i])
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('Detected Frequencies - Overlay (Absolute)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(14500, 20500)

# Plot 3: Detected frequencies (normalized time)
ax = axes[1]
for i, result in enumerate(results):
    norm_times = result['detected_times'] / result['detected_times'][-1] * 100
    ax.plot(norm_times, result['detected_freqs'], 
            color=colors[i], linewidth=1.5, alpha=0.7, label=labels[i])
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (%)')
ax.set_title('Detected Frequencies - Normalized Time')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(14500, 20500)

# Plot 4: Frequency variations (centered)
ax = axes[2]
for i, result in enumerate(results):
    freq_var = result['detected_freqs'] - result['f_center']
    ax.plot(result['detected_times'], freq_var, 
            color=colors[i], linewidth=1.5, alpha=0.7, label=labels[i])
ax.set_ylabel('Frequency Variation (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('Frequency Variation (Centered)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.set_ylim(-3000, 3000)

# Plot 5: First 5 seconds detail
ax = axes[3]
for i, result in enumerate(results):
    mask = result['detected_times'] <= 5
    ax.plot(result['detected_times'][mask], result['detected_freqs'][mask], 
            color=colors[i], linewidth=2, alpha=0.7, label=labels[i], 
            marker='o', markersize=4)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time (s)')
ax.set_title('First 5 Seconds (Detailed)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(14500, 20500)

# Plot 6: Magnitudes check
ax = axes[4]
for i, result in enumerate(results):
    ax.plot(result['detected_times'], result['detected_mags'], 
            color=colors[i], linewidth=1, alpha=0.7, label=labels[i])
ax.set_ylabel('Detected Magnitude')
ax.set_xlabel('Time (s)')
ax.set_title('Magnitude of Detected Points (Should be ~constant)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_yscale('log')

plt.tight_layout()

output_dir = "Exercise Inputs-20251113/Task 2/analysis"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "correct_detection_comparison.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to {output_file}")

print("\n" + "="*80)
