# Audio Watermarking Project - Summary

## Overview
This project implements inaudible audio watermarking techniques using Fourier Transform and explores different methods of time-stretching watermarked audio while preserving frequency content.

---

## Task 1: Watermark Implementation

### Good Watermark (Inaudible)
**Technique**: Frequency-domain watermarking in the inaudible range

**Implementation**:
```python
def add_watermark(audio, sample_rate, 
                  watermark_freq_start=20000,  # Start at 20 kHz (inaudible)
                  freq_spacing=500,             # 500 Hz spacing
                  watermark_width=50,           # 50 Hz width per peak
                  amplitude_add=1000.0)         # Amplitude boost
```

**Process**:
1. **FFT**: Convert audio signal to frequency domain
2. **Watermark Pattern**: Add comb pattern (periodic peaks) starting at 20 kHz
   - Frequencies: 20000 Hz, 20500 Hz, 21000 Hz, etc.
   - Width: 50 Hz per peak
3. **Phase Preservation**: Maintain original phase relationships
4. **Conjugate Symmetry**: Preserve real signal properties
5. **IFFT**: Convert back to time domain

**Key Features**:
- ✅ **Inaudible**: Above 20 kHz (human hearing limit ~20 kHz)
- ✅ **Visible in Spectrum**: Clear comb pattern in frequency analysis
- ✅ **Minimal Distortion**: Preserves audio quality
- ✅ **Robust**: Can survive some processing operations

### Bad Watermark (Audible - for comparison)
**Technique**: Same algorithm but starting at 100 Hz (audible range)

**Why it's bad**:
- ❌ **Audible**: Creates noticeable artifacts in the audio
- ❌ **Easily Detectable**: Obvious in frequency spectrum
- ❌ **Quality Degradation**: Affects listening experience

---

## Task 2: Watermark Detection

### Detection Algorithm: STFT-based Frequency Tracking
**Purpose**: Detect time-varying FM watermark patterns

**Method**:
```python
def detect_watermark_stft(audio, sample_rate,
                         search_freq_min=15000,
                         search_freq_max=20000,
                         nperseg=2048)
```

**Algorithm Steps**:
1. **STFT (Short-Time Fourier Transform)**: Create time-frequency representation
2. **Frequency Tracking**: Find brightest frequency in 15-20 kHz range at each time window
3. **Pattern Analysis**: Track how the peak frequency varies over time
4. **Peak Detection**: Find peaks in the frequency trajectory (using `scipy.signal.find_peaks`)
5. **Frequency Calculation**: Determine watermark frequency from peak spacing

**Watermark Pattern Detected**:
- The watermark is an **FM (Frequency Modulation) signal**
- Center frequency oscillates sinusoidally: `f(t) = f_center + A × sin(2π × f_watermark × t)`
- Example: Center at ~18 kHz, oscillating between 15-20 kHz

**Output**:
- Center frequency
- Frequency amplitude (oscillation range)
- Watermark frequency (Hz)
- Period (seconds)
- Number of cycles in the audio

---

## Task 3: Time-Stretching Methods Analysis

### Scenario
- **Original**: 30-second audio at 44,100 Hz with FM watermark at 20+ kHz
- **Goal**: Slow down to 120 seconds (4× slower)
- **Challenge**: Preserve the high-frequency watermark visibility

### Method 1: Metadata Change (Naive Slowdown)
**Process**:
1. Take original 1,323,008 samples
2. **Change sample rate metadata** from 44,100 Hz → 11,025 Hz
3. Keep the same samples (no resampling)

**Results**:
- ✅ Duration: 120 seconds (1,323,008 ÷ 11,025 = 120 sec)
- ✅ Playback: 4× slower
- ❌ **Frequency compression**: All frequencies divided by 4
  - 20 kHz watermark → 5 kHz
  - 1 kHz tone → 250 Hz
- ❌ **Nyquist limit drops**: 22,050 Hz → 5,512 Hz
- ❌ **Watermark LOST**: Can't represent frequencies above 5.5 kHz
- ❌ **Spectrogram**: Only shows 0-5.5 kHz range

**Why it fails**:
- Sample rate determines the Nyquist frequency (max representable frequency = sample_rate / 2)
- Lowering sample rate to 11 kHz means max frequency is 5.5 kHz
- The 20 kHz watermark is now above Nyquist and gets aliased/destroyed

---

### Method 2: Fourier-Based Resampling (Proper Time-Stretch)
**Process**:
1. Take original 1,323,008 samples at 44,100 Hz
2. **FFT**: Transform to frequency domain
3. **Zero-Pad Spectrum**: Insert zeros between positive and negative frequencies
   - Original: 1,323,008 frequency bins
   - Padded: 5,292,032 frequency bins (4× more)
   - Copy first half (positive frequencies) → first positions
   - Insert zeros in middle
   - Copy second half (negative frequencies) → last positions
   - Scale by 4× to maintain amplitude
4. **IFFT**: Transform back to time domain with 4× more samples
5. Keep sample rate at 44,100 Hz

**Results**:
- ✅ Duration: 120 seconds (5,292,032 ÷ 44,100 = 120 sec)
- ✅ Playback: 4× slower
- ✅ **Frequencies preserved**: All stay at original Hz values
  - 20 kHz watermark → stays at 20 kHz (just lasts longer)
  - 1 kHz tone → stays at 1 kHz (just lasts longer)
- ✅ **Nyquist limit maintained**: 22,050 Hz
- ✅ **Watermark VISIBLE**: FM watermark intact and visible
- ✅ **Spectrogram**: Full 0-22 kHz range displayed

**Why it succeeds**:
- Maintains high sample rate (44.1 kHz) → Nyquist stays at 22 kHz
- Zero-padding in frequency domain = perfect interpolation in time domain (sinc interpolation)
- No new frequencies added, just more samples between existing ones
- Mathematically optimal bandwidth-limited interpolation

---

## Key Insights

### 1. Watermark Design
- **Inaudible watermarks** must be above human hearing range (>20 kHz)
- Requires high sample rates (≥44.1 kHz) to represent these frequencies
- Comb patterns provide robust, easily detectable signatures

### 2. Nyquist Theorem is Critical
- **Maximum frequency** that can be represented = `sample_rate / 2`
- To preserve 20+ kHz watermarks, sample rate must be ≥40 kHz
- Downsampling destroys high-frequency content permanently

### 3. Fourier-Based Resampling
- **Zero-padding in frequency domain** = time-domain interpolation
- Preserves all frequencies perfectly
- Standard technique in audio processing (`scipy.signal.resample`)
- Ideal for time-stretching while preserving spectral content

### 4. Time vs. Frequency Trade-offs
- **Method 1**: Saves space (smaller file) but loses frequency information
- **Method 2**: Preserves quality but requires 4× more samples
- Choice depends on application: bandwidth vs. fidelity

---

## Visualizations Generated

### Spectrograms
- Show time-frequency representation
- Y-axis: Frequency (Hz)
- X-axis: Time (seconds)
- Color: Energy/magnitude (dB)

### Key Comparisons
1. **Task 1**: Good vs. Bad watermark spectrograms
2. **Task 2**: Watermark detection plots showing FM patterns
3. **Task 3**: Method 1 vs. Method 2 spectrograms
   - Method 1: 0-5.5 kHz (watermark missing)
   - Method 2: 0-22 kHz (watermark visible)

---

## Technical Implementation

### Libraries Used
- `numpy`: FFT/IFFT operations, array processing
- `scipy.io.wavfile`: Audio file I/O
- `scipy.signal`: STFT, spectrogram, peak detection
- `matplotlib`: Visualization

### Core Algorithms
1. **FFT/IFFT**: Frequency domain transformations
2. **STFT**: Time-frequency analysis
3. **Zero-padding**: Bandwidth-limited interpolation
4. **Peak detection**: Pattern recognition in signals

---

## Conclusion

This project demonstrates:
- How to implement **inaudible watermarks** using Fourier Transform
- The importance of **sample rate** in preserving high-frequency content
- **Fourier-based resampling** as the proper method for time-stretching
- The trade-offs between **file size, processing, and quality preservation**

The key takeaway: **When working with high-frequency content (like watermarks), maintaining adequate sample rates is essential—frequency information lost through downsampling cannot be recovered.**
