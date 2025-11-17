# Agent.md explaining the project and implementation

# Project: Inaudible Watermarking in Audio Signals using Fourier Transform

## 1. Project Overview
This project focuses on adding an inaudible watermark to audio files (.wav) and detecting watermarks added by other methods. The watermark is placed in high frequencies (above 20 kHz), which are inaudible to humans but visible in the frequency domain. 

Key components:
- Fourier Transform (FFT) of the audio signal
- Insertion of watermark (e.g., comb pattern) in frequency domain
- Preservation of conjugate symmetry to keep signal real
- Inverse FFT to return to time domain
- Detection via spectrum analysis (magnitude, autocorrelation, etc.)

## 2. Implementation Steps

1. **Load the audio**
    - Use `scipy.io.wavfile.read` to read the .wav file
    - Always use the correct sampling rate
    - Normalize the audio to [-1, 1]

2. **Compute FFT**
    - Use `np.fft.fft` for full audio signal
    - Use `np.fft.fftfreq(N, d=1/sample_rate)` to get correct frequency values in Hz

3. **Apply Watermark**
    - Target high frequencies (e.g., >20 kHz)
    - Modify magnitude or phase in FFT bins
    - Preserve conjugate symmetry: `X[-k] = np.conj(X[k])`

4. **Inverse FFT**
    - Use `np.fft.ifft` and take the real part
    - Scale back to int16 and save using `wavfile.write`

5. **Detection**
    - Visualize the log-magnitude spectrum
    - Look for periodic patterns (comb) or peaks at known watermark locations

6. **Visualization**
    - Read .wav file or directory of .wav files
    - Compute log-magnitude spectrum
    - Save plot as PNG/JPEG