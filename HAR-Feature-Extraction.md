# Acquire Filter Segment

(Collect–Wash–Cut)


## Time domain features:
Statistical properties of a signal
- _Central Tendency_: Mean, median
- _Variability_: stddev, range, IQR
    - Interquartile range (IQR), usually between 25th and 75th quartile
- _Distribution Shape_: Skewness, kurtosis
    - Skewness: How symmetric the distribution is
    - Kurtosis: measure of how rounded?
    - Signal entropy and kurtosis
- _extrema_: min, max, peak count
    - peak to peak amplitude + zero crossing rate
- _Signal Magnitude_: √(x²+y²+z²)


## Frequency domain features:
- periodic and cyclic aspects
- Spectral Energy Distribution
- Frequency band ratios


- _Fast Fourier Transform (FFT)_: converts time signals to frequency
- _Dominant Frequency_: main component of movement
    - in Hz
    - in step counting it was around 2 Hz and we singled out those frequencies.
- _Spectral Energy_: overall intensity across frequencies
    - Sum of squares of frequency magnitude
- _Frequency Range Power_: energy in specific bands
- _Spectral Entropy_: complexity of frequency distribution
    - Is the signal like a straight line? Or kind of unpredictable?


## Correlation-Based features:
- relationships between axes
- Correlation between axes
## Combined/Derived Features:
- Information from multiple sources
- Signal Quantiles
- Autoregressive Coefficient