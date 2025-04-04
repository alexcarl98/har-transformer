# Acquire Filter Segment

(Collect–Wash–Cut)

## Time Domain features:

Statistical properties of a signal

- _Central Tendency_: Mean, median
- _Variability_: stddev, range, IQR
  - Interquartile range (IQR): usually between 25th and 75th quartile
- _Distribution Shape_: Skewness, kurtosis
  - Skewness: How symmetric the distribution is
  - Kurtosis: measure of how rounded?
  - Signal entropy and kurtosis
- _extrema_: min, max, peak count
  - peak to peak amplitude + zero crossing rate
- _Signal Magnitude_: √(x²+y²+z²)

## Frequency Domain features:

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

- Relationships between axes
- Correlation between axes (x-y, y-z, x-z) measure how movements along one axis
  relate to movements along another.

#### Purpose:

- **Reveals Coordination Patterns**: In activities like walking or running,
  certain axes might move together, indicating rhythmic or coordinated
  movements.

2. Signal dynamics capture temporal and amplitude variations in the signal:

- _Zero-Crossing Rate (ZCR)_: counts how often the signal crosses the zero line
  in a given time frame
  - High ZCR: rap oscillations (vigorous activities)
  - Low ZCR: smoother, slower motions (walking, sitting)

- _Peak Count_: Number of local maxima or minima in signal:
  - Frequent: jerky movements
  - fewer: smoother, steadier activites movements
- _Autocorrelation_: Measures how similar the signal is to a time delayed
  version of itself.
  - _High Autocorrelation_: periodic, repetitive motions
  - _Low Autocorrelation_: random, non-repetitive motions

## Combined/Derived Features:

- Information from multiple sources
- Signal Quantiles
- Autoregressive Coefficient

## Extraction Process:

1. Segment data into windows (100 samples)
2. Apply sliding window with no overlap
3. Calculate features for each window
4. Create feature vector for each window
5. Associate with activity label

### Window based feature extraction:

- window size too small: miss activity patterns
- window size too large: mix too many patterns
- Overlap provides more training samples
- Eeach window becomes one sample in feature space

### Feature Explosion:

- Many possible features × multiple axes × multiple sensors

- Too many featires:
  - Overfitting
  - Computational Complexity
  - "Curse of Dimensionality"
  - Reduced Interpretability

### Feature Selection: Finding what really matters

1. _Filter Methods_: Rank features based on statistical measures:
   - Correlation w/target variable, variance threshold
2. _Wrapper Methods_: Evaluate feature subsets using the model itself
   - Recursive Feature Elimination (RFE)
   - Tries several combinations of features kind of by brute force
3. _Embedded Methods_: Feature selection as part of model training
   - Feature importance in Random Forests, L1 regularization
4. _Principle Component Analysis (PCA)_: Dimensionality reduction technique
   - Creates new uncorrelated features from original features

## Feature Importance in Activity Recognition
- Standard Deviation of signals (variability)
- Frequency domain energy features
- Signal magnitude features
- Axis correlation features

## Workflow:

1. Understand activity patterns
2. Extract Time Domain Features
3. Extract Frequency domain features
4. Evaluate feature importance
5. Select Optimal feature subset
6. Train and optimize ML model


#### Activity Specific Characteristics:
- Walking: Regular periodic patterns with moderate amplitude
- Joggin: Higher frequency and amplitude than walking
- Stairs: Asymmetric patterns with distinctive vertical components
- Treadmill vs free walking: Treadmill shows more consistent patterns

#### Sensor Placement Consideration
- Waist/hip: Captures overall body movement