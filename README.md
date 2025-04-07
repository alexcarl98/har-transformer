# Location-Invariant Human Activity Recognition Using Transformers

A deep learning approach for human activity classification that works consistently across different sensor locations (waist, ankle, wrist) using transformer-based architectures.

## Features

- Multi-sensor data processing from three body locations (waist, ankle, wrist)
- Advanced signal processing with FFT and statistical features
- Transformer-based architecture for temporal sequence learning
- Comprehensive evaluation metrics including precision, recall, and F1-score
- Real-time visualization of model performance

## 🛠️ Technical Stack

- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data processing and manipulation
- **SciPy**: Signal processing
- **Scikit-learn**: Machine learning utilities and metrics
- **Matplotlib/Seaborn**: Visualization

## 📊 Data Processing Pipeline

1. Raw accelerometer data ingestion
2. Window-based signal feature extraction
3. FFT transformation for frequency domain analysis
4. Statistical feature computation
5. Data normalization and preparation for the transformer model

## Model Architecture

The project implements a custom transformer architecture (`AccelTransformer`) that combines:
- Sequential accelerometer data processing
- Additional metadata feature integration
- Multi-head attention mechanisms
- Classification head for activity prediction

## Performance

The model is evaluated using:
- Confusion matrices
- Per-class precision, recall, and F1-scores
- Overall accuracy metrics
- Cross-validation results

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:

## Repo Structure:

```sh
root/
├── doc/
│   ├── ARC_step_counter.ipynb
│   ├── HAR_Classification.ipynb
│   ├── HAR-Feature-Extraction.md
│   ├── latex/
│   │   └── figure
│   ├── REPORT.md
│   └── wiring_table.md
├── pi_files/
│   ├── init_hardware.sh
│   └── send_accelerometer_stream.py
├── raw_data/
│   ├── 001.csv
│   ├── 002.csv
│   ├──  ...
│   ├── 040.csv
│   └── 041.csv
├── README.md
├── requirements.txt
└── src
    ├── constants.py        
    ├── download_dataset.py
    ├── exp
    │   ├── har_transformer.py
    │   └── NOTE.md
    ├── har_model.py
    ├── preprocessing.py
    ├── __pycache__
    │   └── constants.cpython-312.pyc
    ├── real_time.py
    └── train.py
```

- `doc/`: contains original assignments and responses
- `pi_files/`: Contains scripts that need to be sent to the hardware (raspberry pi + MPU6050)
- `src/`: Contains programs for downloading data, preprocessing data, and training a Transformer on the data.


## Instructions:

### Install Dependencies
While I have provided an `environment.yml` & `requirements.txt` **I would recommend copying these commands manually to initialize your environment:**
> **NOTE**: If you do not have `conda` installed, please see [these instructions](https://www.anaconda.com/docs/getting-started/miniconda/install) for further details

```sh
conda create -y -n har python=3.12

conda activate har

conda install -c conda-forge numpy pandas scipy scikit-learn seaborn tqdm matplotlib wandb
conda install pytorch -c pytorch
conda install -c conda-forge pytest

pip install ahrs    # no conda package for ahrs
```



### Download Dataset
Dataset from [LMU Har-Labs](https://github.com/Har-Lab/HumanActivityData)

```sh
python src/download_dataset.py
```

after running you should have a new raw_data file which would look like this:

```sh
root/
├── doc/
├── pi_files/
├── raw_data/       # new directory with semi-formatted sequences of raw data
│   ├── 001.csv
│   ├── 002.csv
│   ├──  ...
│   ├── 040.csv
│   └── 041.csv
├── README.md
├── requirements.txt
└── src
```


### Train Model
Run this command to start training the transformer
```sh
python src/train.py
```

### Hardware Setup

