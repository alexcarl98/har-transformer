# Location-Invariant Human Activity Recognition Using Transformers

A deep learning approach for human activity classification that works consistently across different sensor locations (waist, ankle, wrist) using transformer-based architectures. 
- Original assignment can be located under `doc/`
- Dataset from [LMU Har-Lab](https://github.com/Har-Lab/HumanActivityData)

## Features

- Multi-sensor data processing from three body locations (waist, ankle, wrist)
- Advanced signal processing with FFT and statistical features
- Transformer-based architecture for temporal sequence learning
- Comprehensive evaluation metrics including precision, recall, and F1-score
- Real-time visualization of model performance

## Performance

The model is evaluated using:
- Confusion matrices
- Per-class precision, recall, and F1-scores
- Overall accuracy metrics
- Cross-validation results

## Getting Started:

### Install Dependencies
While I have provided an `environment.yml` & `requirements.txt` **I would recommend copying these commands manually to initialize your environment:**
> **NOTE**: If you do not have `conda` installed, please see [these instructions](https://www.anaconda.com/docs/getting-started/miniconda/install) for further details

```sh
conda create -y -n har python=3.12

conda activate har

conda install -c conda-forge numpy pandas scipy + tqdm matplotlib wandb
conda install pytorch -c pytorch
conda install -c conda-forge pytest

pip install ahrs    # no conda package for ahrs
```

### Download Dataset

```sh
python src/download_dataset.py
```

After running you should have a new `raw_data/` file which would look like this:

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
├── tests/
├── scripts/
└── src/
```


### Train Model
Run this command to start training the transformer
```sh
python src/train.py
```

### Hardware Setup
You will need:
- Raspberry Pi Zero 2W
- MPU6050 Accelerometer

> **NOTE**: Wiring table for connecting the accelerometer to the pi can be found under `docs/`

After installing raspbian OS onto your pi, copy over the `pi_files/` directory to the root of the rasperry pi. 
```sh
# Obtain the ip address:
ping raspberrypi.local

# Set these variables for your pi
USERNAME="your-user-name"
IP_ADDRESS="XXX.XXX.XX.XX"  # from above
scp -r `pi_files/` $USERNAME@$IP_ADDRESS:~ 
```

SSH into your pi:
```sh
ssh $USERNAME@$IP_ADDRESS
```

Run the files as follows
- PI: Run `~/pi_files/init_hardware.sh`
    - Only done the first time
- HOST: Run `src/real_time.py`
- PI: Run `pi_files/send_accelerometer_stream.py`

