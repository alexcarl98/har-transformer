# Transformer based methods in Sensor Location Invariant HAR-Classification

## Repo Structure:
```
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

