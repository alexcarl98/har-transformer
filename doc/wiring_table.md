Wiring Table for raspberry pi

Components Used:

- IR: MLX90640
- MIC: INMP441
- IMU: MPU6050

| Component  | Signal    | GPIO | Pin | Notes                                 |
| ---------- | --------- | ---- | --- | ------------------------------------- |
| IR,IMU,MIC | VCC (3.3) | —    | 1   | Shared 3.3V → breadboard `+` rail     |
| IR,IMU     | SDA       | 2    | 3   | I²C, Bussed on breadboard a-e, row 10 |
| IR,IMU     | SCL       | 3    | 5   | Bussed on a-e, row 30                 |
| IR,IMU,MIC | GND       | —    | 6   | Shared GND → breadboard `–` rail      |
| MIC        | SCK       | 18   | 12  | I²S Bit Clock                         |
| MIC        | WS        | 19   | 35  | Word Select                           |
| MIC        | SD        | 20   | 38  | Data                                  |
| MIC        | L/R (GND) | _    | 6   | Connected to ground                   |