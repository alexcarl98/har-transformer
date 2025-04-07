# Wiring Table for raspberry pi

Components Used:
- IMU: MPU6050

| Component  | Signal    | GPIO | Pin | Notes                                 |
| ---------- | --------- | ---- | --- | ------------------------------------- |
| IMU | VCC (3.3) | —    | 1   | Shared 3.3V → breadboard `+` rail     |
| IMU     | SDA       | 2    | 3   | I²C, Bussed on breadboard a-e, row 10 |
| IMU     | SCL       | 3    | 5   | Bussed on a-e, rFow 30                 |
| IMU | GND       | —    | 6   | Shared GND → breadboard `–` rail      |