cd ~
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y install python3-dev
sudo apt install -y python3-pip

sudo pip3 install numpy --break-system-packages
sudo pip3 install requests --break-system-packages
sudo pip3 install adafruit-circuitpython-mpu6050 --break-system-packages