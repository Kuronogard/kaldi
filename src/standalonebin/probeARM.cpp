#include "probeARM.h"
#include <iostream>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <byteswap.h>
#include <unistd.h>
#include <fcntl.h>
#include <sstream>
#include <linux/i2c-dev.h>

using namespace std;

ProbeARM::ProbeARM() {


}

ProbeARM::~ProbeARM() {
	close(i2c_file);
}


int ProbeARM::init() {

	char filename[20];

	// Open i2c file
	snprintf(filename, 20, "/dev/i2c-%d", i2c_bus);
	i2c_file = open(filename, O_RDWR);
	if (i2c_file < 0) {
		cerr << "Could not access ARM power info." << endl;
		cerr << "Error opening file: " << filename << endl;
		cerr << "-> " << strerror(errno) << endl;
		return -1;
	}

	int result;
	result = ioctl(i2c_file, I2C_SLAVE, i2c_addr);
	if (result < 0) {
		cerr << "Could not access ARM power info." << endl;
		cerr << "IOCTL error: " << strerror(errno) << endl;
		return -2;
	}

	return 0;
}


double ProbeARM::readPower(uint8_t shunt_voltage_addr, uint8_t bus_voltage_addr, uint8_t shunt_resistance) {

	uint16_t read_temp;
	int shunt_voltage, bus_voltage;
	double current, power;

	read_temp = __bswap_16(i2c_smbus_read_word_data(i2c_file, shunt_voltage_addr));
	shunt_voltage = (int)((read_temp & 0x7FFF) >> 3) * shunt_voltage_lsb;

	read_temp = __bswap_16(i2c_smbus_read_word_data(i2c_file, bus_voltage_addr));
	bus_voltage = (int)((read_temp & 0x7FFF) >> 3) * bus_voltage_lsb;

	current = shunt_voltage / shunt_resistance;
	power = current * bus_voltage / 1000;

	return power/1000;
}


double ProbeARM::fetchPowerCPU() {
	return readPower(cpu_shunt_voltage_addr, cpu_bus_voltage_addr, cpu_shunt_resistance);
}


double ProbeARM::fetchPowerGPU() {
	return readPower(gpu_shunt_voltage_addr, gpu_bus_voltage_addr, gpu_shunt_resistance);
}

