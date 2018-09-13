
#include <stdlib.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <byteswap.h>



class ProbeARM {

public:

	ProbeARM();
	~ProbeARM();

	int init();

	double fetchPowerCPU();
	double fetchPowerGPU();


private:

	double readPower(uint8_t shunt_voltage_addr, uint8_t bus_voltage_addr, uint8_t shunt_resistance);

	int i2c_file;

	const uint8_t gpu_shunt_voltage_addr = 0x03;
	const uint8_t gpu_bus_voltage_addr = 0x04;
	const uint8_t gpu_shunt_resistance = 10;  // milli OHM

	const uint8_t cpu_shunt_voltage_addr = 0x05;
	const uint8_t cpu_bus_voltage_addr = 0x06;
	const uint8_t cpu_shunt_resistance = 10; // milli OHM

	const uint8_t shunt_voltage_lsb = 40;
	const uint8_t bus_voltage_lsb = 8;

	const uint8_t i2c_bus = 1;
	const uint8_t i2c_addr = 0x40;
};
