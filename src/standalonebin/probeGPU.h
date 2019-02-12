#ifndef __PROBEGPU_H
#define __PROBEGPU_H


#include <stdlib.h>
#include <nvml.h>


struct rawPowerGPU_t {
	unsigned int power;
};


struct powerGPU_t {
	double power;
};


class ProbeGPU {

public:

	ProbeGPU();
	~ProbeGPU();

	void init();
	void fetchRawPower(rawPowerGPU_t * rawPower);
	void raw2watt(rawPowerGPU_t * rawPower, powerGPU_t * power);

private:

	nvmlDevice_t device;

};


#endif
