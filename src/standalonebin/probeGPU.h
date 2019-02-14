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

	ProbeGPU(bool verbose = false);
	~ProbeGPU();

	void init();
	void fetchRawPower(rawPowerGPU_t * rawPower);
	void raw2watt(rawPowerGPU_t * rawPower, powerGPU_t * power);

private:

  bool verbose_;
	nvmlDevice_t device;

  void printMessage(char *msg);
};


#endif
