#include "probeGPU.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <nvml.h>


using namespace std;


ProbeGPU::ProbeGPU(bool verbose) {
  verbose_ = verbose;
}


ProbeGPU::~ProbeGPU() {
		

}


void ProbeGPU::printMessage(char *msg) {
  if(verbose_) {
    std::cerr << msg << std::endl;
  }
}

void ProbeGPU::init() {

	char name[100];
	nvmlReturn_t result;

	unsigned int device_count;

	result = nvmlInit();
	if (result != NVML_SUCCESS) {
		cerr << "Could not initiate nvml library. GPU is not monitored." << endl;
		cerr << nvmlErrorString(result) << endl;
		return;
	}

	result = nvmlDeviceGetCount(&device_count);
	if (result != NVML_SUCCESS) {
		cerr << "Could not obtain number of GPU devices." << endl;
		cerr << nvmlErrorString(result) << endl;
		return;
	}

	if (device_count <= 0) {
		cerr << "No GPU devices found." << endl;
		return;
	}

	result = nvmlDeviceGetHandleByIndex(0, &device);
	if (result != NVML_SUCCESS) {
		cerr << "Could not get handle for GPU device 0." << endl;
		cerr << nvmlErrorString(result) << endl;
		return;
	}

	result = nvmlDeviceGetName(device, name, sizeof(name)/sizeof(name[0]));
	if (result != NVML_SUCCESS) {
		cerr << "Could not get name for GPU device 0." << endl;
		cerr << nvmlErrorString(result) << endl;
		return;
	}

	printMessage(name);
}


void ProbeGPU::fetchRawPower(rawPowerGPU_t *rawPower) {
	nvmlReturn_t result;

	result = nvmlDeviceGetPowerUsage(device, &(rawPower->power));
	if (result != NVML_SUCCESS) {
		cerr << "Could not fetch GPU power." << endl;
	}
}

void ProbeGPU::raw2watt(rawPowerGPU_t *rawPower, powerGPU_t *power) {
	power->power = (double)rawPower->power / 1000;
}

