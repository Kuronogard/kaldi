#include "probeGPU.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <nvml.h>


using namespace std;

ProbeGPU::ProbeGPU() {

}


ProbeGPU::~ProbeGPU() {
		

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

	cerr << "monitoring GPU Device 0: " << string(name) << "." << endl;
}


void ProbeGPU::fetchPower(unsigned int *power) {

	nvmlDeviceGetPowerUsage(device, power);
}


