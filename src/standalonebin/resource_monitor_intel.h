#ifndef __RESOURCE_MONITOR_INTEL_H
#define __RESOURCE_MONITOR_INTEL_H

#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "standalonebin/probeCPU.h"
#include "standalonebin/probeGPU.h"

using namespace std;


class ResourceMonitorIntel {

public:
	ResourceMonitorIntel(bool verbose = false);
	~ResourceMonitorIntel();

	double getTotalEnergyCPU();
	double getTotalEnergyGPU();
	double getTotalExecTime();
	double getAveragePowerCPU();
	double getAveragePowerGPU();

	void getPowerCPU(vector<double> &timestamp, vector<double> &power);
	void getPowerGPU(vector<double> &timestamp, vector<double> &power);
	void getPower(vector<double> &timestamp, vector<double> &powerCPU, vector<double> &powerGPU);

  void asyncDataFetch();

	int numData();
	bool hasData();
  void clearData();

private:

	ProbeCPU probeCPU;
	ProbeGPU probeGPU;

  bool verbose_;

	struct timezone _timeZone;

	vector<timeval> _timestamp;
	vector<rawEnergyCPU_t> _energyCPU;
	vector<double> _powerGPU;

	//bool firstRead();
	//void setFirstRead(bool value);

  void printInfo(bool verbose, std::string msg);
	double timeInterval(struct timeval start, struct timeval end);
	double interval_GPU_power(int i);
	double interval_CPU_power(int i);
	double interval_CPU_power_est(int i, double prev_power);
	double timeval2double(struct timeval time);
};

#endif
