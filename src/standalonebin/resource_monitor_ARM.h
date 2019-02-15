#ifndef __RESOURCE_MONITOR_ARM
#define __RESOURCE_MONITOR_ARM

#include <stdlib.h>
#include <pthread.h>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "standalonebin/probeARM.h"

using namespace std;


class ResourceMonitorARM {

public:
	ResourceMonitorARM(bool verbose = false);
	~ResourceMonitorARM();

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

	ProbeARM probeARM;

  bool verbose_;

	vector<timeval> _timestamp;
	vector<double> _powerCPU;
	vector<double> _powerGPU;

  void printInfo(std::string msg);
	double timeInterval(struct timeval start, struct timeval end);
	double interval_GPU_power(int i);
	double interval_CPU_power(int i);
	double interval_CPU_power_est(int i, double prev_power);
	double timeval2double(struct timeval time);
};


#endif /* __RESOURCE_MONITOR_ARM*/
