#ifndef RESOURCE_MONITOR_H
#define RESOURCE_MONITOR_H

#include <stdlib.h>
#include <pthread.h>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "standalonebin/probeCPU.h"
#include "standalonebin/probeGPU.h"

using namespace std;


class ResourceMonitorIntel {

public:
	ResourceMonitorIntel();
	~ResourceMonitorIntel();

	void startMonitoring(double seconds);
	void endMonitoring();
	
	void init();

	double getTotalEnergyCPU();
	double getTotalEnergyGPU();
	double getTotalExecTime();
	double getAveragePowerCPU();
	double getAveragePowerGPU();

	void getPowerCPU(vector<double> &timestamp, vector<double> &power);
	void getPowerGPU(vector<double> &timestamp, vector<double> &power);
	void getPower(vector<double> &timestamp, vector<double> &powerCPU, vector<double> &powerGPU);

	int numData();
	bool hasData();

private:

	ProbeCPU probeCPU;
	ProbeGPU probeGPU;

	bool running;
	pthread_mutex_t lock;
	bool end_monitoring;
	pthread_t monitor_thread;

	struct timeval _startTime;
	struct timeval _endTime;
	struct timezone _timeZone;

	vector<timeval> _timestamp;
	vector<rawEnergyCPU_t> _energyCPU;
	vector<double> _powerGPU;

	useconds_t measure_period;

	static void * background_monitor_handler(void * args);
	bool monitorMustEnd();
	void setEndMonitor(bool value);

	double timeInterval(struct timeval start, struct timeval end);
	double interval_GPU_power(int i);
	double interval_CPU_power(int i);
	double interval_CPU_power_est(int i, double prev_power);
	double timeval2double(struct timeval time);
};

#endif
