#ifndef __RESOURCE_MONITOR_ARM
#define __RESOURCE_MONITOR_ARM

#include <stdlib.h>
#include <pthread.h>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "standalonebin/probeARM.h"

using namespace std;


class ResourceMonitorARM {

public:
	ResourceMonitorARM();
	~ResourceMonitorARM();

	void startMonitoring(double seconds);
	void endMonitoring();

	void startMonitoringNoThread();
	void endMonitoringNoThread();
	
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

	ProbeARM probeARM;

	bool initialized;
	bool running;
	bool runningNoThread;
	pthread_mutex_t lock;
	bool end_monitoring;
	pthread_t monitor_thread;

	struct timeval _startTime;
	struct timeval _endTime;
	struct timezone _timeZone;

	vector<timeval> _timestamp;
	vector<double> _powerCPU;
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


#endif /* __RESOURCE_MONITOR_ARM*/
