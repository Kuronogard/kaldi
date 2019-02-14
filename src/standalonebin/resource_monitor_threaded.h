#ifndef __RESOURCE_MONITOR_THREADED_H
#define __RESOURCE_MONITOR_THREADED_H

#include <stdlib.h>
#include <thread>
#include <mutex>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <string>

#include "resource_monitor.h"

using namespace std;


class ResourceMonitorThreaded {

public:
	ResourceMonitorThreaded(bool verbose = false);
	~ResourceMonitorThreaded();

	void startMonitoring(double seconds);
	void endMonitoring();

  void setVerbose(bool verbose);

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
  void clearData();

private:

  struct monitor_handler_info_t {
  
    ResourceMonitor * monitor;
    useconds_t wait_time;

    // Mutex
    std::mutex * end_monitoring_lock;
    std::mutex * monitor_resource_lock;

    // control variables
    bool * end_monitoring;
  };


  monitor_handler_info_t * handler_info;

	bool running;
  bool verbose_;

  ResourceMonitor * monitor;

  // Mutex
  std::mutex * end_monitoring_lock;
  std::mutex * monitor_resource_lock;

  // control variables
  bool * end_monitoring;

  // thread
	std::thread monitor_thread;

	static void background_monitor_handler(monitor_handler_info_t * info);
	bool monitorMustEnd();
	void setEndMonitor(bool value);
  void printInfo(std::string msg);
};

#endif
