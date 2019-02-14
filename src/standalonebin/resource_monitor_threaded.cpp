#include "resource_monitor_threaded.h"
#include "resource_monitor.h"

#include <iostream>
#include <unistd.h>
#include <string>


void ResourceMonitorThreaded::printInfo(std::string msg) {
  if (verbose_) {
    std::cerr << msg << std::endl;
  }
}

/*
  struct handler_info_t {

    ResourceMonitor * monitor;
    useconds_t wait_time;

    // Mutex
    std::mutex end_monitoring_lock;
    std::mutex monitor_resource_lock;

    // control variables
    bool end_monitoring;
  }*/

ResourceMonitorThreaded::ResourceMonitorThreaded(bool verbose) :
    running(false),
    verbose_(verbose) {
      printInfo("Instantiated ResourceMonitor Object");
      monitor = new ResourceMonitor(verbose);
      end_monitoring_lock = new std::mutex();
      monitor_resource_lock = new std::mutex();
      end_monitoring = new bool();
    }


ResourceMonitorThreaded::~ResourceMonitorThreaded() {
  printInfo("Destroyed ResourceMonitor Object");
  delete end_monitoring_lock;
  delete monitor_resource_lock;
  delete end_monitoring;
  delete monitor;
}


int ResourceMonitorThreaded::numData() {
  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

  return monitor->numData();
}

bool ResourceMonitorThreaded::hasData() {
  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

	return monitor->hasData();
}


void ResourceMonitorThreaded::clearData() {
  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

  monitor->clearData();
}


void ResourceMonitorThreaded::startMonitoring(double seconds) {

  if (running) {
    std::cerr << "WARN: Tried to start a resource monitor that is already running" << std::endl;
    return;
  }

  printInfo("before start monitor");

	setEndMonitor(false);
	running = true;

  monitor_resource_lock->lock();
  monitor->asyncDataFetch(); // (This will ensure that at least 2 values have been readed)
  monitor_resource_lock->unlock();

  handler_info = new monitor_handler_info_t();
  handler_info->monitor = monitor;
  handler_info->end_monitoring_lock = end_monitoring_lock;
  handler_info->monitor_resource_lock = monitor_resource_lock;
  handler_info->end_monitoring = end_monitoring;
  handler_info->wait_time = seconds * 1000000;  // micro seconds

  monitor_thread = std::thread(ResourceMonitorThreaded::background_monitor_handler, handler_info);
}


void ResourceMonitorThreaded::endMonitoring() {

  if (!running) {
    std::cerr << "WARN: Tried to end a resource monitor that is NOT running" << std::endl;
    return;
  }

  printInfo("before end monitor");

  monitor_resource_lock->lock();
	setEndMonitor(true);  // If endMonitor is true, no more measurements will be performed
  printInfo("before async lock");
  monitor->asyncDataFetch(); // (This will ensure that at least 2 values have been readed)
  monitor_resource_lock->unlock();

  printInfo("before join");
  monitor_thread.join();

  printInfo("before numdata");
  // This should never happen...
  int dataInMonitor = numData();
  if (dataInMonitor < 2) {
    cerr << "WARN: resource monitor contains less than 2 measurements (" << dataInMonitor << ") when commanded to end." << endl;
  }

  printInfo("after end monitoring");
  delete handler_info;

	running = false;
}


/*
 * Return total energy consumption from the CPU in mJ
 *
 */
double ResourceMonitorThreaded::getTotalEnergyCPU() {
  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

	return monitor->getTotalEnergyCPU();
}


/*
 * Return total energy consumption from the GPU in mJ
 *
 */
double ResourceMonitorThreaded::getTotalEnergyGPU() {
  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

	return monitor->getTotalEnergyGPU();
}


double ResourceMonitorThreaded::getTotalExecTime() {

	if (running) {
		cerr << "WARN: Tried to read resource monitor when it was still running." << endl;
		return 0;
	}

  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

	return monitor->getTotalExecTime();
}


double ResourceMonitorThreaded::getAveragePowerCPU() {

	if (running) {
		cerr << "WARN: Tried to read resource monitor when it was still running." << endl;
		return 0;
	}

  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

	return monitor->getAveragePowerCPU();
}

double ResourceMonitorThreaded::getAveragePowerGPU() {	

	if (running) {
		cerr << "WARN: Tried to read resource monitor when it was still running." << endl;
		return 0;
	}

  std::lock_guard<std::mutex> lock(*monitor_resource_lock);

	return monitor->getAveragePowerGPU();
}



void ResourceMonitorThreaded::getPowerCPU(vector<double> &timestamp, vector<double> &power) {
	
	if (running) {
		cerr << "WARN: Tried to read resource monitor when it was still running." << endl;
		return;
	}

	std::lock_guard<std::mutex> lock(*monitor_resource_lock);
	
  monitor->getPowerCPU(timestamp, power);
}

void ResourceMonitorThreaded::getPowerGPU(vector<double> &timestamp, vector<double> &power) {

	std::lock_guard<std::mutex> lock(*monitor_resource_lock);

  monitor->getPowerGPU(timestamp, power);
}

/**
 * Returns the power history for CPU and GPU
 * intervalTime contains the number of second for the corresponding interval
 * powerCPU and powerGPU contain the average power for the corresponding interval
 */
void ResourceMonitorThreaded::getPower(vector<double> &timestamp, vector<double> &powerCPU, vector<double> &powerGPU) {

	if (running) {
		cerr << "WARN: Tried to read resource monitor when it was still running." << endl;
		return;
	}

	std::lock_guard<std::mutex> lock(*monitor_resource_lock);

  monitor->getPower(timestamp, powerCPU, powerGPU);
}


void ResourceMonitorThreaded::background_monitor_handler(monitor_handler_info_t * info) {

/*
  struct handler_info_t {
    ResourceMonitor * monitor;
    useconds_t wait_time;
    std::mutex * end_monitoring_lock;
    std::mutex * monitor_resource_lock;
    bool * end_monitoring;
  }
*/
  bool end;
  usleep(info->wait_time);

  info->end_monitoring_lock->lock();
  end = *(info->end_monitoring);
  info->end_monitoring_lock->unlock();

	while(!end) {

    info->monitor_resource_lock->lock();
    info->monitor->asyncDataFetch();
    info->monitor_resource_lock->unlock();

		usleep(info->wait_time);
    info->end_monitoring_lock->lock();
    end = *(info->end_monitoring);
    info->end_monitoring_lock->unlock();
	}

}

bool ResourceMonitorThreaded::monitorMustEnd() {
	
	std::lock_guard<std::mutex> lock(*end_monitoring_lock);

	return *end_monitoring;
}

void ResourceMonitorThreaded::setEndMonitor(bool value) {

  std::lock_guard<std::mutex> lock(*end_monitoring_lock);

	*end_monitoring = value;
}
