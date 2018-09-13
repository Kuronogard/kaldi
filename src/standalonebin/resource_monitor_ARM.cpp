#include "resource_monitor_ARM.h"

#include <sys/time.h>
#include <iostream>
#include <pthread.h>
#include <unistd.h>


bool ResourceMonitorARM::hasData() {
	return _timestamp.size() > 0;
}


double ResourceMonitorARM::interval_GPU_power(int i) {
	return (_powerGPU[i] + _powerGPU[i+1]) / 2;
}


double ResourceMonitorARM::interval_CPU_power(int i) {
	return (_powerCPU[i] + _powerCPU[i+1]) / 2;
}


ResourceMonitorARM::ResourceMonitorARM() {
	running = false;
	initialized = false;
	pthread_mutex_init(&lock, NULL);
}

ResourceMonitorARM::~ResourceMonitorARM() {
	pthread_mutex_destroy(&lock);

}


/**
 * Initialize the monitoring object, trying to access each relevant
 * system
 */
void ResourceMonitorARM::init() {

	if (running) {
		cerr << "WARN: Tried to reinitialize resorce monitor. " << endl;
		cerr << "Doing nothing..." << endl;
		return;
	}

	_timestamp.clear();
	_powerCPU.clear();
	_powerGPU.clear();

	probeARM.init();

	initialized = true;
}


void ResourceMonitorARM::startMonitoring(double seconds) {
	
	if (running) {
		cerr << "WARN: Resource Monitor is already running" << endl;
		cerr << "Doing nothing..." << endl;
		return;
	}

	measure_period = seconds * 1000000;
	setEndMonitor(false);
	running = true;

	gettimeofday(&_startTime, &_timeZone);
	pthread_create(&monitor_thread, NULL, &background_monitor_handler, (void*)this);
}


void ResourceMonitorARM::endMonitoring() {

	if (!running) {
		cerr << "WARN: Tried to end monitoring. But it was not running" << endl;
		cerr << "Doing nothing" << endl;
		return;
	}

	gettimeofday(&_endTime, &_timeZone);
	setEndMonitor(true);

	pthread_join(monitor_thread, NULL);
	running = false;
}


/*
 * Return total energy consumption from the CPU in mJ
 *
 */
double ResourceMonitorARM::getTotalEnergyCPU() {

	if (!hasData()) {
		cerr << "WARN: No data in resource monitor" << endl;
		return 0;
	}

	// probeCPU fetches energy
	// Get the first measure, the last one, and substract them
	double energy = 0;

	for(int i = 0; i < _powerCPU.size()-1; i++) {
		double time, power;

		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		power = interval_CPU_power(i);

		energy += power * time;
	}

	return energy;

}


/*
 * Return total energy consumption from the GPU in mJ
 *
 */
double ResourceMonitorARM::getTotalEnergyGPU() {
	// probeGPU fetches power
	// Read _powerGPU and approximate the power for each interval as the mean value between the
	// starting and the ending power of the interval
	// Compute energy for each interval and sum all of them

	if (!hasData()) {
		cerr << "WARN: No data in resource monitor" << endl;
		return 0;
	}

	double energy = 0;

	for(int i = 0; i < _powerGPU.size()-1; i++) {
		double time, power;

		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		power = interval_GPU_power(i);

		energy += power * time;
	}

	return energy;
}


double ResourceMonitorARM::getTotalExecTime() {

	if (!hasData()) {
		cerr << "WARN: No data in resource monitor" << endl;
		return 0;
	}

	//return timeInterval(_timestamp.front(), _timestamp.back());
	return timeInterval(_startTime, _endTime);
}


double ResourceMonitorARM::getAveragePowerCPU() {

	if (!hasData()) {
		cerr << "WARN: No data in resource monitor" << endl;
		return 0;
	}

	double avg_power = 0;
	int num_intervals = _powerCPU.size()-1;

	for (int i = 0; i < num_intervals; i++) {
		avg_power += interval_CPU_power(i);
	}

	return avg_power/num_intervals;
}



double ResourceMonitorARM::getAveragePowerGPU() {	

	if (!hasData()) {
		cerr << "WARN: No data in resource monitor" << endl;
		return 0;
	}

	double avg_power = 0;
	int num_intervals = _powerGPU.size()-1;

	for (int i = 0; i < num_intervals; i++) {
		avg_power += interval_GPU_power(i);
	}

	return avg_power/num_intervals;
}



void ResourceMonitorARM::getPowerCPU(vector<double> &timestamp, vector<double> &power) {

	timestamp.clear();
	power.clear();

	if (!hasData()) {
		cerr << "WARN: getPowerCPU not implemented" << endl;
		cerr << "WARN: No data in resource monitor" << endl;
		return;
	}

}

void ResourceMonitorARM::getPowerGPU(vector<double> &timestamp, vector<double> &power) {
	timestamp.clear();
	power.clear();

	if (!hasData()) {
		cerr << "WARN: getPowerGPU not implemented" << endl;
		cerr << "WARN: No data in resource monitor" << endl;
		return;
	}

}

/**
 * Returns the power history for CPU and GPU
 * intervalTime contains the number of second for the corresponding interval
 * powerCPU and powerGPU contain the average power for the corresponding interval
 */
void ResourceMonitorARM::getPower(vector<double> &timestamp, vector<double> &powerCPU, vector<double> &powerGPU) {

	double time;
	//double avg_power = getAveragePowerCPU();

	timestamp.clear();
	powerCPU.clear();
	powerGPU.clear();

	if (!hasData()) {
		cerr << "WARN: No data in resource monitor" << endl;
		return;
	}

	for (int i = 0; i < _timestamp.size()-1; i++) {
		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		timestamp.push_back(time);
		powerGPU.push_back(interval_GPU_power(i));
		powerCPU.push_back(interval_CPU_power(i));
	}

}



void * ResourceMonitorARM::background_monitor_handler(void * arg) {

	if (arg == NULL)
		pthread_exit(NULL);

	ResourceMonitorARM *parent = (ResourceMonitorARM*)arg;
	useconds_t wait_time = parent->measure_period;

	while(!parent->monitorMustEnd()) {
		timeval timestamp;
		double cpuPower, gpuPower;

		cerr << "Measuring" << endl;

		// Check time
		gettimeofday(&timestamp, NULL);		

		// Check CPU power
		cpuPower = parent->probeARM.fetchPowerCPU();

		// Check GPU power
		gpuPower = parent->probeARM.fetchPowerGPU();

		parent->_timestamp.push_back(timestamp);
		parent->_powerCPU.push_back(cpuPower);
		parent->_powerGPU.push_back(gpuPower);

		usleep(wait_time);
	}

	pthread_exit(NULL);
}

bool ResourceMonitorARM::monitorMustEnd() {
	bool value;
	pthread_mutex_lock(&lock);
	value = end_monitoring;
	pthread_mutex_unlock(&lock);

	return value;
}

void ResourceMonitorARM::setEndMonitor(bool value) {
	pthread_mutex_lock(&lock);
	end_monitoring = value;
	pthread_mutex_unlock(&lock);
}


double ResourceMonitorARM::timeval2double(struct timeval time) {
	return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)/(1000*1000);
}

double ResourceMonitorARM::timeInterval(struct timeval start, struct timeval end) {
		double time_start, time_end;

		time_start = static_cast<double>(start.tv_sec) + static_cast<double>(start.tv_usec)/(1000*1000);
		time_end = static_cast<double>(end.tv_sec) + static_cast<double>(end.tv_usec)/(1000*1000);
		
		return time_end - time_start;
}
