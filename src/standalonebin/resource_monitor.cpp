#include "resource_monitor.h"

#include <sys/time.h>
#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <nvml.h>



double ResourceMonitor::interval_GPU_power(int i) {
	return (_powerGPU[i] + _powerGPU[i+1]) / 2;
}


double ResourceMonitor::interval_CPU_power(int i) {
	double time = timeInterval(_timestamp[i+1], _timestamp[i]);
	double energy;

	rawEnergyCPU_t rawEnergy_start, rawEnergy_end;
	rawEnergyCPU_t rawEnergy_total;
	energyCPU_t energy_total;

	rawEnergy_start = _energyCPU[i];
	rawEnergy_end = _energyCPU[i+1];

	probeCPU.subtractRawEnergy(&rawEnergy_start, &rawEnergy_end, &rawEnergy_total);
	probeCPU.raw2jul(&rawEnergy_total, &energy_total);

	// Adding all the energy components
	// WARN: probable some of them should not be included
	energy = energy_total.package + energy_total.pp0 + energy_total.pp1 + energy_total.dram;


	return energy/time;
}

double ResourceMonitor::interval_CPU_power_est(int i, double prev_power) {
	double power;
	
	power = interval_CPU_power(i-1);

	return power*2 - prev_power;
}



ResourceMonitor::ResourceMonitor() {
	running = false;

}

ResourceMonitor::~ResourceMonitor() {

}


/**
 * Initialize the monitoring object, trying to access each relevant
 * system
 */
void ResourceMonitor::init() {

	if (running) {
		cerr << "Already running. Doing nothing." << endl;
		return;
	}

	_timestamp.clear();
	_energyCPU.clear();
	_powerGPU.clear();

	probeCPU.init();
	probeGPU.init();

}

void ResourceMonitor::startMonitoring() {

	setEndMonitor(false);
	running = true;

	gettimeofday(&_startTime, &_timeZone);
	pthread_create(&monitor_thread, NULL, background_monitor_handler, this);
}


void ResourceMonitor::endMonitoring() {

	gettimeofday(&_endTime, &_timeZone);
	setEndMonitor(true);

	pthread_join(monitor_thread, NULL);
	running = false;
}


/*
 * Return total energy consumption from the CPU in mJ
 *
 */
double ResourceMonitor::getTotalEnergyCPU() {
	// probeCPU fetches energy
	// Get the first measure, the last one, and substract them

	double energy;
	rawEnergyCPU_t rawEnergy_start;
	rawEnergyCPU_t rawEnergy_end;
	rawEnergyCPU_t rawEnergy_total;
	energyCPU_t energy_total;

	// Read start and end energy readings
	rawEnergy_start = _energyCPU.front();
	rawEnergy_end = _energyCPU.back();
	probeCPU.subtractRawEnergy(&rawEnergy_start, &rawEnergy_end, &rawEnergy_total);

	// Energy is stored as 'raw energy', which has to be converted to Jules using probeCPU.raw2jul()
	probeCPU.raw2jul(&rawEnergy_total, &energy_total);

	// Adding all the energy components
	// WARN: probable some of them should not be included
	energy = energy_total.package + energy_total.pp0 + energy_total.pp1 + energy_total.dram;

	return energy;
}


/*
 * Return total energy consumption from the GPU in mJ
 *
 */
double ResourceMonitor::getTotalEnergyGPU() {
	// probeGPU fetches power
	// Read _powerGPU and approximate the power for each interval as the mean value between the
	// starting and the ending power of the interval
	// Compute energy for each interval and sum all of them

	double energy = 0;

	for(int i = 0; i < _powerGPU.size()-1; i++) {
		double time, power;

		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		power = interval_GPU_power(i);

		energy += power * time;
	}

	return energy;
}


double ResourceMonitor::getTotalExecTime() {

	return timeInterval(_timestamp.front(), _timestamp.back());
}

double ResourceMonitor::getAveragePowerCPU() {

	double energy, time;
	time = timeInterval(_timestamp.front(), _timestamp.back());
	energy = getTotalEnergyCPU();

	return energy/time;
}

double ResourceMonitor::getAveragePowerGPU() {	

	double avg_power = 0;
	int num_intervals = _powerGPU.size()-1;

	for (int i = 0; i < num_intervals; i++) {
		avg_power = interval_GPU_power(i);
	}

	return avg_power/num_intervals;
}



void ResourceMonitor::getPowerCPU(vector<double> &timestamp, vector<double> &power) {
	
	double avg_power = getAveragePowerCPU();
	double time;

	timestamp.clear();
	power.clear();

	time = timeval2double(_timestamp[0]);
	timestamp.push_back(time);
	power.push_back(avg_power);

	for (int i = 1; i < _timestamp.size(); i++) {
		time = timeval2double(_timestamp[i]);
		timestamp.push_back(time);	
		power.push_back(interval_CPU_power_est(i, power[i-1]));
	}
}

void ResourceMonitor::getPowerGPU(vector<double> &timestamp, vector<double> &power) {

	double time;

	timestamp.clear();
	power.clear();

	for (int i = 0; i < _timestamp.size(); i++) {
		time = timeval2double(_timestamp[i]);
		timestamp.push_back(time);
		power.push_back(_powerGPU[i]);
	}
}

void ResourceMonitor::getPower(vector<double> &timestamp, vector<double> &powerCPU, vector<double> &powerGPU) {

	double time;
	double avg_power = getAveragePowerCPU();

	timestamp.clear();
	powerCPU.clear();
	powerGPU.clear();

	time = timeval2double(_timestamp[0]);
	timestamp.push_back(time);
	powerCPU.push_back(avg_power);
	powerGPU.push_back(_powerGPU[0]);
	
	for (int i = 1; i < timestamp.size(); i++) {
		time = timeval2double(_timestamp[i]);
		timestamp.push_back(time);
		powerGPU.push_back(_powerGPU[i]);
		powerCPU.push_back(interval_CPU_power_est(i, powerCPU[i-1]));
	}

}



void * ResourceMonitor::background_monitor_handler(void * arg) {

	if (arg == NULL)
		pthread_exit(NULL);

	ResourceMonitor *parent = (ResourceMonitor*)arg;

	while(parent->checkEndMonitor()) {
		// Check time
		// Check CPU energy
		// Check GPU energy

		usleep(0.5*1000000); // 1/2 second
	}

	pthread_exit(NULL);
}

bool ResourceMonitor::checkEndMonitor() {
	bool value;
	pthread_mutex_lock(&lock);
	value = end_monitoring;
	pthread_mutex_unlock(&lock);

	return value;
}

void ResourceMonitor::setEndMonitor(bool value) {
	pthread_mutex_lock(&lock);
	end_monitoring = value;
	pthread_mutex_unlock(&lock);
}


double ResourceMonitor::timeval2double(timeval &time) {
	return (double)time.tv_sec + (double)time.tv_usec/(1000*1000);
}

double ResourceMonitor::timeInterval(struct timeval start, struct timeval end) {
		double time_start, time_end;

		time_start = (double)start.tv_sec + (double)start.tv_usec/(1000*1000);
		time_end = (double)end.tv_sec + (double)end.tv_usec/(1000*1000);
		
		return time_end - time_start;
}
