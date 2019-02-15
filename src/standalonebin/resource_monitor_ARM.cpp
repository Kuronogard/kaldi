#include "resource_monitor_ARM.h"

#include <sys/time.h>
#include <iostream>
#include <string>
#include <pthread.h>
#include <unistd.h>


void ResourceMonitorARM::printInfo(std::string msg) {
  if (verbose_) {
    std::cerr << msg << std::endl;
  }
}


int ResourceMonitorARM::numData() {
	return _timestamp.size();
}

bool ResourceMonitorARM::hasData() {
	return _timestamp.size() > 0;
}


void ResourceMonitorARM::clearData() {
  _timestamp.clear();
  _powerCPU.clear();
  _powerGPU.clear();
}


double ResourceMonitorARM::interval_GPU_power(int i) {
	return (_powerGPU[i] + _powerGPU[i+1]) / 2;
}


double ResourceMonitorARM::interval_CPU_power(int i) {
	return (_powerCPU[i] + _powerCPU[i+1]) / 2;
}



ResourceMonitorARM::ResourceMonitorARM(bool verbose) {
  verbose_ = verbose;
  clearData();

  probeARM.init();
}


ResourceMonitorARM::~ResourceMonitorARM() {}


void ResourceMonitorARM::asyncDataFetch() {

  timeval timestamp;
  double cpuPower, gpuPower;

  gettimeofday(&timestamp, NULL);
  cpuPower = probeARM.fetchPowerCPU();
  gpuPower = probeARM.fetchPowerGPU();

  _timestamp.push_back(timestamp);
  _powerCPU.push_back(cpuPower);
  _powerGPU.push_back(gpuPower);
}


/*
 * Return total energy consumption from the CPU in mJ
 *
 */
double ResourceMonitorARM::getTotalEnergyCPU() {

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
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

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
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

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return 0;
	}

	return timeInterval(_timestamp.front(), _timestamp.back());
	//return timeInterval(_startTime, _endTime);
}


double ResourceMonitorARM::getAveragePowerCPU() {

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
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

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
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

  cerr << "WARN: getPowerCPU not implemented" << endl;

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return;
	}
}

void ResourceMonitorARM::getPowerGPU(vector<double> &timestamp, vector<double> &power) {
	timestamp.clear();
	power.clear();

  cerr << "WARN: getPowerGPU not implemented" << endl;

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
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

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return;
	}

	for (int i = 0; i < _timestamp.size()-1; i++) {
		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		timestamp.push_back(time);
		powerGPU.push_back(interval_GPU_power(i));
		powerCPU.push_back(interval_CPU_power(i));
	}

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
