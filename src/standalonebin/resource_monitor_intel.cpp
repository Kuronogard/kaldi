#include "resource_monitor_intel.h"
#include <sys/time.h>
#include <iostream>
#include <unistd.h>
#include <nvml.h>
#include <string>


void ResourceMonitorIntel::printInfo(std::string msg) {
  if (verbose_) {
    std::cerr << msg << std::endl;
  }
}


int ResourceMonitorIntel::numData() {
	return _timestamp.size();
}

bool ResourceMonitorIntel::hasData() {
	return _timestamp.size() > 0;
}


void ResourceMonitorIntel::clearData() {
  _timestamp.clear();
  _powerGPU.clear();
  _energyCPU.clear();
}

double ResourceMonitorIntel::interval_GPU_power(int i) {
	return (_powerGPU[i] + _powerGPU[i+1]) / 2;
}


double ResourceMonitorIntel::interval_CPU_power(int i) {
	double time = timeInterval(_timestamp[i], _timestamp[i+1]);
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

double ResourceMonitorIntel::interval_CPU_power_est(int i, double prev_power) {
	double power;
	
	power = interval_CPU_power(i-1);

	return power*2 - prev_power;
}



ResourceMonitorIntel::ResourceMonitorIntel(bool verbose) {
  verbose_ = verbose;
  clearData();

  probeCPU.init();
  probeGPU.init();
}


ResourceMonitorIntel::~ResourceMonitorIntel() {}


/*
 * Return total energy consumption from the CPU in mJ
 *
 */
double ResourceMonitorIntel::getTotalEnergyCPU() {
	// probeCPU fetches energy
	// Get the first measure, the last one, and substract them
	double energy = 0;
	rawEnergyCPU_t rawEnergy_start;
	rawEnergyCPU_t rawEnergy_end;
	rawEnergyCPU_t rawEnergy_total;
	energyCPU_t energy_total;

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return 0;
	}

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
double ResourceMonitorIntel::getTotalEnergyGPU() {
	// probeGPU fetches power
	// Read _powerGPU and approximate the power for each interval as the mean value between the
	// starting and the ending power of the interval
	// Compute energy for each interval and sum all of them

	double energy = 0;

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return 0;
	}

	for(int i = 0; i < _powerGPU.size()-1; i++) {
		double time, power;

		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		power = interval_GPU_power(i);

		energy += power * time;
	}

	return energy;
}


double ResourceMonitorIntel::getTotalExecTime() {

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return 0;
	}

	return timeInterval(_timestamp.front(), _timestamp.back());
	//return timeInterval(_startTime, _endTime);
}


double ResourceMonitorIntel::getAveragePowerCPU() {

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return 0;
	}

	double energy, time;
	time = getTotalExecTime();
	energy = getTotalEnergyCPU();

	return energy/time;
}

double ResourceMonitorIntel::getAveragePowerGPU() {	

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



void ResourceMonitorIntel::getPowerCPU(vector<double> &timestamp, vector<double> &power) {
	
	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return;
	}

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

void ResourceMonitorIntel::getPowerGPU(vector<double> &timestamp, vector<double> &power) {

	double time;

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return;
	}

	timestamp.clear();
	power.clear();

	for (int i = 0; i < _timestamp.size(); i++) {
		time = timeval2double(_timestamp[i]);
		timestamp.push_back(time);
		power.push_back(_powerGPU[i]);
	}
}

/**
 * Returns the power history for CPU and GPU
 * intervalTime contains the number of second for the corresponding interval
 * powerCPU and powerGPU contain the average power for the corresponding interval
 */
void ResourceMonitorIntel::getPower(vector<double> &timestamp, vector<double> &powerCPU, vector<double> &powerGPU) {

	double time;
	//double avg_power = getAveragePowerCPU();

	if (numData() < 2) {
		cerr << "WARN: You need at least two measurements." << endl;
		return;
	}

	timestamp.clear();
	powerCPU.clear();
	powerGPU.clear();

	//time = 0;
	//timestamp.push_back(time);
	//powerCPU.push_back(avg_power);
	//powerGPU.push_back(_powerGPU[0]);

	//cerr << "Measured " << _timestamp.size() << " times" << endl;
	
	for (int i = 0; i < _timestamp.size()-1; i++) {
		time = timeInterval(_timestamp[i], _timestamp[i+1]);
		timestamp.push_back(time);
		powerGPU.push_back(interval_GPU_power(i));
		powerCPU.push_back(interval_CPU_power(i));

		//time = timeInterval(_startTime, _timestamp[i]);
		//timestamp.push_back(time);
		//powerGPU.push_back(_powerGPU[i]);
		//powerCPU.push_back(interval_CPU_power_est(i, powerCPU[i-1]));
	}

}


void ResourceMonitorIntel::asyncDataFetch() {
  timeval timestamp;
  rawEnergyCPU_t cpuRawEnergy;
  rawPowerGPU_t gpuRawPower;
  powerGPU_t gpuPower;

  //cerr << "Measuring" << endl;
  // Check time
  gettimeofday(&timestamp, NULL);		

  // Check CPU energy
  probeCPU.fetchRawEnergy(&cpuRawEnergy);

  // Check GPU energy
  probeGPU.fetchRawPower(&gpuRawPower);
  probeGPU.raw2watt(&gpuRawPower, &gpuPower);

  _timestamp.push_back(timestamp);
  printInfo("BEFORE push_back");
  _energyCPU.push_back(cpuRawEnergy);
  printInfo("AFTER push_back");
  _powerGPU.push_back(gpuPower.power);
}



double ResourceMonitorIntel::timeval2double(struct timeval time) {
	return static_cast<double>(time.tv_sec) + static_cast<double>(time.tv_usec)/(1000*1000);
}

double ResourceMonitorIntel::timeInterval(struct timeval start, struct timeval end) {
		double time_start, time_end;

		time_start = static_cast<double>(start.tv_sec) + static_cast<double>(start.tv_usec)/(1000*1000);
		time_end = static_cast<double>(end.tv_sec) + static_cast<double>(end.tv_usec)/(1000*1000);
		
		return time_end - time_start;
}
