#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "resource_monitor.h"



using namespace std;

int main(int argc, char **argv) {


	ResourceMonitor monitor;
	monitor.init();

	cerr << "Start monitoring" << endl;
	monitor.startMonitoring();
	cerr << "Start sleep 5" << endl;
	sleep(5);
	cerr << "End sleep" << endl;
	monitor.endMonitoring();
	cerr << "Monitoring finished" << endl;

	double execTime, cpuEnergy, cpuPower;
	double gpuEnergy, gpuPower;

	execTime = monitor.getTotalExecTime();
	cpuEnergy = monitor.getTotalEnergyCPU();
	cpuPower = monitor.getAveragePowerCPU();
	gpuEnergy = monitor.getTotalEnergyGPU();
	gpuPower = monitor.getAveragePowerGPU();

	vector<double> timestamp;
	vector<double> cpuPowerReading;
	vector<double> gpuPowerReading;

	monitor.getPower(timestamp, cpuPowerReading, gpuPowerReading);

	cout << "RESULTS" << endl << "=========================" << endl << endl;

	cout << "interval time (s), cpu power (W), gpu power (W)" << endl;
	cout << "--------------------------------------------------------" << endl;
	for ( int i = 0; i < timestamp.size(); i++) {
		cout << setw(15) << timestamp[i] << ", " << setw(12) << gpuPowerReading[i] << ", " << setw(12) << gpuPowerReading[i];
		cout << endl;
	} 
	cout << endl;


	cout << "  time (s), cpu energy (J), cpu power (J), gpu energy (W), gpu power (W)" << endl;
	cout << "--------------------------------------------------------" << endl;
	cout << setw(11) << execTime << ", " << setw(11) << cpuEnergy;
	cout << ", " << setw(11) << cpuPower << ", " << setw(11) << gpuEnergy;
	cout << ", " << setw(11) << gpuPower << endl << endl;
	


  return 0;
}
