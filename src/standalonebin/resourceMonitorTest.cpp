#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "resource_monitor_threaded.h"



using namespace std;

int main(int argc, char **argv) {


	ResourceMonitor monitor;
  ResourceMonitor monitor2;

	cerr << "Start monitoring" << endl;
	//monitor.startMonitoring(0.5);
  monitor.asyncDataFetch();
  monitor2.asyncDataFetch();
  //monitor2.startMonitoring(0.0001);
	//cerr << "Start sleep 5" << endl;
	//sleep(5);
	//cerr << "End sleep" << endl;
  //monitor2.endMonitoring();
  monitor2.asyncDataFetch();
  monitor.asyncDataFetch();
	//monitor.endMonitoring();
	cerr << "Monitoring finished" << endl;

	double execTime, cpuEnergy, cpuPower;
	double gpuEnergy, gpuPower;


	double execTime2, cpuEnergy2, cpuPower2;
	double gpuEnergy2, gpuPower2;

	execTime = monitor.getTotalExecTime();
	cpuEnergy = monitor.getTotalEnergyCPU();
	cpuPower = monitor.getAveragePowerCPU();
	gpuEnergy = monitor.getTotalEnergyGPU();
	gpuPower = monitor.getAveragePowerGPU();

	execTime2 = monitor2.getTotalExecTime();
	cpuEnergy2 = monitor2.getTotalEnergyCPU();
	cpuPower2 = monitor2.getAveragePowerCPU();
	gpuEnergy2 = monitor2.getTotalEnergyGPU();
	gpuPower2 = monitor2.getAveragePowerGPU();

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


	cout << "  time2 (s), cpu energy2 (J), cpu power2 (J), gpu energy2 (W), gpu power2 (W)" << endl;
	cout << "--------------------------------------------------------" << endl;
	cout << setw(11) << execTime2 << ", " << setw(11) << cpuEnergy2;
	cout << ", " << setw(11) << cpuPower2 << ", " << setw(11) << gpuEnergy2;
	cout << ", " << setw(11) << gpuPower2 << endl << endl;

	cout << "  time (s), cpu energy (J), cpu power (J), gpu energy (W), gpu power (W)" << endl;
	cout << "--------------------------------------------------------" << endl;
	cout << setw(11) << execTime << ", " << setw(11) << cpuEnergy;
	cout << ", " << setw(11) << cpuPower << ", " << setw(11) << gpuEnergy;
	cout << ", " << setw(11) << gpuPower << endl << endl;
	


  return 0;
}
