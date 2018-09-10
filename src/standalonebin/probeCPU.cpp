#include "probeCPU.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


ProbeCPU::ProbeCPU() {
	fd = 0;

}


ProbeCPU::~ProbeCPU() {
	close(fd);

}



int ProbeCPU::open_msr(int core) {

  char msr_filename[BUFSIZ];
  int fd;

  sprintf(msr_filename, "/dev/cpu/%d/msr", core);
  fd = open(msr_filename, O_RDONLY);
  if ( fd < 0 ) {
    if ( errno == ENXIO ) {
      fprintf(stderr, "rdmsr: No CPU %d\n", core);
      exit(2);
    } else if ( errno == EIO ) {
      fprintf(stderr, "rdmsr: CPU %d doesn't support MSRs\n", core);
      exit(3);
    } else {
      perror("rdmsr:open");
      fprintf(stderr,"Trying to open %s\n",msr_filename);
      exit(127);
    }
  }

  return fd;
}



uint64_t ProbeCPU::read_msr(int fd, int which) {

  uint64_t data;

  if ( pread(fd, &data, sizeof data, which) != sizeof data ) {
    perror("rdmsr:pread");
    exit(127);
  }

  return data;
}


int ProbeCPU::detect_cpu(void) {

	FILE *fff;

	int family,model=-1;
	char buffer[BUFSIZ],*result;
	char vendor[BUFSIZ];

	fff=fopen("/proc/cpuinfo","r");
	if (fff==NULL) return -1;

	while(1) {
		result=fgets(buffer,BUFSIZ,fff);
		if (result==NULL) break;

		if (!strncmp(result,"vendor_id",8)) {
			sscanf(result,"%*s%*s%s",vendor);

			if (strncmp(vendor,"GenuineIntel",12)) {
				printf("%s not an Intel chip\n",vendor);
				return -1;
			}
		}

		if (!strncmp(result,"cpu family",10)) {
			sscanf(result,"%*s%*s%*s%d",&family);
			if (family!=6) {
				printf("Wrong CPU family %d\n",family);
				return -1;
			}
		}

		if (!strncmp(result,"model",5)) {
			sscanf(result,"%*s%*s%d",&model);
		}

	}

	fclose(fff);

	switch(model) {
		case CPU_SANDYBRIDGE:
			printf("Found Sandybridge CPU\n");
			break;
		case CPU_SANDYBRIDGE_EP:
			printf("Found Sandybridge-EP CPU\n");
			break;
		case CPU_IVYBRIDGE:
			printf("Found Ivybridge CPU\n");
			break;
		case CPU_IVYBRIDGE_EP:
			printf("Found Ivybridge-EP CPU\n");
			break;
		case CPU_HASWELL:
			printf("Found Haswell CPU\n");
			break;
		case CPU_HASWELL_EP:
			printf("Found Haswell-EP CPU\n");
			break;
		case CPU_BROADWELL:
			printf("Found Broadwell CPU\n");
			break;
		default:	printf("Unsupported model %d\n",model);
				model=-1;
				break;
	}

	return model;
}


void ProbeCPU::init() {
	
	uint64_t result;

	cpu_model=detect_cpu();
	if (cpu_model<0) 
		printf("Unsupported CPU type\n");
		

	printf("Checking core #%d\n",core);

	fd=open_msr(core);

	/* Calculate the units used */
	result=read_msr(fd,MSR_RAPL_POWER_UNIT);

	power_units=pow(0.5,(double)(result&0xf));
	cpu_energy_units=pow(0.5,(double)((result>>8)&0x1f));
	time_units=pow(0.5,(double)((result>>16)&0xf));

	/* On Haswell EP the DRAM units differ from the CPU ones */
	if (cpu_model==CPU_HASWELL_EP) {
		dram_energy_units=pow(0.5,(double)16);
	}
	else {
		dram_energy_units=cpu_energy_units;
	}

	printf("Power units = %.3fW\n",power_units);
	printf("CPU Energy units = %.8fJ\n",cpu_energy_units);
	printf("DRAM Energy units = %.8fJ\n",dram_energy_units);
	printf("Time units = %.8fs\n",time_units);
	printf("\n");

	/* Show package power info */
	result=read_msr(fd,MSR_PKG_POWER_INFO);
	thermal_spec_power=power_units*(double)(result&0x7fff);
	printf("Package thermal spec: %.3fW\n",thermal_spec_power);
	minimum_power=power_units*(double)((result>>16)&0x7fff);
	printf("Package minimum power: %.3fW\n",minimum_power);
	maximum_power=power_units*(double)((result>>32)&0x7fff);
	printf("Package maximum power: %.3fW\n",maximum_power);
	time_window=time_units*(double)((result>>48)&0x7fff);
	printf("Package maximum time window: %.6fs\n",time_window);

	/* Show package power limit */
	result=read_msr(fd,MSR_PKG_RAPL_POWER_LIMIT);
	printf("Package power limits are %s\n", (result >> 63) ? "locked" : "unlocked");
	double pkg_power_limit_1 = power_units*(double)((result>>0)&0x7FFF);
	double pkg_time_window_1 = time_units*(double)((result>>17)&0x007F);
	printf("Package power limit #1: %.3fW for %.6fs (%s, %s)\n",
		pkg_power_limit_1, pkg_time_window_1,
		(result & (1LL<<15)) ? "enabled" : "disabled",
		(result & (1LL<<16)) ? "clamped" : "not_clamped");
	double pkg_power_limit_2 = power_units*(double)((result>>32)&0x7FFF);
	double pkg_time_window_2 = time_units*(double)((result>>49)&0x007F);
	printf("Package power limit #2: %.3fW for %.6fs (%s, %s)\n", 
		pkg_power_limit_2, pkg_time_window_2,
		(result & (1LL<<47)) ? "enabled" : "disabled",
		(result & (1LL<<48)) ? "clamped" : "not_clamped");

	printf("\n");
}


void ProbeCPU::fetchRawEnergy(rawEnergyCPU_t *rawEnergy) {

	uint64_t result;

	rawEnergy->package = 0;
	rawEnergy->pp0 = 0;
	rawEnergy->pp1 = 0;
	rawEnergy->dram = 0;

	result=read_msr(fd,MSR_PKG_ENERGY_STATUS);
	rawEnergy->package = result;

	result=read_msr(fd,MSR_PP0_ENERGY_STATUS);
	rawEnergy->pp0 = result;


 	result=read_msr(fd,MSR_PP1_ENERGY_STATUS);
	rawEnergy->pp1 = result;

	/* Updated documentation (but not the Vol3B) says Haswell and	*/
	/* Broadwell have DRAM support too				*/
	if ((cpu_model==CPU_SANDYBRIDGE_EP) || (cpu_model==CPU_IVYBRIDGE_EP) || (cpu_model==CPU_HASWELL_EP) ||
		(cpu_model==CPU_HASWELL) || (cpu_model==CPU_BROADWELL)) {

		result=read_msr(fd,MSR_DRAM_ENERGY_STATUS);
		rawEnergy->dram = result;
	}


}



void ProbeCPU::raw2jul(rawEnergyCPU_t *rawEnergy, energyCPU_t *energy) {

	energy->package = (double)rawEnergy->package * cpu_energy_units;
	energy->pp0 = (double)rawEnergy->pp0 * cpu_energy_units;
	energy->pp1 = (double)rawEnergy->pp1 * cpu_energy_units;
	energy->dram = (double)rawEnergy->dram * dram_energy_units;
}



#define OVERFLOW_SUB(start, end, result, max) \
do { \
	if ((start) <= (end)) result = (end) - (start); \
	else result = ((max) - (start)) + end; \
} while(0)


void ProbeCPU::subtractRawEnergy(rawEnergyCPU_t * start, rawEnergyCPU_t *end, rawEnergyCPU_t *result) {

	OVERFLOW_SUB(start->package, end->package, result->package, UINT64_MAX);
	OVERFLOW_SUB(start->pp0, end->pp0, result->pp0, UINT64_MAX);
	OVERFLOW_SUB(start->pp1, end->pp1, result->pp1, UINT64_MAX);
	OVERFLOW_SUB(start->dram, end->dram, result->dram, UINT64_MAX);
}
