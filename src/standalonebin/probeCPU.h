

/* Hamid: rapl-read section starts  ------------------------------ 
---------------------------------------------------------------- */

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

#include <sys/syscall.h>
#include <linux/perf_event.h>

#define MSR_RAPL_POWER_UNIT		0x606

/*
 * Platform specific RAPL Domains.
 * Note that PP1 RAPL Domain is supported on 062A only
 * And DRAM RAPL Domain is supported on 062D only
 */
/* Package RAPL Domain */
#define MSR_PKG_RAPL_POWER_LIMIT	0x610
#define MSR_PKG_ENERGY_STATUS		0x611
#define MSR_PKG_PERF_STATUS		0x613
#define MSR_PKG_POWER_INFO		0x614

/* PP0 RAPL Domain */
#define MSR_PP0_POWER_LIMIT		0x638
#define MSR_PP0_ENERGY_STATUS		0x639
#define MSR_PP0_POLICY			0x63A
#define MSR_PP0_PERF_STATUS		0x63B

/* PP1 RAPL Domain, may reflect to uncore devices */
#define MSR_PP1_POWER_LIMIT		0x640
#define MSR_PP1_ENERGY_STATUS		0x641
#define MSR_PP1_POLICY			0x642

/* DRAM RAPL Domain */
#define MSR_DRAM_POWER_LIMIT		0x618
#define MSR_DRAM_ENERGY_STATUS		0x619
#define MSR_DRAM_PERF_STATUS		0x61B
#define MSR_DRAM_POWER_INFO		0x61C

/* RAPL UNIT BITMASK */
#define POWER_UNIT_OFFSET	0
#define POWER_UNIT_MASK		0x0F

#define ENERGY_UNIT_OFFSET	0x08
#define ENERGY_UNIT_MASK	0x1F00

#define TIME_UNIT_OFFSET	0x10
#define TIME_UNIT_MASK		0xF000

#define CPU_SANDYBRIDGE		42
#define CPU_SANDYBRIDGE_EP	45
#define CPU_IVYBRIDGE		58
#define CPU_IVYBRIDGE_EP	62
#define CPU_HASWELL		60
#define CPU_HASWELL_EP		63
#define CPU_BROADWELL		61



struct rawEnergyCPU_t {
	long long package;
	long long pp0;
	long long pp1;
	long long dram;
};


struct energyCPU_t {
	double package;
	double pp0;
	double pp1;
	double dram;
};


class ProbeCPU {

public:

	ProbeCPU();
	~ProbeCPU();

	void init();
	void fetchRawEnergy(rawEnergyCPU_t *rawEnergy);
	void raw2jul(rawEnergyCPU_t *rawEnergy, energyCPU_t *energy);
	
	static void subtractRawEnergy(rawEnergyCPU_t *start, rawEnergyCPU_t *end, rawEnergyCPU_t *result);


private:

	int open_msr(int core);
	uint64_t read_msr(int fd, int which);
	int detect_cpu(void);	

	int fd;
	double power_units,time_units;
	double cpu_energy_units,dram_energy_units;
	double package_before,package_after;
	double thermal_spec_power,minimum_power,maximum_power,time_window;
	int cpu_model;
  int core = 0;
	
};
