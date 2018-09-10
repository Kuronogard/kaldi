#include <stdlib.h>
#include <nvml.h>


class ProbeGPU {

public:

	ProbeGPU();
	~ProbeGPU();

	void init();
	void fetchPower(unsigned int *power);

private:

	nvmlDevice_t device;

};
