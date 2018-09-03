#ifndef __GLOBAL_H
#define __GLOBAL_H

#include <cstdint>
#include <vector>
#include <limits>

using namespace std;

using addr_t = std::uint64_t;


struct Token {
	bool epsilon;
	float graph_cost;
	float ac_cost;
	float final_cost;
	addr_t orig_state;
	addr_t dst_state;
	uint16_t frame;
	unsigned int wordid; // Output label of best incoming arc
	unsigned int pdf;
	int blink;

	inline Token() :
		epsilon(0), graph_cost(0), ac_cost(0), final_cost(std::numeric_limits<float>::infinity()), orig_state(0), dst_state(0), frame(0), wordid(0){}
};

#endif
