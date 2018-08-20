#ifndef __GLOBAL_H
#define __GLOBAL_H

#include <cstdint>
#include <vector>

using namespace std;

using addr_t = std::uint64_t;

struct Token {
	float graph_cost;
	float ac_cost;
	addr_t orig_state;
	addr_t dst_state;
	uint16_t frame;
	unsigned int wordid; // Output label of best incoming arc
	int blink;

	inline Token() :
		graph_cost(0), ac_cost(0), orig_state(0), dst_state(0), frame(0), wordid(0){}
};

#endif
