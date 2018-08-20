#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/kaldi-lattice.h"

#include "lat/lattice-functions.h"
#include "base/timer.h"
#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "accel_types.h"

int main(int argc, char **argv) {

	using namespace kaldi;
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;
	using fst::VectorFst;
	using fst::StdArc;
	using fst::ReadFstKaldi;


	const char *usage = 
			"Converts a lattice from KALDI_SIM accelerator output format to kaldi's lattice format."
			"usage: aclattice2lattice [options] <utterance-name> <aclattice-filename> <lattice-wspecifier";

	ParseOptions po(usage);
	std::string time_log = "";

	po.Register("time-log", &time_log, "File to store time measurements");

	po.Read(argc, argv);

	if (po.NumArgs() != 3) {
		po.PrintUsage();
		exit(1);
	}


	std::string utterance_name = po.GetArg(1),
							aclattice_filename = po.GetArg(1),
							lattice_wspecifier = po.GetArg(3);


	Timer convert_timer;
	
	LatticeWriter lattice_writer(lattice_wspecifier);

	std::ofstream time_o;
	if ( time_log != "" ) {
		time_o.open(time_log);
		if ( !time_o.is_open() ) {
			KALDI_WARN << "Could not open time log file " << time_log;
		} 
		else {
			time_o << "Utterance, time (s)" << std::endl;
		}
	}
	

	vector<Token> tokens;
	Lattice lat;
	uint16_t last_frame = 0;
	vector<unordered_map<addr_t, int> > token_map;

	// This hash table list maps from fst state IDs to token IDs of the final lattice
	// Needed to create the arcs
	//unordered_map<addr_t, int> state_to_token(total_frames);

	ifstream aclattice_reader(aclattice_filename, ios::binary);
	
	vector<Token>::size_type num_tokens;
	aclattice_reader.read((char*)&num_tokens, sizeof(num_tokens));

	cerr << "Num tokens: " << num_tokens << endl;

	for (int i = 0; i < num_tokens; i++) {
		Token new_token;
		aclattice_reader.read((char*)&new_token, sizeof(new_token));
		tokens.push_back(new_token);

		if (tokens[i].frame > last_frame) 
			last_frame = tokes[i].frame;
	}


	token_map.resize(last_frame+1);
	convert_timer.Reset();		

/*
	// Convert from vector<Token> to compactLattice

	// for each token
	//	hash[frame].at(tok.orig_state) = lat.addState();
	//	if (frame == last_frame)
	//		hash[frame+1].at(tok.dst_state) = lat.addState();
	lat.SetStart(0)

	// for each token
	//	orig_tok = hash[frame].at(tok.orig_state);
	//	dest_tok = hash[frame+1].at(tok.dest:state);
	//  Arc arc(tok.wordid, tok.wordid, Weight(tok.graph_cost, tok.ac_cost), dst_tok);
	//	lat.addArc(orig_tok, arc);

*/

	double convert_elapsed = convert_timer.Elapsed();


	lattice_writer.Write(utterance_name, lat);



	if ( time_o.is_open() ) {
		time_o << utterance_name << ", " << convert_elapsed;
		time_o << std::endl;
	}



	time_o.close();

	return 0;
}
