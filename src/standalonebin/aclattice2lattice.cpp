#include <iostream>
#include <fstream>
#include <limits>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

#include "decoder/decoder-wrappers.h"
#include "hmm/transition-model.h"
#include "lat/lattice-functions.h"
#include "base/timer.h"
#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "accel_types.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"


class tok_map_elem {
public:
	int token;
	int state;

	tok_map_elem() : token(0), state(0) {}
	tok_map_elem(int tok, int st) : token(tok), state(st) {}
};


int main(int argc, char **argv) {

	using namespace kaldi;
	using namespace kaldi::nnet3;
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;
	using fst::SymbolTable;
	using fst::VectorFst;
	using fst::StdArc;
	using fst::ReadFstKaldi;
	using fst::Fst;

	typedef LatticeArc Arc;
	typedef Arc::Weight Weight;

	const char *usage = 
			"Converts a lattice from KALDI_SIM accelerator output format to kaldi's lattice format."
			"usage: aclattice2lattice [options] <utterance-name> <AM-nnet-filename> <aclattice-filename> <transcription-wspecifier> <lattice-wspecifier>";

	ParseOptions po(usage);
	std::string symbol_table = "";
	std::string time_log = "";
	BaseFloat acoustic_scale = 0.1;

	po.Register("time-log", &time_log, "File to store time measurements");
	po.Register("acoustic-scale", &acoustic_scale, "Acoustic scale applied to lattice before computing best path");
	po.Register("symbol-table", &symbol_table, "Symbol table. If provided, the transcription will be shown on standard output");

	po.Read(argc, argv);

	if (po.NumArgs() != 5) {
		po.PrintUsage();
		exit(1);
	}


	std::string utterance_name = po.GetArg(1),
							am_nnet_filename = po.GetArg(2),
							aclattice_filename = po.GetArg(3),
							transcription_wspecifier = po.GetArg(4),
							lattice_wspecifier = po.GetArg(5);


	TransitionModel trans_model;
	{
		bool binary;
		Input ki(am_nnet_filename, &binary);
		trans_model.Read(ki.Stream(), binary);
	}



	Timer convert_timer;
	
	CompactLatticeWriter clattice_writer(lattice_wspecifier);
	//LatticeWriter lattice_writer(lattice_wspecifier);


	fst::SymbolTable *word_symbols = 0;
	if (symbol_table != "") {
		if (!(word_symbols = fst::SymbolTable::ReadText(symbol_table)) ) {
			cerr << "Could not open symbol table " << symbol_table << endl;
		}
	}


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
	


	Int32VectorWriter trans_writer(transcription_wspecifier);


	vector<Token> tokens;
	Lattice lat;
	uint16_t last_frame = 0;
	unsigned int last_pdf = 0;
	vector<unordered_map<addr_t, tok_map_elem> > token_map;

	// This hash table list maps from fst state IDs to token IDs of the final lattice
	// Needed to create the arcs
	//unordered_map<addr_t, int> state_to_token(total_frames);

	ifstream aclattice_reader(aclattice_filename, ios::binary);
	if (!aclattice_reader.is_open()) {
		cerr << "Could not open aclattice " << aclattice_filename << endl;
		return 0;
	}

	vector<Token>::size_type num_tokens = 0;
	aclattice_reader.read((char*)&num_tokens, sizeof(vector<Token>::size_type));

	cerr << "Num tokens: " << num_tokens << endl;

	for (int i = 0; i < num_tokens; i++) {
		Token new_token;
		aclattice_reader.read((char*)&new_token, sizeof(new_token));
		tokens.push_back(new_token);

		if (new_token.orig_state == new_token.dst_state) {
			if (new_token.wordid != 0) {
				cerr << "Self connection With WORD !!" << endl;
			}
		}

		if (tokens[i].frame > last_frame) {
			last_frame = tokens[i].frame;
		}

		if (i > 0) {
			if ((new_token.pdf != std::numeric_limits<unsigned int>::max()) && (new_token.pdf > last_pdf)){
				last_pdf = new_token.pdf;
			}
		}

	}
	
	token_map.resize(last_frame+2);
	convert_timer.Reset();		


	if (last_pdf == std::numeric_limits<unsigned int>::max()) {
		cerr << "Too much pdfs..." << endl;
		return 0;
	}

	// Convert from vector<Token> to compactLattice

	// for each token
	//	hash[frame].at(tok.orig_state) = lat.addState();
	//	if (frame == last_frame)
	//		hash[frame+1].at(tok.dst_state) = lat.addState();
	//lat.SetStart(0)
	int added_states = 0;

	ofstream tok_debug("tokenlist.txt");

	// Frame 0 should only have the start state
	tok_map_elem start_tok;
	start_tok.token = lat.AddState();
	start_tok.state = 0;
	token_map[0][0] = start_tok;
	lat.SetStart(0);

	// Add dst state for all the tokens
	for(int i = 0; i < tokens.size(); i++) {
		tok_debug << "TOKEN: " << i << " frame: "  << tokens[i].frame;
		tok_debug << " link: " << tokens[i].orig_state << " --> " << tokens[i].dst_state;
		tok_debug << " (pdf: " << tokens[i].pdf << " word: " << tokens[i].wordid << ")";
		tok_debug << " graphweight: " << tokens[i].graph_cost;
		tok_debug << " blink: " << tokens[i].blink;
		tok_debug << " EPS: " << tokens[i].epsilon;
		tok_debug << " Final: " << tokens[i].final_cost;
		tok_debug << std::endl;

		int dstframe;

		// Epsilon arc do not consume frames
		if (tokens[i].epsilon) 
			dstframe = tokens[i].frame;
		else
			dstframe = tokens[i].frame+1;

		tok_map_elem tok_elem;

		// If the state does not exist, create it
		if (token_map[dstframe].count(tokens[i].dst_state) == 0) {
			tok_elem.token = lat.AddState();
			tok_elem.state = tokens[i].dst_state;

			//cerr << "Added state (" << dstframe << ") " << tok_elem.token << endl;
			token_map[dstframe][tokens[i].dst_state] = tok_elem;
			added_states++;
		} else if (token_map[dstframe][tokens[i].dst_state].state != tokens[i].dst_state) {
			cerr << "Hash collision... !!" << endl;
		} else {
			tok_elem = token_map[dstframe][tokens[i].dst_state];
		}
	
		// If the token corresponds to the last frame, set 
		// dst_state as final
		if (tokens[i].frame == last_frame) {
			lat.SetFinal(tok_elem.token, LatticeWeight(tokens[i].final_cost, 0));
		}

	}

	tok_debug.close();
	//cout << "Added " << added_states << " states." << endl;

	// for each token
	//	orig_tok = hash[frame].at(tok.orig_state);
	//	dest_tok = hash[frame+1].at(tok.dest:state);
	//  Arc arc(tok.wordid, tok.wordid, Weight(tok.graph_cost, tok.ac_cost), dst_tok);
	//	lat.addArc(orig_tok, arc);
	int added_arcs = 0;

	for(int i = 0; i < tokens.size(); i++) {
		// Sanity check
		int origframe = tokens[i].frame;
		int dstframe;
		if (tokens[i].epsilon)
			dstframe = tokens[i].frame;
		else
			dstframe = tokens[i].frame+1;


		if (token_map[origframe].count(tokens[i].orig_state) != 1) {
			cerr << "Origin state " << origframe << ":" << tokens[i].orig_state << " for token " << i << " is bad." << endl;
			cerr << "Count returned " << token_map[origframe].count(tokens[i].orig_state) << endl;
			return 0;
		}

		if (token_map[dstframe].count(tokens[i].dst_state) != 1) {
			cerr << "Destination state " << dstframe << ":" << tokens[i].dst_state << " for token " << i << " is bad." << endl;
			cerr << "Count returned " << token_map[dstframe].count(tokens[i].orig_state) << endl;	
			return 0;
		}

		int orig_tok = token_map[origframe][tokens[i].orig_state].token;
		int dst_tok = token_map[dstframe][tokens[i].dst_state].token;

		// epsilons are encoded as '0' in phone space. As i am not translating pdfs to phones, 
		// but instead working directly with pdfs, i have to change pdf 0 to a different ID, and 
		// pdf -1 to 0 (epsilon)
		unsigned int pdf;
/*
		if (tokens[i].pdf == 0) {
			pdf = last_pdf+1;
		} else if (tokens[i].pdf == std::numeric_limits<unsigned int>::max()) {
			pdf = 0;
		} else {
			pdf = tokens[i].pdf;
		}
*/
		pdf = tokens[i].pdf;

		Arc arc(pdf, tokens[i].wordid, Weight(tokens[i].graph_cost, tokens[i].ac_cost), dst_tok);
		lat.AddArc(orig_tok, arc);
		//cerr << "Added arc " << orig_tok << " -> " << dst_tok << ", " << pdf << ", " << tokens[i].wordid << ", (" << tokens[i].ac_cost << ", " << tokens[i].graph_cost << ")" << endl;
		added_arcs++;
	}
	
	cout << "Added " << added_arcs << " arcs." << endl;


	double convert_elapsed = convert_timer.Elapsed();


	fst::Connect(&lat);
	CompactLattice clat;

	

/*
  247   void InitArcIterator(StateId state, ArcIteratorData<Arc> *data) const {
  248     data->base = nullptr;
  249     data->narcs = states_[state]->NumArcs();
  250     data->arcs = states_[state]->Arcs();
  251     data->ref_count = nullptr;
  252   }
  253 


	int numNodes = lat.NumStates();
	for (int i = 0; i < numNodes; i++) {
		lat.GetArcs(i);

	}
	lat.InitArcIterator(0, );
*/
	// Show the arcs in the lattice
	// for each node (node iterator)
	// for each arc
	// show inode, onode, graph_cost, ac_cost


	//cout << "Connect" << endl;
	//cout << "States: " << lat.NumStates() << endl;


  bool ans = true;

	fst::MutableFst<kaldi::LatticeArc> *ifst;
  double beam;
	fst::MutableFst<kaldi::CompactLatticeArc> *ofst;


	ifst = &lat;
	ofst = &clat;
	beam = 8;

  fst::Invert(ifst);

	//cout << "Invert" << endl;
	//cout << "States: " << lat.NumStates() << endl;

	//cout << "Top sort" << endl;
  if (ifst->Properties(fst::kTopSorted, true) == 0) {
    if (!fst::TopSort(ifst)) {
      // Cannot topologically sort the lattice -- determinization will fail.
      cerr << "Topological sorting of state-level lattice failed (probably"
           << " your lexicon has empty words or your LM has epsilon cycles"
           << ")." << endl;
    }
  }

  fst::ILabelCompare<kaldi::LatticeArc> ilabel_comp;

	//cout << "Arc srt" << endl;
  fst::ArcSort(ifst, ilabel_comp);
  ans = fst::DeterminizeLatticePruned<kaldi::LatticeWeight, kaldi::int32>(*ifst, beam, ofst);

	if (!ans) {
		cerr << "Error determinizing lattice." << endl;
		return 0;
	}

  Connect(ofst);
	//cout << "Connect" << endl;
	//cout << "States: " << clat.NumStates() << endl;

	//lattice_writer.Write(utterance_name, lat);


	//fst::DeterminizeLatticePrunedOptions config;
	//DeterminizeLatticePruned(&lat, 8., &clat);


	CompactLattice clat_best_path;
	CompactLatticeShortestPath(clat, &clat_best_path);

	Lattice best_path;
	ConvertLattice(clat_best_path, &best_path);
	if (best_path.Start() == fst::kNoStateId) {
		cerr << "Best path failed for " << utterance_name << endl;
		return 0;
	}

	std::vector<int32> alignment;
	std::vector<int32> words;
	LatticeWeight weight;

	GetLinearSymbolSequence(best_path, &alignment, &words, &weight);
	if (words.size() == 0) {
		cerr << "[WARN." << utterance_name << "]: Empty transcription" << endl; 
	}
	else {
	
		trans_writer.Write(utterance_name, words);

		if (word_symbols != 0) {
			cout << "  " << utterance_name << ": ";
			for (size_t i = 0; i < words.size(); i++) {
				string s = word_symbols->Find(words[i]);
				if (s == "") s = "<ERR>";
				cout << s << ' ';
			}
			cout << endl;
		}
	}


	clattice_writer.Write(utterance_name, clat);



	if ( time_o.is_open() ) {
		time_o << utterance_name << ", " << convert_elapsed;
		time_o << std::endl;
	}

	//lattice_writer.Close();
	clattice_writer.Close();
	trans_writer.Close();
	time_o.close();

	return 0;
}
