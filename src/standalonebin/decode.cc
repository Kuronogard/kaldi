

#include <iostream>
#include <fstream>

#include "cudamatrix/cu-device.h"
#include "base/kaldi-common.h"
#include "base/timer.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "decoder/decoder-wrappers.h"
#include "fstext/fstext-lib.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "standalonebin/resource_monitor.h"


int main(int argc, char **argv) {



	using namespace kaldi;
	using namespace kaldi::nnet3;
	typedef kaldi::int32 int32;
	using fst::SymbolTable;
	using fst::Fst;
	using fst::StdArc;
	using fst::VectorFst;


	const char *usage =
			"Decode utterance using the given decoding graph.\n"
			"Usage: decode [options] <AM-nnet-filename> <HCLG-graph-filename> <posteriors-rspecifier> <transcription-wspecifier> [<lattice-wspecifier>]\n";

	ParseOptions po(usage);

	LatticeFasterDecoderConfig config;

	std::string symbol_table = "";
	std::string time_log = "";
	std::string use_gpu = "yes";
	BaseFloat acoustic_scale = 0.1;

	po.Register("use-gpu", &use_gpu, "Use gpu when possible (yes|no) (default: yes)");
	po.Register("acoustic-scale", &acoustic_scale, "This should be the acoustic scale used to compute the acoustic probabilities.");
	po.Register("symbol-table", &symbol_table, "Symbol table. If provided, the transcriptions will be shown on standard output");
	po.Register("time-log", &time_log, "File to store time measurements");
	config.Register(&po);

	po.Read(argc, argv);

	if (po.NumArgs() != 5) {
		po.PrintUsage();
		exit(1);
	}
	

	Timer decode_timer;
	ResourceMonitor resourceMonitor;

	std::string am_nnet_filename = po.GetArg(1),
							decode_graph_filename = po.GetArg(2),
							posteriors_rspecifier = po.GetArg(3),
							transcription_wspecifier = po.GetArg(4),
							lattice_wspecifier = po.GetArg(5);



#if HAVE_CUDA==1
		CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif



	//----------------------------------------------------------
	//   Decoding objects
	//----------------------------------------------------------

	TransitionModel trans_model;
	AmNnetSimple am_nnet;
	{
		bool binary;
		Input ki(am_nnet_filename, &binary);
		trans_model.Read(ki.Stream(), binary);
		am_nnet.Read(ki.Stream(), binary);
		SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
		SetDropoutTestMode(true, &(am_nnet.GetNnet()));
		CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
	}


	SequentialBaseFloatMatrixReader posteriors_reader(posteriors_rspecifier);
	Int32VectorWriter trans_writer(transcription_wspecifier);

	CompactLatticeWriter compact_lattice_writer;
	if (lattice_wspecifier != "") {
		compact_lattice_writer.Open(lattice_wspecifier);
	}

	Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(decode_graph_filename);
	LatticeFasterDecoder decoder(*decode_fst, config);


	fst::SymbolTable *word_symbols = 0;
	if (symbol_table != "") { 
		if ( !(word_symbols = fst::SymbolTable::ReadText(symbol_table)) ) {
			KALDI_ERR << "Could not read symbol table " << symbol_table;
		}
	}

	std::ofstream time_o;
	if (time_log != "") {
		time_o.open(time_log);
		if (!time_o.is_open()) {
			KALDI_ERR << "Could not open time log file " << time_log;
		}
		else {
			time_o << "Utterance, decode_time (s)" << std::endl;
		}
	}


	kaldi::int64 frame_count = 0;
	int32 num_success = 0, num_fail = 0;

	for(; !posteriors_reader.Done(); posteriors_reader.Next()) {

		std::string utt = posteriors_reader.Key();
		const Matrix<BaseFloat> &posteriors(posteriors_reader.Value());

		double decode_elapsed;
		//----------------------------------------------------------
		//   Decode
		//----------------------------------------------------------

		std::cout << "Decoding utterance " << utt << std::endl;

		if (posteriors.NumRows() == 0) {
			std::cout << "[WARN." << utt << "]: Zero lenght utterance" << std::endl;
			num_fail++;
			continue;
		}

		DecodableAmNnetSimpleWithIO nnet_decodable(trans_model);
		nnet_decodable.ReadProbsFromMatrix(posteriors);

		decode_timer.Reset();
		if (!decoder.Decode(&nnet_decodable)) {
			std::cout << "[WARN." << utt << "]: Failed to decode" << std::endl;
			num_fail++;
			continue;
		}
		
		decode_elapsed = decode_timer.Elapsed();

		if (!decoder.ReachedFinal()) {
			std::cout << "[WARN. " << utt << "]: No final state reached" << std::endl;
		}


		//----------------------------------------------------------
		//   Extract Best Path
		//----------------------------------------------------------
		VectorFst<LatticeArc> best_path;
		decoder.GetBestPath(&best_path);
		
		std::vector<int32> phones;
		std::vector<int32> words;
		LatticeWeight weight;
		GetLinearSymbolSequence(best_path, &phones, &words, &weight);
		
		if (words.size() == 0) {
			std::cout << "[WARN." << utt << "]: Empty transcription" << std::endl;
			num_fail++;
			continue;
		}

		num_success++;
		frame_count+= posteriors.NumRows();

		trans_writer.Write(utt, words);


		if (word_symbols != 0) {
			std::cout << "  " << utt << ": ";
			for (size_t i = 0; i < words.size(); i++) {
				std::string s = word_symbols->Find(words[i]);
				if (s == "") s = "<ERR>";
				std::cout << s << ' ';
			}
			std::cout << std::endl;
		}


		//----------------------------------------------------------
		//   Extract Lattice
		//----------------------------------------------------------
		// Get raw lattice
		// remove unsuccesfull paths (connect)
		// determinize
		// store

		if (compact_lattice_writer.IsOpen()) {
			Lattice lat;
			decoder.GetRawLattice(&lat);
			fst::Connect(&lat);

			CompactLattice clat;
			DeterminizeLatticePhonePrunedWrapper(trans_model, &lat, config.lattice_beam, 
																				&clat, config.det_opts);

			if (acoustic_scale != 0.0) {
				fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &clat);
			}

			if (clat.Start() == fst::kNoStateId) {
				std::cout << "[WARN." << utt << "]: Empty lattice" << std::endl;
			}

			compact_lattice_writer.Write(utt, clat);
		}


		if (time_o.is_open()) {
			time_o << utt << ", " << decode_elapsed;
			time_o << std::endl;
		}

	}

	std::cout << "Overall: " << num_success << " success, " << num_fail << " fail. " << frame_count << " total frames." << std::endl;


	delete word_symbols;
	time_o.close();

	return 0;
}
