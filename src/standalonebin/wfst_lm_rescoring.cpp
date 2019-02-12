


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/kaldi-lattice.h"

#include "lat/lattice-functions.h"
#include "lm/const-arpa-lm.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "standalonebin/resource_monitor_ARM.h"

int main(int argc, char **argv) {

	using namespace kaldi;
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;
	using fst::VectorFst;
	using fst::StdArc;
	using fst::ReadFstKaldi;


	const char *usage = 
			"Obtains the best path through a lattice rescored with a new language model.\n"
			"For that, it substitutes the graph cost corresponding to oldLM from paths in lattice with the \n"
			"cost in newLM. "
			"Does this by composing first the lattice with the oldLM, whose weights are scaled by -1, and \n"
			"then with composing the resulting graph with newLM.\n"
			"usage: wfst-lm-rescoring [options] <lattice-rspecifier> <oldLM-filename> <newLM-filename> <transcription-wspecifier";

	ParseOptions po(usage);
	BaseFloat rescore_lm_scale = 1.0;
  BaseFloat lm_scale = 1.0;
	BaseFloat acoustic_scale = 1.0;
	BaseFloat acoustic2lm_scale = 0.0;
	BaseFloat lm2acoustic_scale = 0.0;
	BaseFloat word_ins_penalty = 0.0;
	std::string symbol_table = "";
	std::string time_log = "";
	std::string profile = "";
	std::string use_gpu = "yes";

	int32 num_states_cache = 50000;
	double measure_period = 0.1;


	po.Register("use-gpu", &use_gpu, "Use gpu when possible (yes|no) (default: yes)");
	po.Register("rescore-lm-scale", &rescore_lm_scale, "Scaling factor for language model cost, used only when rescoring");
	po.Register("lm-scale", &lm_scale, "Scaling factor for language model cost. Used only for best path evaluation");
	po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic cost. Used only for best path evaluation");
	po.Register("word-ins-penalty", &word_ins_penalty, "Word Insertion Penalty. This value is added to the graph weight of every arc in the lattice acceptor with a word label (those with no empty labels)");
	po.Register("symbol-table", &symbol_table, "Symbol table. if provided, the transcriptions will be shown on standart output");
	po.Register("time-log", &time_log, "File to store time measurements");
	po.Register("profile", &profile, "File to store time, energy nd power.");
	po.Register("measure-period", &measure_period, "Time (seconds) between energy measurements.");

	po.Read(argc, argv);

	if (po.NumArgs() != 4) {
		po.PrintUsage();
		exit(1);
	}


	std::string lats_rspecifier = po.GetArg(1),
							oldLM_filename = po.GetArg(2),
							newLM_filename = po.GetArg(3),
							transcription_wspecifier = po.GetArg(4);


#if HAVE_CUDA==1
		CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif



	//Timer rescore_timer;
	ResourceMonitorARM resourceMonitor;
	resourceMonitor.init();

  // read old WFST LM (ARPA)
	VectorFst<StdArc> *std_old_LM;
	std_old_LM = ReadFstKaldi(oldLM_filename);
	if (std_old_LM->Properties(fst::kILabelSorted, true) == 0) {
		ArcSort(std_old_LM, fst::ILabelCompare<StdArc>());
	}


	// old_LM is the LM interpreted using the LatticeWeight semiring.
	fst::CacheOptions cache_opts(true, num_states_cache);
	fst::MapFstOptions mapfst_opts(cache_opts);
	fst::StdToLatticeMapper<BaseFloat> mapper;
	fst::MapFst<StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> > old_LM(*std_old_LM, mapper, mapfst_opts);

  delete std_old_LM;



	// These lines are an optimization for the composition
	fst::TableComposeOptions compose_opts(fst::TableMatcherOptions(), true, fst::SEQUENCE_FILTER, fst::MATCH_INPUT);
	fst::TableComposeCache<fst::Fst<LatticeArc> > lm_compose_cache(compose_opts);

  // read new WFST LM (constARPA)
	ConstArpaLm new_LM;
	ReadKaldiObject(newLM_filename, &new_LM);

  // read lattice (Normal lattice)
	SequentialLatticeReader lattice_reader(lats_rspecifier);

	// create transcription writter and symbol-table reader objects
	Int32VectorWriter trans_writer(transcription_wspecifier);

	fst::SymbolTable *word_symbols = 0;
	if ( symbol_table != "" ) {
		if ( !(word_symbols = fst::SymbolTable::ReadText(symbol_table)) ) {
			KALDI_ERR << "Could not read symbol table " << symbol_table;
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
	
	std::ofstream profile_o;
	if ( profile != "" ) {
		profile_o.open(profile);
		if ( !profile_o.is_open() ) {
			KALDI_WARN << "Could not open profile log file " << profile;
		} 
		else {
			profile_o << "Utterance, time (s), avg power CPU (W), avg power GPU (W), energy CPU (J), energy GPU (J)";
			profile_o << ", num values" << std::endl;
		}
	}

	int32 num_success = 0, num_fail = 0;

	for(; !lattice_reader.Done(); lattice_reader.Next()) {
		std::string utt = lattice_reader.Key();
		Lattice lat = lattice_reader.Value();
		lattice_reader.FreeCurrent();
		
		std::cout << "Starting LM rescore for " << utt << std::endl;

		//rescore_timer.Reset();		

		// ---------------------------------------------------
		//		Remove old LM weights
		// ---------------------------------------------------
		// Escalar lattice -1/lm_scale y ordenar por oLabel
		// componer con oldLM
		// Invertir lattice
		// Determinizar lattice
		// Escalar lattice -1


		//std::cout << "Remove old weights" << std::endl;
		
		resourceMonitor.startMonitoring(measure_period);
		fst::ScaleLattice(fst::GraphLatticeScale(-1.0/rescore_lm_scale), &lat);

		//KALDI_WARN << "old_LM arcs " << old_LM.NumArcs(1);
		//KALDI_WARN << "qewdqwe";

		ArcSort(&lat, fst::OLabelCompare<LatticeArc>());



		Lattice lat_nolm;
		//TableCompose(lat, old_LM, &lat_nolm, &lm_compose_cache);
		Compose(lat, old_LM, &lat_nolm);
		if ( lat_nolm.Start() == fst::kNoStateId ) {
			resourceMonitor.endMonitoring();
			std::cout << "[WARN." << utt << "]: Unscored lattice is empty." << std::endl;
			num_fail++;
      continue;
		}

		Invert(&lat_nolm);
		
    CompactLattice clat_nolm_determinized;
		DeterminizeLattice(lat_nolm, &clat_nolm_determinized);


		fst::ScaleLattice(fst::GraphLatticeScale(-1), &clat_nolm_determinized);


		// ---------------------------------------------------
		//		Add new LM weights	
		// ---------------------------------------------------
		// componer con newLM
		// Invertir lattice
		// Determinizar lattice
		// Escalar por lm_scale

		//std::cout << "Add new weights" << std::endl;

		ArcSort(&clat_nolm_determinized, fst::OLabelCompare<CompactLatticeArc>());
		ConstArpaLmDeterministicFst new_LM_wrapped(new_LM);

		//std::cout << "Compose with LM" << std::endl;
		CompactLattice clat_rescored;
		ComposeCompactLatticeDeterministic(clat_nolm_determinized, &new_LM_wrapped, &clat_rescored);
		if (clat_rescored.Start() == fst::kNoStateId) {
			resourceMonitor.endMonitoring();
			std::cout << "[WARN." << utt << "]: Rescored lattice is empty." << std::endl;
			num_fail++;
			continue;
		}


		//std::cout << "Convert to normal lattice and invert" << std::endl;
		Lattice lat_rescored;
		ConvertLattice(clat_rescored, &lat_rescored);
		Invert(&lat_rescored);

		//std::cout << "Determinize lattice" << std::endl;
		CompactLattice clat_rescored_determinized;
		DeterminizeLattice(lat_rescored, &clat_rescored_determinized);


		//std::cout << "Scale lattice" << std::endl;
		fst::ScaleLattice(fst::GraphLatticeScale(rescore_lm_scale), &clat_rescored_determinized);

		resourceMonitor.endMonitoring();
		//double rescore_elapsed = rescore_timer.Elapsed();

		// ---------------------------------------------------
		//		Compute Best Path
		// ---------------------------------------------------
		// Escalar (lattice-scale --inv-acoustic-scale=7..17)
		// Sumar penalty (lattice-add-penalty --word-ins-penalty=0.0, 0.5, 1.0)
		// Calcular best path (lattice-best-path)
	
		//std::cout << "Compute best path" << std::endl;

		std::vector<std::vector<double> > scale(2);
		scale[0].resize(2);
		scale[1].resize(2);
		scale[0][0] = lm_scale;
		scale[0][1] = acoustic2lm_scale;
		scale[1][0] = lm2acoustic_scale;
		scale[1][1] = acoustic_scale;

		ScaleLattice(scale, &clat_rescored_determinized);
		AddWordInsPenToCompactLattice(word_ins_penalty, &clat_rescored_determinized);

	
		CompactLattice clat_best_path;
		CompactLatticeShortestPath(clat_rescored_determinized, &clat_best_path);


		// ---------------------------------------------------
		//		Save Symbol Sequences
		// ---------------------------------------------------

		std::cout << "Store symbol sequences" << std::endl;

		Lattice lat_best_path;
		ConvertLattice(clat_best_path, &lat_best_path);

		std::vector<int32> phones;
		std::vector<int32> words;
		LatticeWeight weight;
		GetLinearSymbolSequence(lat_best_path, &phones, &words, &weight);		

		if (words.size() == 0) {
			std::cout << "[WARN." << utt << "]: Empty transcription" << std::endl;
      num_fail++;
			continue;
		}

		num_success++;

		// write symbols
		trans_writer.Write(utt, words);


		// write elapsed
		if ( time_o.is_open() ) {
			double rescore_elapsed = resourceMonitor.getTotalExecTime();
			time_o << utt << ", " << rescore_elapsed;
			time_o << std::endl;
		}

		if (profile_o.is_open()) {
			double elapsed = resourceMonitor.getTotalExecTime();
			double cpuPower = resourceMonitor.getAveragePowerCPU();
			double gpuPower = resourceMonitor.getAveragePowerGPU();
			double cpuEnergy = resourceMonitor.getTotalEnergyCPU();
			double gpuEnergy = resourceMonitor.getTotalEnergyGPU();
			int numValues = resourceMonitor.numData();
	
			profile_o << utt << ", " << elapsed << ", " << cpuPower << ", " << gpuPower << ", " << cpuEnergy;
			profile_o << ", " << gpuEnergy << ", " << numValues << endl;
		}


		// If no word_symbol available, skip this part
		if ( word_symbols != 0 ) {
			std::cout << "  " << utt << ": ";
			for (size_t i = 0; i < words.size(); i++) {
				std::string s = word_symbols->Find(words[i]);
				if (s == "") s = "<ERR>";
				std::cout << s << ' ';
			}
			std::cout << std::endl;
		}

	}

  std::cout << "Overall: " << num_success << " success, " << num_fail << " fail." << std::endl;


	delete word_symbols;
	if (time_o.is_open()) time_o.close();
	if (profile_o.is_open()) profile_o.close();		

	return 0;
}
