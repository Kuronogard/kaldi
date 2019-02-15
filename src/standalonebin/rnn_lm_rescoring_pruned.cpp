#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/kaldi-lattice.h"

#include "lat/lattice-functions.h"
#include "lat/compose-lattice-pruned.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "standalonebin/resource_monitor_threaded.h"
#include "standalonebin/resource_monitor.h"


#define ABS_(a) (((a)>=0)? (a) : -(a))
#define MAXABS_(a, b)  ((ABS_(a) > ABS_(b))? ABS_(a) : ABS_(b))

using kaldi::nnet3::QuantParams;

QuantParams quant_params[8] = {
  QuantParams(-0.845536,0.845536, -1.78486,  1.78486, -127, 127, -127, 127),
  QuantParams(-8.64707, 8.64707,  -1.98485,  1.98485, -127, 127, -127, 127),
  QuantParams(-1,       1,        -1.02597,  1.02597, -127, 127, -127, 127),
  QuantParams(-6.24459, 6.24459,  -0.897963, 0.897963,-127, 127, -127, 127),
  QuantParams(-15.2109, 15.2109,  -2.42414,  2.42414, -127, 127, -127, 127),
  QuantParams(-1,       1,        -1.39773,  1.39773, -127, 127, -127, 127),
  QuantParams(-8.27293, 8.27293,  -1.66043,  1.66043, -127, 127, -127, 127),
  QuantParams(-29.213,  29.213,   -2.90258,  2.90258, -127, 127, -127, 127)
};



int main(int argc, char **argv) {

	using namespace kaldi;
	using namespace kaldi::nnet3;
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

	rnnlm::RnnlmComputeStateComputationOptions opts;
  ComposeLatticePrunedOptions compose_pruned_opts;
	BaseFloat rescore_lm_scale = 1.0;
  BaseFloat lm_scale = 1.0;
	BaseFloat acoustic_scale = 1.0;
	BaseFloat acoustic2lm_scale = 0.0;
	BaseFloat lm2acoustic_scale = 0.0;
	BaseFloat word_ins_penalty = 0.0;
	int32 max_ngram_order = 3;
	std::string symbol_table = "";
	std::string time_log = "";
	std::string quant_log = "";
	std::string use_gpu = "yes";
	std::string profile = "";
	double measure_period = 0.1;
	double rnnlm_measure_period = 0.001;
	bool quantize = false;

	int32 num_states_cache = 50000;

	
	po.Register("use-gpu", &use_gpu, "Use gpu when possible (yes|no) (default: yes)");
	po.Register("rescore-lm-scale", &rescore_lm_scale, "Scaling factor for language model cost, used only when rescoring");
	po.Register("lm-scale", &lm_scale, "Scaling factor for language model cost. Used only for best path evaluation");
	po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic cost. Used only for best path evaluation");
	po.Register("max-ngram-order", &max_ngram_order, "Max ngram used for the ngram approximation of the rnn computation");
	po.Register("word-ins-penalty", &word_ins_penalty, "Word Insertion Penalty. This value is added to the graph weight of every arc in the lattice acceptor with a word label (those with no empty labels)");
	po.Register("symbol-table", &symbol_table, "Symbol table. if provided, the transcriptions will be shown on standart output");
	po.Register("time-log", &time_log, "File to store time measurements");
	po.Register("quant-range-log", &quant_log, "File to write input min and max values");
	po.Register("num-states-cache", &num_states_cache, "Number of states cached when mapping LM FST to lattice type. Consumes more memory but it is faster. (I don't know if it affects accuracy)");
	po.Register("profile", &profile, "File to store profile info: power, energy and execution time.");
	po.Register("measure-period", &measure_period, "Time between energy measurements");
	po.Register("rnnlm-measure-period", &rnnlm_measure_period, "Time between energy measurements for internal rnn evaluations");
	po.Register("quantize", &quantize, "Quantize network");

	opts.Register(&po);
  compose_pruned_opts.Register(&po);

	po.Read(argc, argv);

	if (po.NumArgs() != 5) {
		po.PrintUsage();
		exit(1);
	}


	std::string lats_rspecifier = po.GetArg(1),
							oldLM_filename = po.GetArg(2),
							rnnlm_filename = po.GetArg(3),
							word_embedding_filename = po.GetArg(4),
							transcription_wspecifier = po.GetArg(5);


#if HAVE_CUDA==1
		CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif



	//Timer rescore_timer;
	ResourceMonitorThreaded resourceMonitor;

	std::ofstream quant_o;
	if (quant_log != "") {
		quant_o.open(quant_log);
		if (!quant_o.is_open()) {
			KALDI_ERR << "Could not open quant ranges log file " << quant_log;
		}
		quant_o << "component, in min value, in max value, w min value, w max value, num inputs" << std::endl;
	}



	// Read Old LM
	fst::ScaleDeterministicOnDemandFst *lm_to_subtract_det_scale = NULL;
	fst::BackoffDeterministicOnDemandFst<StdArc> *lm_to_subtract_det_backoff = NULL;
	VectorFst<StdArc> *lm_to_subtract_fst = NULL;
	
	lm_to_subtract_fst = fst::ReadAndPrepareLmFst(oldLM_filename);
	lm_to_subtract_det_backoff = new fst::BackoffDeterministicOnDemandFst<StdArc>(*lm_to_subtract_fst);
	lm_to_subtract_det_scale = new fst::ScaleDeterministicOnDemandFst(-lm_scale, lm_to_subtract_det_backoff);


  // read new RNN LM (RNN LSTM)
	kaldi::nnet3::Nnet rnnlm;
	ReadKaldiObject(rnnlm_filename, &rnnlm);

	KALDI_ASSERT(IsSimpleNnet(rnnlm));

	int quantized_layers = 0;
	if (quant_o.is_open() || quantize) {
		for (int i = 0; i < rnnlm.NumComponents(); i++) {
			Component *comp = rnnlm.GetComponent(i);
			if (comp->Properties() & kaldi::nnet3::ComponentProperties::kQuantizableComponent) {
				std::cerr << "Quantize weights of layers " << quantized_layers << std::endl;
				if (quantized_layers >= 8) {
					std::cerr << "Stopper quantization at layer 8" << std::endl;
					break;
				}
				QuantizableSubComponent *qComp = dynamic_cast<QuantizableSubComponent*>(comp);
				qComp->SetQuantizationParams(quant_params[quantized_layers]);
				qComp->SetQuantize(quantize);
				qComp->QuantizeWeights();
				quantized_layers++;
			}
		}
	}

	CuMatrix<BaseFloat> word_embedding_mat;
	ReadKaldiObject(word_embedding_filename, &word_embedding_mat);

	const rnnlm::RnnlmComputeStateInfo info(opts, rnnlm, word_embedding_mat);


  // read lattice (Normal lattice)
	SequentialCompactLatticeReader clattice_reader(lats_rspecifier);

	// create transcription writter and symbol-table reader objects
	Int32VectorWriter trans_writer(transcription_wspecifier);

	fst::SymbolTable *word_symbols = 0;
	if ( symbol_table != "" ) {
		if ( !(word_symbols = fst::SymbolTable::ReadText(symbol_table)) ) {
			KALDI_ERR << "Could not read symbol table" << symbol_table;
		}
	}

	std::ofstream time_o;
	if ( time_log != "" ) {
		time_o.open(time_log);
		if ( !time_o.is_open() ) {
			KALDI_WARN << "Could not open time log file" << time_log;
		}
		else {
			time_o << "Utterance, time (s)" << std::endl;
		}
	}
	
	std::ofstream profile_o;
	if ( profile != "" ) {
		profile_o.open(profile);
		if ( !profile_o.is_open() ) {
			KALDI_WARN << "Could not open profile log file" << profile;
			measure_period = 0;
		}
		else {
      /*
			profile_o << "Utterance, time (s), avg power CPU (W), avg power GPU (W), energy CPU (J), energy GPU (J)";	
			profile_o << ", num values, rnnlm time, rnnlm energy, rnnlm num execs, rnnlm extra time";
      profile_o << ", latCompose totalTime (s), latComposeStats totalEnergy (J), latCompose rnnTime, latCompose rnnEnergy";
      profile_o << ", latComposeStats HeuristicTime, latComposeStats HeuristicEnergy";
      profile_o << std::endl;
      */
      profile_o << "Utterance, GlobalTime, rnnlmTime, heuristicTime, numRnnlmExecs, rnnlmExactTime, rnnlmExtraTime";
      profile_o << std::endl;
		}
	}
	else {
		measure_period = 0;
	}


	try {

		//rnnlm::KaldiRnnlmDeterministicFst* lm_to_add_orig = new rnnlm::KaldiRnnDeterministicFst(max_ngram_order, info);

		int32 num_success = 0, num_fail = 0;
		for(; !clattice_reader.Done(); clattice_reader.Next()) {
			std::string utt = clattice_reader.Key();
			CompactLattice clat = clattice_reader.Value();
			clattice_reader.FreeCurrent();
		
			std::cout << "Starting LM rescore for " << utt << std::endl;

      rnnlm::KaldiRnnlmDeterministicFst* lm_to_add_orig = new rnnlm::KaldiRnnlmDeterministicFst(max_ngram_order, info, rnnlm_measure_period);
			//rescore_timer.Reset();		
			resourceMonitor.startMonitoring(measure_period);


			fst::DeterministicOnDemandFst<StdArc> *lm_to_add = new fst::ScaleDeterministicOnDemandFst(lm_scale, lm_to_add_orig);
			fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &clat);
		
			TopSortCompactLatticeIfNeeded(&clat);

			fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(lm_to_subtract_det_scale, lm_to_add);
      std::cerr << "Before Rescoring" << std::endl;

      ComposeLatticePrunedStats latComposeStats;
			CompactLattice composed_clat;
			ComposeCompactLatticePruned(compose_pruned_opts, clat, &combined_lms, &composed_clat, &latComposeStats);
      std::cerr << "After Rescoring" << std::endl;

			//lm_to_add_orig->Clear();

			if (composed_clat.NumStates() == 0) {
				num_fail++;
			}
			else {
				fst::ScaleLattice(fst::AcousticLatticeScale(1.0/acoustic_scale), &composed_clat);
				num_success++;
			}

			//double rescore_elapsed = rescore_timer.Elapsed();
			resourceMonitor.endMonitoring();

			// ---------------------------------------------------
			//		Compute Best Path
			// ---------------------------------------------------
			// Escalar (lattice-scale --inv-acoustic-scale=7..17)
			// Sumar penalty (lattice-add-penalty --word-ins-penalty=0.0, 0.5, 1.0)
			// Calcular best path (lattice-best-path)
	
			std::cout << "Compute best path" << std::endl;

			std::vector<std::vector<double> > scale(2);
			scale[0].resize(2);
			scale[1].resize(2);
			scale[0][0] = lm_scale;
			scale[0][1] = acoustic2lm_scale;
			scale[1][0] = lm2acoustic_scale;
			scale[1][1] = acoustic_scale;

			ScaleLattice(scale, &composed_clat);
			AddWordInsPenToCompactLattice(word_ins_penalty, &composed_clat);

	
			CompactLattice clat_best_path;
			CompactLatticeShortestPath(composed_clat, &clat_best_path);


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
				std::cout << "[WARN." << utt << "] Transcription is empty." << std::endl;
				num_fail++;
				continue;
			}

			// write symbols
			trans_writer.Write(utt, words);
			std::cerr << "Done " << num_success+num_fail << " uterrances."; 
			std::cerr << "(success:" << num_success << ", fail:" << num_fail << ")" << std::endl;

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

        kaldi::rnnlm::RnnlmDeterministicFstStats rnnlm_stats;
				lm_to_add_orig->GetStatistics(rnnlm_stats);	

				if (numValues < 20) cerr << "WARN: less than 20 measures (" << numValues << ")" << endl; 
/*
				profile_o << utt << ", " << elapsed << ", " << cpuPower << ", " << gpuPower << ", " << cpuEnergy;
				profile_o << ", " << gpuEnergy << ", " << numValues;
				profile_o << ", " << rnnlm_stats.exec_time << ", " << rnnlm_stats.energy << ", " << rnnlm_stats.rnnlm_num_executions;
        profile_o << ", " << rnnlm_stats.extra_time;
        profile_o << ", " << latComposeStats.totalTime << ", " << latComposeStats.totalEnergy;
        profile_o << ", " << latComposeStats.rnnComputationsTime << ", " << latComposeStats.rnnComputationsEnergy;
        profile_o << ", " << latComposeStats.computeHeuristicTime << ", " << latComposeStats.computeHeuristicEnergy;
        profile_o << std::endl;
*/
        profile_o << utt << ", " << elapsed;
        profile_o << ", " << latComposeStats.rnnComputationsTime << ", " << latComposeStats.computeHeuristicTime;
        profile_o << ", " << rnnlm_stats.rnnlm_num_executions << ", " << rnnlm_stats.execTime << ", " << rnnlm_stats.extraTime;
        profile_o << std::endl;
			}


			// If no word_symbol available, skip this part
			if ( word_symbols != 0 ) {
				std::cout << "  " << utt << ": ";
				for (size_t i = 0; i < words.size(); i++) {
					std::string s = word_symbols->Find(words[i]);
					if (s == "") s = "<ERR>";
					std::cout << s << ' ';
				}
			}

	    delete lm_to_add;
		} // End of the lattice for loop

	} catch (std::exception &e) {
		std::cerr << "Exception: " << e.what() << std::endl;
	} catch (...) {
    std::cerr << "Interrupted by non-std exception" << std::endl;
	}

	if (quant_o.is_open()) {
		for (int i = 0; i < rnnlm.NumComponents(); i++) {
			Component *comp = rnnlm.GetComponent(i);
			ComponentStatistics stats;
			comp->GetStatistics(stats);

			quant_o << comp->Type() << ", " << stats.in_range_min << ", " << stats.in_range_max;
			quant_o << ", " << stats.w_range_min << ", " << stats.w_range_max;
			quant_o << ", " << stats.num_inputs << std::endl;
		}
	}

	if (quant_o.is_open()) {
		quant_o << std::endl << std::endl;
		for (int i = 0; i < rnnlm.NumComponents(); i++) {
			Component *comp = rnnlm.GetComponent(i);
			ComponentStatistics stats;
			comp->GetStatistics(stats);
			BaseFloat in = MAXABS_(stats.in_range_min, stats.in_range_max);
			BaseFloat w = MAXABS_(stats.w_range_min, stats.w_range_max);

			quant_o << comp->Type() << ", " << -in << ", " << in;
			quant_o << ", " << -w << ", " << w;
			quant_o << ", " << stats.num_inputs << std::endl;
		}
	}

	delete word_symbols;
	if (time_o.is_open()) time_o.close();
  if (quant_o.is_open()) quant_o.close();
  if (profile_o.is_open()) profile_o.close();

	return 0;
}
