/*
 * Extract probabilities from 40-mfcc, cmvn and 60-ivectors
 */

#include <iostream>
#include <fstream>
#include <vector>

#include "cudamatrix/cu-device.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "standalonebin/resource_monitor.h"

int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		
		
		const char *usage =
				"Generate posterior probabilities from features.\n"
				"Usage: extract_probs <online-ivector-rspecifier> <features-rspecifier> <AM-nnet-filename> <posteriors-wspecifier>";

		ParseOptions po(usage);
		
		NnetSimpleComputationOptions decodable_opts;

		int32 online_ivector_period = 10;
		std::string use_gpu = "yes";
		std::string time_log = "";
		std::string energy_log = "";
		std::string profile = "";

		//Timer timer;
		ResourceMonitor resourceMonitor;

		//bool pad_input = true;
		//BaseFloat acoustic_scale = 0.1;

		po.Register("use-gpu", &use_gpu, "Use GPU when possible (yes|no) (default:yes).");
		po.Register("time-log", &time_log, "File to store time logs.");
		po.Register("energy-log", &energy_log, "File to store snergy logs.");
		po.Register("profile", &profile, "File to store profile information, such as execution time and energy consumption. This file contains a sumary of all the different logs.");

		decodable_opts.Register(&po);

		po.Read(argc, argv);
		if (po.NumArgs() < 4 || po.NumArgs() > 4) {
			po.PrintUsage();
			exit(1);
		}

		std::string online_ivector_rspecifier = po.GetArg(1),
								features_rspecifier = po.GetArg(2),
								ac_model_filename = po.GetArg(3),
								posterior_wspecifier = po.GetArg(4);	


		
		std::ofstream time_o;
		if (time_log != "") {
			time_o.open(time_log);
			if (!time_o.is_open()) {
				KALDI_ERR << "Could not open time log file " << time_log;
			}
			time_o << "utterance, frames, time (s)" << std::endl;
		}


		std::ofstream energy_o;
		if (energy_log != "") {
			energy_o.open(energy_log);
			if (!energy_o.is_open()) {
				KALDI_ERR << "Could not open energy log file " << energy_log;
			}
			energy_o << "utterance, energy (mJ)" << std::endl;
		}


		std::ofstream profile_o;
		if (profile != "") {
			profile_o.open(profile);
			if (!profile_o.is_open()) {
				KALDI_ERR << "Could not open profile file " << profile;
			}
			profile_o << "utterance, frames, time (s) ";
			profile_o << ", avg power CPU (mW), avg power GPU (mW) ";
			profile_o << ", energy CPU (mJ), energy GPU (mJ)" << std::endl;
		}


#if HAVE_CUDA==1
		CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif


		TransitionModel trans_model;
		AmNnetSimple am_nnet;
		{
			bool binary;
			Input ki(ac_model_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_nnet.Read(ki.Stream(), binary);
			SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
			SetDropoutTestMode(true, &(am_nnet.GetNnet()));
			CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
		}


		RandomAccessBaseFloatMatrixReader online_ivector_reader(online_ivector_rspecifier);

		SequentialBaseFloatMatrixReader features_reader(features_rspecifier);
		BaseFloatMatrixWriter posterior_writer(posterior_wspecifier);

		CachingOptimizingCompiler compiler(am_nnet.GetNnet(), decodable_opts.optimize_config);



		kaldi::int64 frame_count = 0;
		kaldi::int64 utt_count = 0;
		
		resourceMonitor.init();
	
		for(; !features_reader.Done(); features_reader.Next()) {
			std::string utt = features_reader.Key();
			const Matrix<BaseFloat> &features(features_reader.Value());
			const Matrix<BaseFloat> &online_ivectors(online_ivector_reader.Value(utt));

			std::cout << "Procesando: " << utt << std::endl;
			
			DecodableAmNnetSimpleWithIO nnet_decodable(trans_model);

			
			//timer.Reset();
			resourceMonitor.startMonitoring();
			nnet_decodable.ComputeFromModel(decodable_opts, am_nnet, features, 
								&online_ivectors, online_ivector_period, &compiler);
			resourceMonitor.endMonitoring();
			//double elapsed = timer.Elapsed();

			// Extract measurements and log them if required
			double elapsed = resourceMonitor.getTotalExecTime();
			double avgPowerCPU = resourceMonitor.getAveragePowerCPU();
			double avgPowerGPU = resourceMonitor.getAveragePowerGPU();
			double energyCPU = resourceMonitor.getTotalEnergyCPU();
			double energyGPU = resourceMonitor.getTotalEnergyGPU();

			// Store utterance name, elapsed time

			if (time_o.is_open()) {
				time_o << utt << ", " << features.NumRows() << ", " << elapsed << std::endl;
			}

			if (energy_o.is_open()) {
				energy_o << utt << ", " << energyCPU + energyGPU << std::endl;
			}

			if (profile_o.is_open()) {
				profile_o << utt << ", " << features.NumRows() << ", " << elapsed;
				profile_o << ", " << avgPowerCPU << ", " << avgPowerGPU;
				profile_o << ", " << energyCPU << ", " << energyGPU;
				profile_o << std::endl;
			}
		
			posterior_writer.Write(utt, nnet_decodable.GetLogProbs());

			frame_count += features.NumRows();
			utt_count++;

		}
		std::cout << "Finished computing posterior probabilities." << std::endl;
		std::cout << "Total frames is " << frame_count << " for " << utt_count << " utterances." << std::endl;

		if (time_o.is_open()) time_o.close();
		if (energy_o.is_open()) energy_o.close();
		if (profile_o.is_open()) profile_o.close();

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
