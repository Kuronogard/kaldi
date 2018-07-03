/*
 * Extract probabilities from 40-mfcc, cmvn and 60-ivectors
 */





#include <iostream>
#include <fstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"


int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		
		
		const char *usage =
				"Generate posterior probabilities from features";

		ParseOptions po(usage);
		
		NnetSimpleComputationOptions decodable_opts;

		int32 online_ivector_period = 10;
		decodable_opts.Register(&po);

		Timer timer;
		//bool pad_input = true;
		//BaseFloat acoustic_scale = 0.1;

		po.Read(argc, argv);
		if (po.NumArgs() < 5 || po.NumArgs() > 5) {
			po.PrintUsage();
			exit(1);
		}

		std::string online_ivector_rspecifier = po.GetArg(1);
		std::string features_rspecifier = po.GetArg(2);
		std::string ac_model_filename = po.GetArg(3);
		std::string posterior_wspecifier = po.GetArg(4);
		std::string time_log_filename = po.GetArg(5);	

		std::ofstream elapsed_time_o(time_log_filename);

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


		elapsed_time_o << "utterance, frames, fps, execution time" << std::endl;

		kaldi::int64 frame_count = 0;
		kaldi::int64 utt_count = 0;		

		for(; !features_reader.Done(); features_reader.Next()) {
			std::string utt = features_reader.Key();
			const Matrix<BaseFloat> &features(features_reader.Value());
			const Matrix<BaseFloat> &online_ivectors(online_ivector_reader.Value(utt));

			std::cout << "Procesando: " << utt << std::endl;
			
			DecodableAmNnetSimpleWithIO nnet_decodable(trans_model);
			
			timer.Reset();
			nnet_decodable.ComputeFromModel(decodable_opts, am_nnet, features, 
								&online_ivectors, online_ivector_period, &compiler);
			
			double elapsed = timer.Elapsed();

			// Store utterance name, elapsed time and real time factor
			elapsed_time_o << utt << ", " << features.NumRows() << ", 100, " << elapsed << std::endl;

			posterior_writer.Write(utt, nnet_decodable.GetLogProbs());
			//nnet_decodable.WriteProbsToTable(posterior_writer, utt);
			frame_count += features.NumRows();
			utt_count++;
		}
		std::cout << "Finished computing posterior probabilities." << std::endl;
		std::cout << "Total frames is " << frame_count << " for " << utt_count << " utterances." << std::endl;

		elapsed_time_o.close();

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
