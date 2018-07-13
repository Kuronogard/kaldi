

#include <iostream>
#include <fstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet2/decodable-am-nnet.h"


int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet2;
		
		
		const char *usage =
				"Generate posterior probabilities from features";

		ParseOptions po(usage);
		
		Timer timer;
		bool pad_input = true;
		BaseFloat acoustic_scale = 0.1;

		po.Read(argc, argv);
		if (po.NumArgs() < 4 || po.NumArgs() > 4) {
			po.PrintUsage();
			exit(1);
		}

		std::string features_rspecifier = po.GetArg(1);
		std::string ac_model_filename = po.GetArg(2);
		std::string posterior_wspecifier = po.GetArg(3);
		std::string time_log_filename = po.GetArg(4);
	
		std::ofstream elapsed_time_o(time_log_filename);

		TransitionModel trans_model;
		AmNnet am_nnet;
		{
			bool binary;
			Input ki(ac_model_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_nnet.Read(ki.Stream(), binary);
		}

		SequentialBaseFloatCuMatrixReader features_reader(features_rspecifier);
		BaseFloatMatrixWriter posterior_writer(posterior_wspecifier);


		elapsed_time_o << "utterance, frames, fps, execution time" << std::endl;
		kaldi::int64 frame_count = 0;
		kaldi::int64 utt_count = 0;

		for(; !features_reader.Done(); features_reader.Next()) {
			std::string utt = features_reader.Key();
			const CuMatrix<BaseFloat> &features(features_reader.Value());

			std::cout << "Procesando: " << utt << std::endl;
			
			DecodableAmNnetWithIO nnet_decodable(trans_model);
			
			timer.Reset();
			nnet_decodable.ComputeFromModel(am_nnet, features, pad_input, acoustic_scale);
			double elapsed = timer.Elapsed();

			// Store utterance name, elapsed time and real time factor
			elapsed_time_o << utt << ", " << features.NumRows() << ", 100," << elapsed << std::endl;

			posterior_writer.Write(utt, nnet_decodable.GetLogProbs());
			//nnet_decodable.WriteProbsToTable(posterior_writer, utt);
			frame_count += features.NumRows();
			utt_count++;
		}
		std::cout << "Terminado" << std::endl;
		std::cout << "Total frames are " << frame_count << " for " << utt_count << " utterances." << std::endl;

		elapsed_time_o.close();

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
