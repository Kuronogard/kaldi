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


using kaldi::nnet3::QuantParams;

QuantParams quant_params[24] = {
    QuantParams(-55.262,  55.262,  -0.124089, 0.124089, -127, 127, -127, 127),
    QuantParams(-21.6077, 21.6077, -0.165391, 0.165391, -127, 127, -127, 127),
    QuantParams(-59.3132, 59.3132, -0.146865, 0.146865, -127, 127, -127, 127),
    QuantParams(-28.7919, 28.7919, -0.171094, 0.171094, -127, 127, -127, 127),
    QuantParams(-59.5458, 59.5458, -0.123518, 0.123518, -127, 127, -127, 127),
    QuantParams(-25.194,  25.194,  -0.110173, 0.110173, -127, 127, -127, 127),
    QuantParams(-63.91,   63.91,   -0.173243, 0.173243, -127, 127, -127, 127),
    QuantParams(-30.1098, 30.1098, -0.177467, 0.177467, -127, 127, -127, 127),
    QuantParams(-60.7332, 60.7332, -0.118517, 0.118517, -127, 127, -127, 127),
    QuantParams(-18.2573, 18.2573, -0.116912, 0.116912, -127, 127, -127, 127),
    QuantParams(-84.1307, 84.1307, -0.127527, 0.127527, -127, 127, -127, 127),
    QuantParams(-23.6543, 23.6543, -0.194101, 0.194101, -127, 127, -127, 127),
    QuantParams(-40.8354, 40.8354, -0.133696, 0.133696, -127, 127, -127, 127),
    QuantParams(-13.1671, 13.1671, -0.129242, 0.129242, -127, 127, -127, 127),
    QuantParams(-34.4113, 34.4113, -0.127431, 0.127431, -127, 127, -127, 127),
    QuantParams(-23.6543, 23.6543, -0.171385, 0.171385, -127, 127, -127, 127),
    QuantParams(-40.3526, 40.3526, -0.129784, 0.129784, -127, 127, -127, 127),
    QuantParams(-16.0995, 16.0995, -0.115702, 0.115702, -127, 127, -127, 127),
    QuantParams(-139.502, 139.502, -0.145005, 0.145005, -127, 127, -127, 127),
    QuantParams(-22.8551, 22.8551, -0.182333, 0.182333, -127, 127, -127, 127),
    QuantParams(-55.732,  55.732,  -0.190798, 0.190798, -127, 127, -127, 127),
    QuantParams(-14.0215, 14.0215, -0.159494, 0.159494, -127, 127, -127, 127),
    QuantParams(-33.8009, 33.8009, -0.151049, 0.151049, -127, 127, -127, 127),
    QuantParams(-13.6535, 13.6535, -0.239642, 0.239642, -127, 127, -127, 127)
};




int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		
		
		const char *usage =
				"Reads a prob matrix and converts it to the format expected by the viterbi emulator and simulator\n"
				"Usage: extract_probs_B_accel [options] <ivector-rspecifier> <feats-rspecifier> <AM_filename> <probs_base_filename>\n"
				"NOTE: the final name for the prob file is composed as <probs_base_filename>_<utt_name>.acprobs. <probs_base_filename> can be just a directory, and all the acprobs will be writen inside it as _<utterance_name>.acprobs\n";

		ParseOptions po(usage);
		
		NnetSimpleComputationOptions decodable_opts;
		std::string use_gpu = "no";
		std::string time_log = "";
		std::string quant_log = "";
		bool quantize = false;
		int32 online_ivector_period = 10;
	
		std::string acprobs_base_filename = "";
		std::string probs_wspecifier = "";

  	Timer timer;
		//bool pad_input = true;
		//BaseFloat acoustic_scale = 0.1;
  	decodable_opts.Register(&po);
    po.Register("time-log", &time_log, "Log file to write time measurements");
		po.Register("quant-range-log", &quant_log, "Log file to write input min and max values");
		po.Register("use-gpu", &use_gpu, "Use GPU when possible (yes|no) (default:no)");
		po.Register("acprobs-base-filename", &acprobs_base_filename, "Save probabilities in <probs_base_filename>. If false, it won't try to create the file");
		po.Register("probs-wspecifier", &probs_wspecifier, "Save probabilities");
		po.Register("quantize", &quantize, "Quantize network");

		po.Read(argc, argv);
		if (po.NumArgs() != 3) {
			po.PrintUsage();
			exit(1);
		}

		std::string online_ivector_rspecifier = po.GetArg(1),
								features_rspecifier = po.GetArg(2),
								ac_model_filename = po.GetArg(3);

		std::ofstream quant_o;
		if (quant_log != "") {
			quant_o.open(quant_log);
			if (!quant_o.is_open()) {
				KALDI_ERR << "Could not open quant ranges log file " << quant_log;
			}
			quant_o << "component, w min value, w max value, in min value, in max value, num inputs" << std::endl;
		}


		std::ofstream time_o;
		if (time_log != "") {
			time_o.open(time_log);
			if (!time_o.is_open()) {
				KALDI_ERR << "Could not open time log file " << time_log;
			}
			time_o << "Utterance, frames, time (s)" << std::endl;
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

		int quantized_layers = 0;
		if (quant_o.is_open() || quantize) {
			for (int i = 0; i < am_nnet.GetNnet().NumComponents(); i++) {
				Component *comp = am_nnet.GetNnet().GetComponent(i);
				if (comp->Properties() & kaldi::nnet3::ComponentProperties::kQuantizableComponent) {
					std::cerr << "Quantize Weights of layer " << quantized_layers << std::endl;
					if (quantized_layers >= 24) {
						std::cerr << "Stopped quantization at layer 24" << std::endl;
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

		BaseFloatMatrixWriter posterior_writer(probs_wspecifier);
		if (probs_wspecifier != "") {
			posterior_writer.Open(probs_wspecifier);
			if (!posterior_writer.IsOpen()) {
				std::cerr << "Could not open wspecifier " << probs_wspecifier << std::endl;
			}
		}

		RandomAccessBaseFloatMatrixReader online_ivector_reader(online_ivector_rspecifier);

		SequentialBaseFloatMatrixReader features_reader(features_rspecifier);

		CachingOptimizingCompiler compiler(am_nnet.GetNnet(), decodable_opts.optimize_config);

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
			if (time_o.is_open()) {
				time_o << utt << ", " << features.NumRows() << elapsed << std::endl;
			}

			if (acprobs_base_filename != "") {
				std::ofstream acf(acprobs_base_filename + utt + ".acprobs");
				assert(acf.is_open());
				const Matrix<BaseFloat>& ac_mat = nnet_decodable.GetLogProbs();
				unsigned int num_frames = static_cast<unsigned int>(ac_mat.NumRows());
				unsigned int num_pdfs = static_cast<unsigned int>(ac_mat.NumCols());
				std::vector<float> ac_vec;
				for (MatrixIndexT i = 0; i < ac_mat.NumRows(); i++)
					for (MatrixIndexT j = 0; j < ac_mat.NumCols(); j++)
						ac_vec.push_back( static_cast<float>(ac_mat(i, j)) );
				acf.write((char*)&num_frames, sizeof(unsigned int));
				acf.write((char*)&num_pdfs, sizeof(unsigned int));
				acf.write((char*)ac_vec.data(), sizeof(float) * ac_vec.size());
				acf.close();
			}

			if (posterior_writer.IsOpen()) {
				posterior_writer.Write(utt, nnet_decodable.GetLogProbs());
			}

			frame_count += features.NumRows();
			utt_count++;
		}

		if (quant_o.is_open()) { 
			for (int i = 0; i < am_nnet.GetNnet().NumComponents(); i++) {
				Component *comp = am_nnet.GetNnet().GetComponent(i);
				ComponentStatistics stats;
				comp->GetStatistics(stats);
				quant_o << comp->Type() << ", " << stats.w_range_min << ", " << stats.w_range_max;
				quant_o << ", " << stats.in_range_min << ", " << stats.in_range_max;
				quant_o << ", " << stats.num_inputs << std::endl;
			}
		}

		std::cout << "Finished computing posterior probabilities." << std::endl;
		std::cout << "Total frames is " << frame_count << " for " << utt_count << " utterances." << std::endl;

		if (time_o.is_open()) time_o.close();
		if (quant_o.is_open()) quant_o.close();

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
