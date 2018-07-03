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
				"Extract statistics from a Nnet3 network";

		ParseOptions po(usage);
		
		NnetSimpleComputationOptions decodable_opts;

		int32 online_ivector_period = 10;
		decodable_opts.Register(&po);

		Timer timer;
		//bool pad_input = true;
		//BaseFloat acoustic_scale = 0.1;

		po.Read(argc, argv);
		if (po.NumArgs() < 1 || po.NumArgs() > 1) {
			po.PrintUsage();
			exit(1);
		}

//		std::string online_ivector_rspecifier = po.GetArg(1);
//		std::string features_rspecifier = po.GetArg(2);
		std::string ac_model_filename = po.GetArg(1);
//		std::string posterior_wspecifier = po.GetArg(4);
	
		std::ofstream elapsed_time_o("nnet_time.csv");
		std::ofstream nnet_stats_o("nnet_statistics.csv");
		std::ofstream nnet_full_stats_o("nnet_full_stats.csv");

		TransitionModel trans_model;
//		AmNnetSimple am_nnet;
		Nnet am_nnet;
		{
			bool binary;
			Input ki(ac_model_filename, &binary);
//			trans_model.Read(ki.Stream(), binary);
			am_nnet.Read(ki.Stream(), binary);
//			SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
//			SetDropoutTestMode(true, &(am_nnet.GetNnet()));
//			CollapseModel(CollapseModelConfig(), &(am_nnet.GetNnet()));
		}

		int32 num_inputs = 0;
		int32 num_outputs = 0;
		int32 num_weights = 0;
		int32 sizeBytes = 0;

		nnet_stats_o << "name, type, inputs, outputs, weights, size (bytes)" << std::endl;
		//Nnet network = am_nnet.GetNnet();
		Nnet network = am_nnet;
		for (int i = 0; i < network.NumNodes(); i++) {
			NetworkNode node = network.GetNode(i);
			switch(node.node_type) {
				case kInput:
					num_inputs += node.dim;
					break;
				case kComponent:
				{
					int32 comp_inputs, comp_outputs, comp_weights;
					Component *component;
					component = network.GetComponent(node.u.component_index);
					nnet_stats_o << network.GetNodeName(i) << ", " << component->Type() << ", ";
					if (component->Properties() & kUpdatableComponent) {
						UpdatableComponent* upComponent = dynamic_cast<UpdatableComponent*>(component);
						comp_weights = upComponent->NumParameters();
						comp_inputs = upComponent->InputDim();
						comp_outputs = upComponent->OutputDim();

						num_weights += comp_weights;
						sizeBytes += comp_weights*sizeof(BaseFloat);
					// DEBUG
//					ComponentStatistics statistics;
//					component->GetStatistics(statistics);
//					nnet_stats_o << network.GetNodeName(i) << ", " << component->Type() << ", ";
//					if (statistics.valid) {
						// If statistics are valid, report them
//						num_weights += statistics.numWeights;
//						sizeBytes += statistics.sizeBytes;
//						nnet_stats_o << statistics.numInputs << ", " << statistics.numOutputs << ", ";
//						nnet_stats_o << statistics.numWeights << ", " << statistics.sizeBytes << std::endl;
//					} else {
//						nnet_stats_o << "[NOT VALID]" << std::endl;
//					}

						
						nnet_stats_o << comp_inputs << ", " << comp_outputs << ", " << comp_weights << ", ";
						nnet_stats_o << comp_weights*sizeof(BaseFloat) << std::endl;
					}else {
						nnet_stats_o << component->InputDim() << ", " << component->OutputDim() << ", 0, 0" << std::endl;
					}
					break;
				}
				case kDescriptor:
					std::cerr << "I don't know what to do with kDescriptor" << std::endl;
					break;
				case kDimRange:
					std::cerr << "I don't know what to do with kDimRange" << std::endl;
					break;
				case kNone:
					std::cerr << "I don't know what to do with kNone" << std::endl;
					break;
				default:
					std::cerr << "Error. Wrong node type." << std::endl;
					break;
			}
			// Recorrer nodos
				// si el nodo es kinput, acumular como input de la red
				// si el nodo es koutput, acumular como output de la red
				// si el nodo es kcomponent:
					// component = network.GetComponent(nodo.u.component_index);
					// component.GetStatistics();
		}


		nnet_full_stats_o << "inputs, outpus, weights, size (bytes)" << std::endl;
		nnet_full_stats_o << num_inputs << ", " << num_outputs << ", " <<  num_weights << ", ";
		nnet_full_stats_o << sizeBytes << std::endl;

/*
		RandomAccessBaseFloatMatrixReader online_ivector_reader(online_ivector_rspecifier);

		SequentialBaseFloatMatrixReader features_reader(features_rspecifier);
		BaseFloatMatrixWriter posterior_writer(posterior_wspecifier);

		CachingOptimizingCompiler compiler(am_nnet.GetNnet(), decodable_opts.optimize_config);


		kaldi::int64 frame_count = 0;

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
			elapsed_time_o << utt << ", " << elapsed << ", " << features.NumRows() << ", " << (elapsed*100.0/features.NumRows()) << std::endl;

			posterior_writer.Write(utt, nnet_decodable.GetLogProbs());
			//nnet_decodable.WriteProbsToTable(posterior_writer, utt);
			frame_count += features.NumRows();
		}
		std::cout << "Terminado" << std::endl;

*/
		elapsed_time_o.close();
		nnet_stats_o.close();
		nnet_full_stats_o.close();

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
