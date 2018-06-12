// standalone/baseline_A

// Copyright      2018   Universitat Polit√cnica de Catalunya (author: Dennis Pinto)
//					     Based on nnet2bin/nnet-latgen-faster.cc

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


// "ark,s,cs:apply-cmvn  --utt2spk=ark:data/test_clean/split20/1/utt2spk scp:data/test_clean/split20/1/cmvn.scp scp:data/test_clean/split20/1/feats.scp ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/nnet6a_clean_460_gpu/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/test_clean/split20/1/utt2spk ark:exp/tri5b/decode_tgsmall_test_clean/trans.1 ark:- ark:- |"

// Get the features and transform them in the way nnet6 expects
// apply-cmvn  --utt2spk=ark:data/test_clean/split20/1/utt2spk scp:data/test_clean/split20/1/cmvn.scp scp:data/test_clean/split20/1/feats.scp ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/nnet6a_clean_460_gpu/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/test_clean/split20/1/utt2spk ark:exp/tri5b/decode_tgsmall_test_clean/trans.1 ark:- ark:feat.ark 

// TODO:
// Look for a way to obtain cmvn.scp, feats.scp, final.mat trans

// feats.scp is probably just a kaldi's 'script' file pointing to the feat files related to the specific test
// 


#include <iostream>
#include <fstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/kaldi-fst-io.h"
#include "decoder/decoder-wrappers.h"
#include "nnet2/decodable-am-nnet.h"
#include "base/timer.h"

#include "fstext/fstext-lib.h"
#include "fstext/kaldi-fst-io.h"
#include "lm/const-arpa-lm.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/compose-lattice-pruned.h"



bool DecodeUtteranceLattice_();


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
		typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;
		using fst::VectorFst;
		using fst::ReadFstKaldi;

    const char *usage =
        "Generate lattices using neural net model.\n"
        "Usage: baseline_A [options] <nnet-in-filename> <fst-in-filename> <feature-rspecifier>\n";
    ParseOptions po(usage);
    Timer timer;
		Timer global_timer;
    BaseFloat acoustic_scale = 0.1;
		bool pad_input = true;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
		
		ComposeLatticePrunedOptions compose_opts;
		BaseFloat lm_scale = 1.0;
		bool add_const_arpa = false;


    //po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Read(argc, argv);

    if (po.NumArgs() != 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string ac_model_filename = po.GetArg(1),
        decode_fst_filename = po.GetArg(2),
				posteriors_rspecifier = po.GetArg(3),		
				word_symbol_filename = po.GetArg(4),
				words_wspecifier = po.GetArg(5),
				lm_to_subtract_filename = po.GetArg(6),
				lm_to_add_filename = po.GetArg(7);

		std::ofstream time_o("decode_time.csv");


		// Decode objects
    TransitionModel trans_model;
		{
      bool binary;
      Input ki(ac_model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }

		//SequentialBaseFloatCuMatrixReader feature_reader(feature_rspecifier);
		SequentialBaseFloatMatrixReader posteriors_reader(posteriors_rspecifier);
		Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(decode_fst_filename);

		// lmrescore objects
		VectorFst<StdArc> *lm_to_subtract_fst = fst::ReadAndPrepareLmFst(lm_to_subtract_filename);
		ConstArpaLm const_arpa;

		ReadKaldiObject(lm_to_add_filename, &const_arpa);

		fst::BackoffDeterministicOnDemandFst<StdArc> lm_to_subtract_det_backoff(*lm_to_subtract_fst);
		fst::ScaleDeterministicOnDemandFst lm_to_subtract_det_scale(-lm_scale, &lm_to_subtract_det_backoff);

		fst::DeterministicOnDemandFst<StdArc> *lm_to_add_orig = NULL;
		fst::DeterministicOnDemandFst<StdArc> *lm_to_add = new ConstArpaLmDeterministicFst(const_arpa);

		if (lm_scale != 1.0) {
			lm_to_add_orig = lm_to_add;
			lm_to_add = new fst::ScaleDeterministicOnDemandFst(lm_scale, lm_to_add_orig);
		}

		time_o << "utterance, global, decode, lmrescore, path_extraction, global_RTF" << std::endl;

		// Symbol table and word writer
		fst::SymbolTable *word_syms;
		if (!(word_syms = fst::SymbolTable::ReadText(word_symbol_filename))) {
			KALDI_ERR << "Could not read symbol table from " << word_symbol_filename;
		}

		Int32VectorWriter words_writer(words_wspecifier);


		//
		//  DECODE
		//timer.Reset();

    kaldi::int64 frame_count = 0;
    int32 num_success = 0, num_fail = 0;
		int32 num_success_lm = 0, num_fail_lm = 0;
		{
			LatticeFasterDecoder decoder(*decode_fst, config);

			for (; !posteriors_reader.Done(); posteriors_reader.Next()) {
				std::string utt = posteriors_reader.Key();
				
				double decode_elapsed;
				double global_elapsed;
				double lmrescore_elapsed;
				double bestpath_elapsed;
				const Matrix<BaseFloat> &posteriors (posteriors_reader.Value());


				std::cout << "Decoding file " << utt << std::endl;

				if (posteriors.NumRows() == 0) {
					KALDI_WARN << "Zero-length utterance: " << utt;
					num_fail++;
					continue;
				}

				DecodableAmNnetWithIO nnet_decodable(trans_model);
				
				// Execute one of the options below
				//nnet_decodable.ComputeFromModel(am_nnet, features, pad_input, acoustic_scale);
				//nnet_decodable.ReadProbsFromFile(posteriors_filename, true);
				nnet_decodable.ReadProbsFromMatrix(posteriors);
				
				global_timer.Reset();
				timer.Reset();
				if (!decoder.Decode(&nnet_decodable)) {
					KALDI_WARN << "Failed to decode file " << utt;
					num_fail++;
					continue;
				}
				decode_elapsed = timer.Elapsed();

				if (!decoder.ReachedFinal()) {
					KALDI_WARN << "No final state reached for file " << utt;
				}

				frame_count += posteriors.NumRows();
				num_success++;

				Lattice lat;
				CompactLattice clat;
				decoder.GetRawLattice(&lat);

				timer.Reset();
				fst::Connect(&lat);

				if ( !DeterminizeLatticePhonePrunedWrapper (
								trans_model,
								&lat,
								decoder.GetOptions().lattice_beam,
								&clat,
								decoder.GetOptions().det_opts) ) {
				   KALDI_WARN << "Determinization finished earlier than the beam for file " << utt;
				}


				// LM RESCORE
				TopSortCompactLatticeIfNeeded(&clat);

				fst::ComposeDeterministicOnDemandFst<StdArc> combined_lms(&lm_to_subtract_det_scale, lm_to_add);

				CompactLattice composed_clat;
				ComposeCompactLatticePruned(compose_opts, clat, &combined_lms, &composed_clat);
				if (composed_clat.NumStates() == 0) {
					num_fail_lm++;
					continue;
				}
				
				lmrescore_elapsed = timer.Elapsed();
				num_success_lm++;



				std::cout << "Decoded file " << utt << std::endl;

				//
				// GET BEST PATH
				//


				// lattice-scale (--inv_acoustic_scale=(7 to 17))
				// lattice-add-penalty (--word_ins_penalty=(0.0, 0.5 1.0))
				// lattice-best-path
				double acoustic2lm_scale = 1.0;
				double lm2acoustic_scale = 1.0;


				std::vector<std::vector<double> > scale(2);
				scale[0].resize(2);
				scale[1].resize(2);
				scale[0][0] = lm_scale;
				scale[0][1] = acoustic2lm_scale;
				scale[1][0] = lm2acoustic_scale;
				scale[1][1] = acoustic_scale;
				
				timer.Reset();
				ScaleLattice(scale, &composed_clat);

				BaseFloat word_ins_penalty = 0.0;
				AddWordInsPenToCompactLattice(word_ins_penalty, &composed_clat);


				CompactLattice clat_best_path;
				Lattice best_path;

				CompactLatticeShortestPath(composed_clat, &clat_best_path);
				bestpath_elapsed = timer.Elapsed();
				global_elapsed = global_timer.Elapsed();				

				ConvertLattice(clat_best_path, &best_path);				


				//VectorFst<LatticeArc> decoded;
				//if (!decoder.GetBestPath(&decoded)) {
				//	KALDI_ERR << "Failed to get traceback for utterance " << utt;
				//}



				std::vector<int32> alignment;
				std::vector<int32> words;
				LatticeWeight weight;
				GetLinearSymbolSequence(best_path, &alignment, &words, &weight);

				// write words
				words_writer.Write(utt, words);


				// Print words
				std::cout << utt << ": ";
				for (size_t i = 0; i < words.size(); i++) {
					std::string s = word_syms->Find(words[i]);
					if (s == "") s = "<ERR>";
					std::cout << s << ' ';
				}
				std::cout << std::endl;

				// Print time measurements to csv file
				time_o << utt << ", " << global_elapsed << ", " << decode_elapsed << ", ";
				time_o << lmrescore_elapsed << ", " << bestpath_elapsed << ", ";
				time_o << posteriors.NumRows() << ", " << (global_elapsed*100.0/posteriors.NumRows()); 
				time_o << std::endl;
				
			}
		}
		delete decode_fst; // delete this only after decoder goes out of scope.
		time_o.close();

    //double elapsed = timer.Elapsed();
/*
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";
*/
    if (num_success != 0) return 0;
    else return 1;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


