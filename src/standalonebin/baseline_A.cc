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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/kaldi-fst-io.h"
#include "decoder/decoder-wrappers.h"
#include "nnet2/decodable-am-nnet.h"
#include "base/timer.h"



bool DecodeUtteranceLattice_();


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;
		using fst::VectorFst;

    const char *usage =
        "Generate lattices using neural net model.\n"
        "Usage: baseline_A [options] <nnet-in-filename> <fst-in-filename> <feature-rspecifier>\n";
    ParseOptions po(usage);
    Timer timer;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;

    std::string word_syms_filename;
    config.Register(&po);
    //po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string ac_model_filename = po.GetArg(1),
        decode_fst_filename = po.GetArg(2),
				feature_rspecifier = po.GetArg(3),		
				word_symbol_filename = po.GetArg(4),
				words_wspecifier = po.GetArg(5);


    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(ac_model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }


		fst::SymbolTable *word_syms;
		if (!(word_syms = fst::SymbolTable::ReadText(word_symbol_filename))) {
			KALDI_ERR << "Could not read symbol table from " << word_symbol_filename;
		}

		Int32VectorWriter words_writer(words_wspecifier);


    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;


		//
		//  DECODE
		//
		SequentialBaseFloatCuMatrixReader feature_reader(feature_rspecifier);

		Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(decode_fst_filename);
		//timer.Reset();

		{
			LatticeFasterDecoder decoder(*decode_fst, config);

			for (; !feature_reader.Done(); feature_reader.Next()) {
				std::string utt = feature_reader.Key();
				const CuMatrix<BaseFloat> &features (feature_reader.Value());


				std::cout << "Decoding file " << utt << std::endl;

				if (features.NumRows() == 0) {
					KALDI_WARN << "Zero-length utterance: " << utt;
					num_fail++;
					continue;
				}

				bool pad_input = true;
				DecodableAmNnetWithIO nnet_decodable(trans_model);
				
				// Execute one of the options below
				nnet_decodable.ComputeFromModel(am_nnet, features, pad_input, acoustic_scale);
				// nnet_decodable.ReadProbsFromFile(posteriors_file, true);

				if (!decoder.Decode(&nnet_decodable)) {
					KALDI_WARN << "Failed to decode file " << utt;
					num_fail++;
					continue;
				}

				if (!decoder.ReachedFinal()) {
					KALDI_WARN << "No final state reached for file " << utt;
				}

				frame_count += features.NumRows();
				num_success++;

				Lattice lat;
				decoder.GetRawLattice(&lat);

/*
				if ( !DeterminizeLatticePhonePrunedWrapper (
								trans_model,
								&lat,
								decoder.GetOptions.lattice_beam,
								&clat,
								decoder.GetOptions().det_opts) ) {
				   KALDI_WARN << "Determinization finished earlier than the beam for file " << utt;
				}
*/
				std::cout << "Decoded file " << utt << std::endl;

				//
				// GET BEST PATH
				//
				VectorFst<LatticeArc> decoded;
				if (!decoder.GetBestPath(&decoded)) {
					KALDI_ERR << "Failed to get traceback for utterance " << utt;
				}



				std::vector<int32> alignment;
				std::vector<int32> words;
				LatticeWeight weight;
				GetLinearSymbolSequence(decoded, &alignment, &words, &weight);

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

				
			}
		}
		delete decode_fst; // delete this only after decoder goes out of scope.

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


