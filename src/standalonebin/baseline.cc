// nnet2bin/nnet-latgen-faster.cc

// Copyright      2018   Universitat Politècnica de Catalunya (author: Daniel Pinto)

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



// Baseline system for tests
// features: Mfcc+ivector
// AC: TDNN
// LM: pruned 3-gram
// LM resc: 4-gram


#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"

#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"
#include "base/timer.h"


int main(int argc, char **argv) {
    
    try {
        using namespace kaldi;
        
        
        const char *usage =
            "Baseline ASR system.\n"
            "Usage: baseline <wav-rspecifier>";
        
        ParseOptions po(usage);
        MfccOptions mfcc_opts;
        
        mfcc_opts.Register(&po);
        
        po.Read(argc, argv);
        
        if (po.NumArgs() != 1) {
            po.PrintUsage();
            exit(1);
        }
        
        std::string wav_rspecifier = po.GetArg(1);
        
        
        //-------------------------
        //  Compute FEATURES
        // ------------------------
        
        // Create kaldi .wav reader
        SequentialTableReader<WaveHolder> reader(wav_rspecifier);
        
        
        for(; !reader.Done(); reader.Next()) {
            // Create Mfcc object
            
            Mfcc mfcc(mfcc_opts);
            
            
            std::string utt = reader.Key();
            
            // extract the features from the .wav file (all frames)
            const WaveData &wave_data = reader.Value();
            
            // defaulting to channel 0
            int channel = 0;
            
            SubVector<BaseFloat> waveform(wave_data.Data(), channel);
            Matrix<BaseFloat> features;
            try {
                // 1.0 means no vtln warping
                mfcc.ComputeFeatures(waveform, wave_data.SampFreq(), 1.0 , &features);
                
            } catch (...) {
                KALDI_WARN << "Failed to compute features for utterance " << utt;
                continue;
            }
            
            
            // extract i-vectors from the features
            
        
        }
        
/*
        
        //-------------------------
        //  generate LATTICE
        // ------------------------
        std::string model_in_filename = "model";
        
        TransitionModel trans_model;
        AmNnetSimple am_nnet;
        {
            bool binary;
            Input ki(model_in_filename, &binary);
            trans_model.Read(ki.Stream(), binary);
            am_nnet.Read(ki.Stream, binary);
            SetBatchnormTestMode(true, &(am_nnet.GetNet()));
            SetDropoutTestMode(true, &(am_nnet.GetNet()));
            CollapseModel(CollapseModelConfig(), &(am_nnet.GetNet()));
        }
        
        bool determinize=true;
        std::string fst_in_str="decoder_fst";
        
        Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
        CompactLattice clat;
        
        {
            // decode the utterance (generate lattice)
            LatticeFasterDecoder decoder(*decode_fst, config);
            DecodableAmNnetSimple nnet_decodable(
                decodable_opts, trans_model, am_nnet,
                features, ivector, online_ivectors,
                online_ivector_period, &compiler);
                
            if (!decoder.Decode(&decodable)) {
                KARLDI_WARM << "Failed to decode file " << utt;
                continue;
            }
            
            if (!decoder.ReachedFinal()) {
                KALDI_WARN << "Decoder final not reached";
            }
         
            // get lattice and determinize
            Lattice lat;
            decoder.GetRawLattice(&lat);
            fst::Connect(&lat) // I don't know what this is for
            if (determinize) {
                DeterminizeLatticePhonePrunedWrapper(
                    trans_model, &lat, decoder.GetOptions().lattice_beam,
                    &clat, decoder.GetOptions.det_opts);
            }
        }
        
        
        
        //-------------------------
        //  rescore with the LANGUAGE MODEL
        // ------------------------
        std::string lm_to_subtract_rxfilename = "lm_old";
        std::string lm_to_add_rxfilename = "lm_new"
        
        VectorFst<StdArc> *lm_to_subtract_fst = fst::ReadAndPrepareLmFst(
            lm_to_subtract_rxfilename);
            
        VectorFst<StdArc> *lm_to_add_fst = NULL;
        ConstArpaLM const_arpa;
        ReadKaldiObject(lm_to_add_rxfilename, &const_arpa)
        
        
        fst::BackoffDeterministicOnDemandFst<StdArc> lm_to_subtract_det_backoff(
            *lm_to_subtract_fst);
        fst::ScaleDeterministicOnDemandFst lm_to_subtract_det_scale(
            -lm_scale, &lm_to_subtract_det_backoff);
        
        fst::DeterministicOnDemandFst<StdArc> *lm_to_add_orig = NULL;
        fst::DeterministicOnDemandFst<StdArc> *lm_to_add = NULL;
        
        lm_to_add = new ConstArpaLmDeterministicFst(const_arpa);
        
        if (lm_scale != 1.0) {
            lm_to_add_orig = lm_to_add;
            lm_to_add = new fst::ScaleDeterministicOnDemandFst(
                lm_scale, lm_to_add_orig); 
        }
        
        
        TopSortCompactLatticeIfNeeded(&clat);
        
        fst::ComposeDeterministicOnDemandFst<FstArc> combined_lms(
            &fst_to_subtract_det_scale, lm_to_add);
        
        CompactLattice composed_clat;
        ComposeCompactLatticePruned(compose_opts, clat, 
            &combined_lms, &composed_clat);
        
        // rescore with the language model
        
        // get best path
        
*/
        
        return 0;
    
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
