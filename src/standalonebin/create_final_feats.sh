#!/bin/bash

#
# This script is not intended for direct execution. Instead, it is used to document 
# the process of creating the features.
#


# The expected features for nnet2 models as trained for librispeech are 16-dimensional mfcc transformed
# with cmvn normalization, LDA transforms and fMLLR transforms (the later for speaker adaptation).
# The resulting feature vector has 40 dimensions
# It seems that the dnn expects a window with 7 frames (3 frames for left context and 3 for right context), but i don't see that

# It need the following directory structure
#
# - feats/
#	- data/
# 		- mfcc/
#		- fmllr/
#	- models
#       	- lda_mllr_related/
#			- final.mat
# 		- fmllr_related/
#       		- final.alimdl
#       		- final.mdl
#       		- HCLG.fst

# The utterancesdirectories  must include
# - utt_dir/
#	- utt2spk
#	- spk2utt
#	- cmvn.scp   (refering to transforms in cmvn.ark)
#	- feats.scp  (refering to features in mfcc.ark)
#	- cmvn.ark
#	- mfcc.ark
#	- wav.scp


name=$1
utt_dir=$2
feat_dir=$3

# Whenever i have to compute transforms from a new utterance set, i would prepare it using the kaldi way, and then compute, for each utterance:
#	- [for nnet6a, nnet7a]: 16-dim mfcc, fmllr
#	- [for tddn]: 40-dim mfcc, ivectors]


# Compute 16-dim mfcc features
compute-mfcc-feats --verbose=2 --use-energy=false scp,p:${utt_dir}/wav.scp ark:- | \
copy-feats --compress=true ark:- ark,scp:${feat_dir}/data/mfcc/raw_mfcc_${name}.ark,${feat_dir}/data/mfcc/raw_mfcc_${name}.scp


# Compute cmvn stats
compute-cmvn-stats --spk2utt=ark:${utt_dir}/spk2utt scp:${feat_dir}/data/raw_mfcc_${name}.scp ark,scp:${feat_dir}/data/cmvn/cmvn_${name}.ark,${feat_dir}/data/cmvn/cmvn_${name}.scp


# Generate per-utterance fMLLR transforms
# ======================================================
# 1. use final.alimdl to decode the utterance (steps/decode.sh -> gmm-latgen-faster)
# 2. use final.mdl to compute first-pass fMLLR transforms with the previous calculated lattices (gmm-est-fmllr)
# 3. use final.mdl to decode the utterance again  (gmm-latgen-faster)
# 4. use final.mdl to estimate fMLLR transforms again with the previous calculated lattices (gmm-est-fmllr)
# 5. compose transforms from both estimation steps (compose-transforms)

feats="ark,s,cs:apply-cmvn --utt2spk=ark:data/test_clean/split20/1/utt2spk scp:data/test_clean/split20/1/cmvn.scp scp:data/test_clean/split20/1/feats.scp ark:- | \
splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
transform-feats ex/tri6b/final.mat ark:- ark:- |"

# Decode the utterance with final.alimdl
gmm-latgen-faster --max-active=2000 --beam=10.0 --lattice-beam=6.0 --acoustic-scale=0.83333 --allow-partial=true --word-symbol-table=exp/tri6b/graph_tgsmall/words.txt exp/tri6b/final.alimdl exp/tri6b/graph_tgsmall/HCLG.fst "${feats}" "ark:|gzip -c > exp/tri6b/decode_tgsmall_test_clean.si/lat.1.gz"


# Compute first-pass fMLLR transforms with final.mdl
gunzip -c exp/tri6b/decode_tgsmall_test_clean.si/lat.1.gz | \
lattice-to-post --acoustic-scale=0.083333 ark:- ark:- | \
weight-silence-post 0.01 1:2:3:4:5:6:7:8:9:10 exp/tri6b/final.alimdl ark:- ark:- | \
gmm-post-to-gpost exp/tri6b/final.alimdl "${feats}" ark:- ark:- | \
gmm-est-fmllr-gpost --fmllr-update-type=full --spk2utt=ark:data/test_clean/split20/1/spk2utt exp/tri6b/final.mdl "${feats}" ark,s,cs:- ark:exp/tri6b/decode_tgsmall_test_clean/pre_trans.1 


feats_prev="${feats} transform-feats --utt2spk=ark:data/test_clean/split20/1/utt2spk ark:exp/tri6b/decode_tgsmall_test_clean/pre_trans.1 ark:- ark:- |"

# Decode the utterance with final.mdl and pre_trans.1 transforms
gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --determinize-lattice=false --allow-partial=true --word-symbol-table=exp/tri6b/graph_tgsmall/words.txt exp/tri6b/final.mdl exp/tri6b/graph_tgsmall/HCLG.fst "${feats_prev}" "ark:|gzip -c > exp/tri6b/decode_tgsmall_test_clean/lat.tmp.1.gz"

# Compute second-pass fMLLR transforms with final.mdl and the previous aligns

lattice-determinize-pruned --acoustic-scale=0.083333 --beam=4.0 "ark:gunzip -c exp/tri6b/decode_tgsmall_test_clean/lat.tmp.1.gz|" ark:- | \
lattice-to-post --acoustic-scale=0.083333 ark:- ark:- | \
weight-silence-post 0.01 1:2:3:4:5:6:7:8:9:10 exp/tri6b/final.mdl ark:- ark:- | \
gmm-est-fmllr --fmllr-update-type=full --spk2utt=ark:data/test_clean/split20/1/spk2utt exp/tri6b/final.mdl "${feats_prev}" ark,s,cs:- ark:exp/tri6b/decode_tgsmall_test_clean/trans_tmp.1

# Compose transforms from both estimation steps
compose-transforms --b-is-affine=true ark:exp/tri6b/decode_tgsmall_test_clean/trans_tmp.1 ark:exp/tri6b/decode_tgsmall_test_clean/pre_trans.1 ark:exp/tri6b/decode_tgsmall_test_clean/trans.1


 
# Apply transforms to raw mfcc features
# This must be done in order to prepare the features the way the DNN expects
apply-cmvn  --utt2spk=ark:data/test_clean/split20/1/utt2spk scp:data/test_clean/split20/1/cmvn.scp scp:raw_mfcc.scp ark:- | \
splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
transform-feats exp/nnet6a_clean_460_gpu/final.mat ark:- ark:- | \
transform-feats --utt2spk=ark:data/test_clean/split20/1/utt2spk ark:exp/tri5b/decode_tgsmall_test_clean/trans.1 ark:- ark:feat.ark

# raw_mfcc.scp --> 16-dim mfcc features extracted directly from the signal
#              ==> steps/make_mfcc.sh (compute-mfcc-feats) [per utterance]
# cmvn.scp     --> cmvn normalization
#	       ==> steps/compute_cmvn_stats.sh (compute-cmvn-stats) [per utterance]
# final.mat    --> LDA transform matrix     [per system]
#	       ==> 
# trans.1      --> fMLLR transform matrix   [per utterance
