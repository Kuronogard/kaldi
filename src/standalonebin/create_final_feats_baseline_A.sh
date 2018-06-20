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
#			- <utt_set>
#					- mfcc/
#					- cmvn/
#					- fmllr_trans/
#	- models
#     - lda_mllr_related/
#						- final.mat
# 		- fmllr_related/
#       		- final.alimdl
#       		- final.mdl
#       		- HCLG.fst
#			- words.txt

# The utterances directories  must include
# - utt_dir/
#	- utt2spk
#	- spk2utt
#	- cmvn.scp   (refering to transforms in cmvn.ark)
#	- feats.scp  (refering to features in mfcc.ark)
#	- cmvn.ark
#	- mfcc.ark
#	- wav.scp


# name -> The name of the utterance set, something like "test_clear". It is going to be 
# used appended to file names as in raw_mfcc_test_clear.ark
# utt_dir -> the directory where the utterance script files (wav.scp) is located
# feat_dir -> The directory to store the features. Something like "feats/librispeech"
name=$1
utt_dir=$2
feat_dir=$3



utt2spk_file=${utt_dir}/utt2spk
spk2utt_file=${utt_dir}/spk2utt
wav_scp_file=${utt_dir}/wav.scp
mfcc_ark_file=${feat_dir}/mfcc/raw_mfcc_${name}.ark
mfcc_scp_file=${feat_dir}/mfcc/raw_mfcc_${name}.scp
cmvn_ark_file=${feat_dir}/cmvn/cmvn_${name}.ark
cmvn_scp_file=${feat_dir}/cmvn/cmvn_${name}.scp
transformed_feats_ark_file=${feat_dir}/fmllr/feat.ark

lda_transform_file=models/lda_mllr_related/final.mat
symbol_table_file=models/words.txt
fmllr_first_pass_mdl_file=models/fmllr_related/final.alimdl
fmllr_second_pass_mdl_file=models/fmllr_related/final.mdl
fmllr_decode_graph_file=models/fmllr_related/HCLG.fst
fmllr_trans_file=${feat_dir}/fmllr_trans/trans.${name}

temp_dir=temp
temp_lattice_file=${temp_dir}/lat.${name}.gz
temp_lattice_2_file=${temp_dir}/lat.${name}.2.gz
temp_fmllr_trans_file=${temp_dir}/pre_trans.${name}
temp_fmllr_trans_2_file=${temp_dir}/temp_trans.${name}

# Whenever i have to compute transforms from a new utterance set, i would prepare it using the kaldi way, and then compute, for each utterance:
#	- [for nnet6a, nnet7a]: 16-dim mfcc, fmllr
#	- [for tddn]: 40-dim mfcc, ivectors]


# Compute 16-dim mfcc features
compute-mfcc-feats --verbose=2 --use-energy=false scp,p:${wav_scp_file} ark:- | \
copy-feats --compress=true ark:- ark,scp:${mfcc_ark_file},${mfcc_scp_file}


# Compute cmvn stats
compute-cmvn-stats --spk2utt=ark:${utt2spk_file} scp:${mfcc_scp_file} ark,scp:${cmvn_ark_file},${cmvn_scp_file}


# Generate per-utterance fMLLR transforms
# ======================================================
# 1. use final.alimdl to decode the utterance (steps/decode.sh -> gmm-latgen-faster)
# 2. use final.mdl to compute first-pass fMLLR transforms with the previous calculated lattices (gmm-est-fmllr)
# 3. use final.mdl to decode the utterance again  (gmm-latgen-faster)
# 4. use final.mdl to estimate fMLLR transforms again with the previous calculated lattices (gmm-est-fmllr)
# 5. compose transforms from both estimation steps (compose-transforms)

feats="ark,s,cs:apply-cmvn --utt2spk=ark:${utt2spk_file} scp:${cmvn_scp_file} scp:${mfcc_scp_file} ark:- | \
splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
transform-feats ${lda_transform_file} ark:- ark:- |"

# Decode the utterance with final.alimdl
gmm-latgen-faster --max-active=2000 --beam=10.0 --lattice-beam=6.0 --acoustic-scale=0.83333 --allow-partial=true --word-symbol-table=${symbol_table_file} ${fmllr_first_pass_mdl_file} ${fmllr_decode_graph_file} "${feats}" "ark:|gzip -c > ${temp_lattice_file}"


# Compute first-pass fMLLR transforms with final.mdl
gunzip -c ${temp_lattice_file} | \
lattice-to-post --acoustic-scale=0.083333 ark:- ark:- | \
weight-silence-post 0.01 1:2:3:4:5:6:7:8:9:10 ${fmllr_first_pass_mdl_file} ark:- ark:- | \
gmm-post-to-gpost ${fmllr_first_pass_mdl_file} "${feats}" ark:- ark:- | \
gmm-est-fmllr-gpost --fmllr-update-type=full --spk2utt=ark:${spk2utt_file} ${fmllr_second_pass_mdl_file} "${feats}" ark,s,cs:- ark:${temp_fmllr_trans_file} 


feats_prev="${feats} transform-feats --utt2spk=ark:${utt2spk_file} ark:${temp_fmllr_trans_file} ark:- ark:- |"

# Decode the utterance with final.mdl and pre_trans.1 transforms
gmm-latgen-faster --max-active=7000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.083333 --determinize-lattice=false --allow-partial=true --word-symbol-table=${symbol_table_file} ${fmllr_second_pass_mdl_file} ${fmllr_decode_graph_file} "${feats_prev}" "ark:|gzip -c > ${temp_lattice_2_file}"


# Compute second-pass fMLLR transforms with final.mdl and the previous aligns
lattice-determinize-pruned --acoustic-scale=0.083333 --beam=4.0 "ark:gunzip -c ${temp_lattice_2_file}|" ark:- | \
lattice-to-post --acoustic-scale=0.083333 ark:- ark:- | \
weight-silence-post 0.01 1:2:3:4:5:6:7:8:9:10 ${fmllr_second_pass_mdl_file} ark:- ark:- | \
gmm-est-fmllr --fmllr-update-type=full --spk2utt=ark:${spk2utt_file} ${fmllr_second_pass_mdl_file} "${feats_prev}" ark,s,cs:- ark:${temp_fmllr_trans_2_file}

# Compose transforms from both estimation steps
compose-transforms --b-is-affine=true ark:${temp_fmllr_trans_2_file} ark:${temp_fmllr_trans_file} ark:${fmllr_trans_file}


 
# Apply transforms to raw mfcc features
# This must be done in order to prepare the features the way the DNN expects
apply-cmvn  --utt2spk=ark:${utt2spk_file} scp:${cmvn_scp_file} scp:${mfcc_scp_file} ark:- | \
splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
transform-feats ${lda_transform_file} ark:- ark:- | \
transform-feats --utt2spk=ark:${utt2spk_file} ark:${fmllr_trans_file} ark:- ark:${transformed_feats_ark_file}

# raw_mfcc.scp --> 16-dim mfcc features extracted directly from the signal
#              ==> steps/make_mfcc.sh (compute-mfcc-feats) [per utterance]
# cmvn.scp     --> cmvn normalization
#	       ==> steps/compute_cmvn_stats.sh (compute-cmvn-stats) [per utterance]
# final.mat    --> LDA transform matrix     [per system]
#	       ==> 
# trans.1      --> fMLLR transform matrix   [per utterance
