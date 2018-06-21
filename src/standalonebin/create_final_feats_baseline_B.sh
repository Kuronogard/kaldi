#!/bin/bash

#
# This script is not intended for direct execution. Instead, it is used to document 
# the process of creating the features.
#

# It seems that

# The expected features for nnet3 models as trained for librispeech are 40-dimensional mfcc concatenated
# with 60-dim i-vectors (the later for speaker adaptation).
# The resulting feature vector has 100 dimensions


# It need the following directory structure
#
# - feats/
#			- <utt_set>
#					- mfcc/
#					- cmvn/
#					- ivectors
#	- models
#			- ivector_related/
#					- 
#     - lda_mllr_related/
#					- final.mat
# 		- fmllr_related/
#       	- final.alimdl
#       	- final.mdl
#       	- HCLG.fst
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
mfcc_ark_file=${feat_dir}/mfcc/raw_mfcc40_${name}.ark
mfcc_scp_file=${feat_dir}/mfcc/raw_mfcc40_${name}.scp
cmvn_ark_file=${feat_dir}/cmvn/cmvn_${name}.ark
cmvn_scp_file=${feat_dir}/cmvn/cmvn_${name}.scp
transformed_feats_ark_file=${feat_dir}/fmllr/feat.ark

lda_transform_file=models/lda_mllr_related/final.mat
symbol_table_file=models/words.txt
fmllr_first_pass_mdl_file=models/fmllr_related/final.alimdl
fmllr_second_pass_mdl_file=models/fmllr_related/final.mdl
fmllr_decode_graph_file=models/fmllr_related/HCLG.fst
fmllr_trans_file=${feat_dir}/fmllr_trans/trans.${name}

ivector_ark_file=${feat_dir}/ivector/ivector_${name}.ark
ivector_scp_file=${feat_dir}/ivector/ivector_${name}.scp

temp_dir=temp
temp_lattice_file=${temp_dir}/lat.${name}.gz
temp_lattice_2_file=${temp_dir}/lat.${name}.2.gz
temp_fmllr_trans_file=${temp_dir}/pre_trans.${name}
temp_fmllr_trans_2_file=${temp_dir}/temp_trans.${name}

mfcc_conf_file=${temp_dir}/mfcc.conf
online_cmvn_conf_file=${temp_dir}/online_cmvn.conf
splice_conf_file=${temp_dir}/splice.conf
ivector_conf_file=${temp_dir}/ivector_extract.conf


diag_ubm_file=models/ivector_related/final.dubm
ivector_extractor_file=models/ivector_related/final.ie
global_cmvn_stats_file=model/ivector_related/global_cmvn.stats


# Whenever i have to compute transforms from a new utterance set, i would prepare it using the kaldi way, and then compute, for each utterance:
#	- [for nnet6a, nnet7a]: 16-dim mfcc, fmllr
#	- [for tddn]: 40-dim mfcc, ivectors]


##############################################
## CREATE CONFIG FILES
##############################################

# Generate mfcc config
echo "# Mfcc configuration" >> ${mfcc_conf_file}
echo "--use-energy=false" >> ${mfcc_conf_file}
echo "--num-mel-bins=40" >> ${mfcc_conf_file}
echo "--num-ceps=40" >> ${mfcc_conf_file}
echo "--low-freq=20" >> ${mfcc_conf_file}
echo "--high-freq=-400" >> ${mfcc_conf_file}

# Generate splice config
echo "--left-context=3" >> ${splice_conf_file}
echo "--right-context=3" >> ${splice_conf_file}

# Generate cmvn config
echo "# config file for apply-cmvn-online" >> ${online_cmvn_conf_file}


# Generate ivector extraction config
echo "--cmvn-config=${online_cmvn_conf_file}" >> ${ivector_conf_file}
echo "--splice-config=${splice_conf_file}" >> ${ivector_conf_file}
echo "--ivector-period=10" >> ${ivector_conf_file}
echo "--lda-matrix=${lda_transform_file}" >> ${ivector_conf_file}
echo "--global-cmvn-stats=${global_cmvn_stats_file}" >> ${ivector_conf_file}
echo "--diag-ubm=${diag_ubm_file}" >> ${ivector_conf_file}
echo "--ivector-extractor=${ivector_extractor_file}" >> ${ivector_conf_file}
echo "--num-gselect=5" >> ${ivector_conf_file}
echo "--min-gpost=0.025" >> ${ivector_conf_file}
echo "--posterior-scale=0.1" >> ${ivector_conf_file}
echo "--max-remembered-frames=1000" >> ${ivector_conf_file}
echo "--max-count=0" >> ${ivector_conf_file}



##############################################
## COMPUTE MFCC (FEATS) AND CMVN (TRANSFORMS)
##############################################

# Compute 40-dim mfcc features
# make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd "${train_cmd}" data/<test-set>_hires
compute-mfcc-feats --verbose=2 --config=${mfcc_conf_file} scp,p:${wav_scp_file} ark:- | \
copy-feats --compress=true ark:- ark,scp:${mfcc_ark_file},${mfcc_scp_file}


# Compute cmvn stats
compute-cmvn-stats --spk2utt=ark:${utt2spk_file} scp:${mfcc_scp_file} ark,scp:${cmvn_ark_file},${cmvn_scp_file}

if ! matrix-sum --binary=false scp:${cmvn_scp_file} - > ${global_cmvn_stats_file} 2> /dev/null; then
  echo "$0: Error summing cmvn stats"
  exit 1
fi



##############################################
## COMPUTE IVECTORS
##############################################

# Generate per-utterance ivectors
#steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
#		data/<testset>_hires exp/nnet3_cleaned/extractor \
#		exp/nnet3_cleaned/ivectors_<testset>_hires
ivector-extract-online2 --config=${ivector_conf_file} ark:${spk2utt_file} scp:${mfcc_scp_file} ark:- | \
copy-feats --compress=true ark:- ark,scp:${ivector_ark_file},${ivector_scp_file}



##############################################
## COMPUTE FINAL FEATURE VECTORS
##############################################

# The final feature vector is a concatenation of the Mfcc vector transformed 
# with the corresponding cmvn transforms, and the corresponding ivector.
#
# This script can apply the cmvn transform and leave the concatenation for the baseline_B program

# cmvn_opts=`cat $srcdir/cmvn_opts`
# feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
# ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
 
