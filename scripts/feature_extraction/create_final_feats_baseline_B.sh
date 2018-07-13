#!/bin/bash

#
# This script is not intended for direct execution. Instead, it is used to document 
# the process of creating the features.
#

# The expected features for nnet3 models as trained for librispeech are 40-dimensional mfcc concatenated
# with 60-dim i-vectors (the later for speaker adaptation).
# The resulting feature vector has 100 dimensions


# utt_dir -> the directory where the utterance script files (wav.scp) is located
# feat_dir -> The directory to store the features. Something like "feats/librispeech"
utt_dir=$1
feat_dir=$2

system_dir=~/scratch/ASR_systems/baseline_B

for dir in $utt_dir $feat_dir $system_dir; do
	if [ ! -d "${dir}" ]; then
		echo "Directory $dir does not exist."
		exit 0
	fi
done

# source kaldi binaries
if [ -z ${KALDI_ROOT} ]; then
	echo "KALDI_ROOT not defined."
	exit 0
fi


##############################################
#            CONFIGURATION
##############################################

# Configure utterance source files
utt2spk_file=${utt_dir}/utt2spk
spk2utt_file=${utt_dir}/spk2utt
wav_scp_file=${utt_dir}/wav.scp

# Configure model source files
lda_transform_file=${system_dir}/lda/final.mat
# symbol_table_file=${system_dir}/words.txt
diag_ubm_file=${system_dir}/ivector/final.dubm
ivector_extractor_file=${system_dir}/ivector/final.ie
global_cmvn_stats_file=${system_dir}/ivector/global_cmvn.stats

# Destination files (generated by this script)
mfcc_ark_file=${feat_dir}/mfcc/raw_mfcc.ark
mfcc_scp_file=${feat_dir}/mfcc/raw_mfcc.scp
cmvn_ark_file=${feat_dir}/cmvn/cmvn.ark
cmvn_scp_file=${feat_dir}/cmvn/cmvn.scp
ivector_ark_file=${feat_dir}/ivector/ivector.ark
ivector_scp_file=${feat_dir}/ivector/ivector.scp
feats_ark_file=${feat_dir}/feats.ark
feats_scp_file=${feat_dir}/feats.scp

# Temporary files
temp_dir=${feat_dir}/temp
mfcc_conf_file=${temp_dir}/mfcc.conf
online_cmvn_conf_file=${temp_dir}/online_cmvn.conf
splice_conf_file=${temp_dir}/splice.conf
ivector_conf_file=${temp_dir}/ivector_extract.conf


# Check that all the source files exist
for file in $utt2spk $spk2utt $wav_scp_file; do
	if [ ! -f "${file}" ]; then
		echo "Utterance directory not correct. Missing '${file}' file."
		exit 0
	fi
done

for file in $symbol_table_file $lda_transform_file $diag_ubm_file $ivector_extractor_file; do
	if [ ! -f ${file} ]; then
		echo "Model directory not correct. Missing '${file}' file."
		exit 0
	fi
done


# Cleanup
rm -rf ${feat_dir}/*

mkdir ${feat_dir}/mfcc
mkdir ${feat_dir}/cmvn
mkdir ${feat_dir}/ivector

mkdir ${temp_dir}


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
echo "--min-post=0.025" >> ${ivector_conf_file}
echo "--posterior-scale=0.1" >> ${ivector_conf_file}
echo "--max-remembered-frames=1000" >> ${ivector_conf_file}
echo "--max-count=0" >> ${ivector_conf_file}



##############################################
## COMPUTE MFCC (FEATS) AND CMVN (TRANSFORMS)
##############################################

# Compute 40-dim mfcc features
# make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf --cmd "${train_cmd}" data/<test-set>_hires
echo "Compute Mfcc features"
time (compute-mfcc-feats --verbose=2 --config=${mfcc_conf_file} scp,p:${wav_scp_file} ark:temp.ark)
sync
copy-feats --compress=true ark:temp.ark ark,scp:${mfcc_ark_file},${mfcc_scp_file}

rm temp.ark


# Compute cmvn stats
compute-cmvn-stats --spk2utt=ark:${utt2spk_file} scp:${mfcc_scp_file} ark,scp:${cmvn_ark_file},${cmvn_scp_file}

#if ! matrix-sum --binary=false scp:${cmvn_scp_file} - > ${global_cmvn_stats_file} 2> /dev/null; then
#  echo "$0: Error summing cmvn stats"
#  exit 1
#fi



##############################################
## COMPUTE IVECTORS
##############################################

# Generate per-utterance ivectors
#steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
#		data/<testset>_hires exp/nnet3_cleaned/extractor \
#		exp/nnet3_cleaned/ivectors_<testset>_hires
echo "Compute iVectors"
time (ivector-extract-online2 --config=${ivector_conf_file} ark:${spk2utt_file} scp:${mfcc_scp_file} ark:temp.ark)
sync
copy-feats --compress=true ark:temp.ark ark,scp:${ivector_ark_file},${ivector_scp_file}

rm temp.ark

sync
exit

##############################################
## COMPUTE FINAL FEATURE VECTORS
##############################################

# The final feature vector is a concatenation of the Mfcc vector transformed 
# with the corresponding cmvn transforms, and the corresponding ivector.
#
# This script can apply the cmvn transform and leave the concatenation for the baseline_B program

#cmvn_opts=`cat $srcdir/cmvn_opts`
#apply-cmvn $cmvn_opts --utt2spk=ark:${utt2spk_file} scp:${cmvn_scp_file} scp:${mfcc_scp_file} ark,scp:${feats_ark_file},${feats_scp_file}
# ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
 
