

# It should extract probs and decode for all the utterances in the librispeech test tests
# The test sets should be divided in several parts, so this can be run in parallel


# Extract probs
#extract_probs_B --extra-left-context=0 --extra-right-context=0 --extra-left-context-initial=-1 --extra-right-context-final=-1 --frame-subsampling-factor=3 --frames-per-chunk=50 --acoustic-scale=1.0  ark:ASR_data/features_baseline_B_test_clean_split1/ivector/ivector.ark ark:ASR_data/features_baseline_B_test_clean_split1/mfcc/raw_mfcc.ark ASR_systems/baseline_B/AC_tdnn.mdl ark:temp/probsB.ark experiments/baseline/tdnn_prob_calc_time_3.csv


# Decode
#baseline_B --max-active=7000 --min-active=200 --lattice-beam=8.0 --beam=15.0  AC_tdnn.mdl decode_HCLG.fst ark:../../temp/probsB.ark ../grammar/tgsmall_symbol_table.txt ark:../../temp/words_baseline_B ../grammar/tgsmall_G.fst ../grammar/fglarge_G.carpa decode.log

TEST_SETS="dev_clean test_clean dev_other test_other"
SPLITS=20
#TEST_SETS="test_clean dev_clean"
#SPLITS=4

DST="/home/dpinto/scratch/experiments/baseline_librispeech"
max_running=1

running=0



rm -r ~/scratch/temp/probs ~/scratch/temp/trans
mkdir ~/scratch/temp/probs
mkdir ~/scratch/temp/trans
mkdir ~/scratch/temp/log

for CURR_TEST_SET in ${TEST_SETS}; do
  for (( CURR_SPLIT=1; CURR_SPLIT<=${SPLITS}; CURR_SPLIT++ )); do
    if [ $running -ge $max_running ]; then
      # Wait for 1 process to finish
      wait -n
      let running=running-1
    fi

    feat_dir="/home/dpinto/scratch/ASR_data/features_baseline_B_${CURR_TEST_SET}_split${CURR_SPLIT}"
    dataset="/home/dpinto/scratch/datasets/sets/librispeech-split20-${CURR_TEST_SET}/${CURR_SPLIT}"
    system_dir="/home/dpinto/scratch/ASR_systems/baseline_B"
    grammar_dir="/home/dpinto/scratch/ASR_systems/grammar"
    probs_file="/home/dpinto/scratch/temp/probs/${CURR_TEST_SET}/${CURR_SPLIT}_probs.ark"
    extract_probs_log="${DST}/${CURR_TEST_SET}_${CURR_SPLIT}_nnet_compute.csv"
    decode_log="${DST}/${CURR_TEST_SET}_${CURR_SPLIT}_decode.csv"
    decode_trans="/home/dpinto/scratch/temp/trans/${CURR_TEST_SET}/${CURR_SPLIT}_transcription"

    log_file="/home/dpinto/scratch/temp/log/${CURR_TEST_SET}_${CURR_SPLIT}.log"

    if [ ! -d ~/scratch/temp/probs/${CURR_TEST_SET} ]; then mkdir ~/scratch/temp/probs/${CURR_TEST_SET}; fi
    if [ ! -d ~/scratch/temp/trans/${CURR_TEST_SET} ]; then mkdir ~/scratch/temp/trans/${CURR_TEST_SET}; fi
    if [ ! -d ${feat_dir} ]; then mkdir ${feat_dir}; fi


    EXTRACT_FEATS="/home/dpinto/scratch/repos/kaldi_dev/scripts/feature_extraction/create_final_feats_baseline_B.sh ${dataset} ${feat_dir}"

    EXTRACT_PROBS="extract_probs_B --extra-left-context=0 --extra-right-context=0 --extra-left-context-initial=-1 --extra-right-context-final=-1 --frame-subsampling-factor=3 --frames-per-chunk=50 --acoustic-scale=1.0  ark:${feat_dir}/ivector/ivector.ark ark:${feat_dir}/mfcc/raw_mfcc.ark ${system_dir}/AC_tdnn.mdl ark:${probs_file} ${extract_probs_log}"

    DECODE=" baseline_B --max-active=7000 --min-active=200 --lattice-beam=8.0 --beam=15.0  ${system_dir}/AC_tdnn.mdl ${system_dir}/decode_HCLG.fst ark:${probs_file} ${grammar_dir}/tgsmall_symbol_table.txt ark,t:${decode_trans} ${grammar_dir}/tgsmall_G.fst ${grammar_dir}/fglarge_G.carpa ${decode_log}"

    # launch probability extraction and decoding for a new split
    (${EXTRACT_FEATS} && ${EXTRACT_PROBS} && ${DECODE}) &> ${log_file} &
    let running=running+1
    echo " - Launched ${CURR_TEST_SET}_${CURR_SPLIT}"
  done
done

# Wait for the remaining processes to finish
wait
