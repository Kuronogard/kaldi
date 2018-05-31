#!/bin/bash

# Generate results as in the RESULTS file usign the commands provided in said file

. cmd.sh
. path.sh

output=exp_statistics/wer.txt

systems="tri4b nnet5a_clean_100_gpu"
test_sets="dev_clean test_clean dev_other test_other"
lms="tgsmall tgmed tgmed_carpa tglarge fglarge rnnlm-lstm"

rm -f ${output}

(
for SYSTEM in ${systems}; do
  for TEST in ${test_sets}; do 
    for LM in ${lms}; do 
      grep WER exp/${SYSTEM}/decode_${LM}_${TEST}/wer* | best_wer.sh
    done
    echo
  done
done
) &>> ${output}
