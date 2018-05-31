#!/bin/bash


# Extract script execution time from log files
# As a temptative aproximation of the execution time of each part of the ASR system


# tri4b
root_dir=`pwd`
output=exp_statistics/times.txt
output_csv=exp_statistics/times.csv

test_dir=${root_dir}/exp

systems="nnet5a_clean_100_gpu"
test_sets="dev_clean test_clean dev_other test_other"
rescore_lms="tgmed tgmed_carpa tglarge fglarge rnnlm-lstm"


rm -f ${output}
rm -f ${output_csv}

touch ${output}
touch ${output_csv}

# decode_tgsmall_test_clean_rnnlm-lstm/log
# print csv header
echo -n "test, frames, " >> ${output_csv}
echo ${rescore_lms} | sed 's/ /, /g' >> ${output_csv}


(
for SYSTEM in ${systems}; do
	echo
	echo ${SYSTEM}
	echo ===============
	echo


	for TEST in ${test_sets}; do

		echo
		echo ${TEST}
		echo --------------
		
		for SPLIT in {1..20}; do
			
			# DECODE step
			FILE=${test_dir}/${SYSTEM}/decode_tgsmall_${TEST}/log/decode.${SPLIT}.log
			if [ ! -f $FILE ]; then 
				STATUS="TEST NOT DONE"; 
				TIME=0
			else 
				STATUS=" ";
				TIME=`cat ${FILE} | grep "elapsed time" | awk '{print $(NF-1)}'` 
			fi

                        FRAMES=`cat ${FILE} | grep "Overall log-likelihood per frame" | awk '{print $(NF-1)}'`
                        echo "SPLIT ${SPLIT}, frames : ${FRAMES} "
			printf "  %15s: %17s: [%5d] secs %s\n"  "tgsmall" "`basename ${FILE}`" ${TIME} "${STATUS}"
			echo -n "${SYSTEM}_${TEST}_${SPLIT}, ${FRAMES}, ${TIME}" >> ${output_csv}

			# RESCORE steps
			for LM in ${rescore_lms}; do
				FILE=${test_dir}/${SYSTEM}/decode_${LM}_${TEST}/log/rescorelm.${SPLIT}.log
                	        if [ ! -f $FILE ]; then 
                        	        STATUS="TEST NOT DONE"; 
                               		TIME=0
                       		else 
                               		STATUS=" ";
                                	TIME=`cat ${FILE} | grep "elapsed time" | awk '{print $(NF-1)}'` 
                        	fi

                                printf "  %15s: %17s: [%5d] secs %s\n"  "${LM}" "`basename ${FILE}`" ${TIME} "${STATUS}"
				echo -n ", ${TIME}" >> ${output_csv}
			done
			echo "" >> ${output_csv}
		done
	done
done
) >> ${output}

exit

	# DECODE step. tgsmall corresponds to decode
	for TEST in dev_clean test_clean dev_other test_other; do
		cd ${SYSTEM}/decode_tgsmall_${TEST}/log
		echo
		echo ${SYSTEM}/decode_tgsmall_${TEST}
		echo ----------------------------------
		for FILE in decode.*; do
			TIME=`cat ${FILE} | grep "elapsed time" | awk '{print $(NF-1)}'`
			FRAMES=`cat ${FILE} | grep "Overall log-likelihood per frame" | awk '{print $(NF-1)}'`
			echo -n "${FILE}: "
			echo "time: [${TIME}], frames: [${FRAMES}]"
		done
		cd ${root_dir}
	done

	# RESCORE step. tgmed, tglarge and fglarge are the different rescorings
	for LM in tgmed tglarge fglarge; do
		for TEST in dev_clean test_clean dev_other test_other; do
			cd ${SYSTEM}/decode_${LM}_${TEST}/log
			echo
			echo ${SYSTEM}/decode_${LM}_${TEST}
			echo -------------------------------
			for FILE in rescorelm.*; do
				TIME=`cat ${FILE} | grep "elapsed time" | awk '{print $(NF-1)}'`
				echo -n "${FILE}: "
				echo "time: [${TIME}]"
			done
			cd ${root_dir}
		done
	done
