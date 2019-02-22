#!/bin/bash

format=pdf # pdf svg
mode=save # display save
lm_scale=0.0
acoustic_scale=0.0
#end of config

. utils/parse_options.sh

if [ $# -lt 3 ]; then
   echo "usage: $0 [--mode display|save] [--format pdf|svg] <utt-id> <lattice-ark> <word-list> [output-path]"
   echo "e.g.:  $0 utt-0001 \"test/lat.*.gz\" tri1/graph/words.txt"
   exit 1;
fi

. ./path.sh

uttid=$1
lat=$2
words=$3
out_dir="."

if [ "$#" -ge 4 ]; then
 out_dir=$4
fi


tmpdir=$(mktemp -d /tmp/kaldi.XXXX); # trap "rm -r $tmpdir" EXIT # cleanup

filename=$(basename -- "$lat")
extension="${filename##*.}"

if [ $extension == "gz" ]; then
  gunzip -c $lat | lattice-to-fst --lm-scale=$lm_scale --acoustic-scale=$acoustic_scale ark:- "scp,p:echo $uttid $tmpdir/$uttid.fst|" || exit 1;
else
  lattice-to-fst --lm-scale=$lm_scale --acoustic-scale=$acoustic_scale ark:${lat} "scp,p:echo $uttid $tmpdir/$uttid.fst|" || exit 1;
fi

! [ -s $tmpdir/$uttid.fst ] && \
  echo "Failed to extract lattice for utterance $uttid (not present?)" && exit 1;
fstdraw --allow_negative_labels=true --portrait=true --osymbols=$words $tmpdir/$uttid.fst | dot -T${format} > $tmpdir/$uttid.${format}

if [ "$(uname)" == "Darwin" ]; then
    doc_open=open
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    doc_open=xdg-open
elif [ $mode == "display" ] ; then
        echo "Can not automaticaly open file on your operating system"
        mode=save
fi

[ $mode == "display" ] && $doc_open $tmpdir/$uttid.${format}
[[ $mode == "display" && $? -ne 0 ]] && echo "Failed to open ${format} format." && mode=save
[ $mode == "save" ] && echo "Saving to $uttid.${format}" && cp $tmpdir/$uttid.${format} ${out_dir}

exit 0
