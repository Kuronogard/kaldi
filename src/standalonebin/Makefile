
all:
	-rm -f arpa2fst
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk


LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = baseline_A extract_probs extract_probs_B baseline_B extract_nnet3_statistics


OBJFILES =


ADDLIBS = ../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
          ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a \
					../base/kaldi-base.a ../rnnlm/kaldi-rnnlm.a ../lm/kaldi-lm.a 


#ADDLIBS = ../tree/kaldi-tree.a \
		  ../sgmm2/kaldi-sgmm2.a \
		  ../util/kaldi-util.a \
          ../cudamatrix/kaldi-cudamatrix.a \
          ../base/kaldi-base.a \
		  ../lm/kaldi-lm.a \
	      ../lat/kaldi-lat.a \
		  ../nnet/kaldi-nnet.a \
		  ../decoder/kaldi-decoder.a \
		  ../ivector/kaldi-ivector.a \
		  ../matrix/kaldi-matrix.a \
		  ../gmm/kaldi-gmm.a \
		  ../nnet3/kaldi-nnet3.a \
		  ../hmm/kaldi-hmm.a \
		  ../online2/kaldi-online2.a \
		  ../fstext/kaldi-fstext.a \
		  ../rnnlm/kaldi-rnnlm.a \
		  ../chain/kaldi-chain.a \
		  ../kws/kaldi-kws.a \
		  ../feat/kaldi-feat.a \
		  ../nnet2/kaldi-nnet2.a \
		  ../transform/kaldi-transform.a

#ADDLIBS = ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
		  ../lat/kaldi-lat.a ../rnnlm/kaldi-rnnlm.a \
		  ../nnet3/kaldi-nnet3.a ../lm/kaldi-lm.a \
		  ../fstext/kaldi-fstext.a ../cudamatrix/kaldi-cudamatrix.a \
		  ../nnet2/kaldi-nnet2.a ../decoder/kaldi-decoder.a \
		  ../nnet/kaldi-nnet.a  

TESTFILES =

cuda-compiled.o: ../kaldi.mk

include ../makefiles/default_rules.mk
