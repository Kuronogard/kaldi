
all:
	-rm -f arpa2fst
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk


LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


BINFILES = aclattice2lattice decode rnn_lm_rescoring wfst_lm_rescoring extract_probs_B extract_probs_B_accel


OBJFILES = resource_monitor.o probeCPU.o probeGPU.o

LIBNAME = standalone-utils

ADDLIBS = ../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
          ../rnnlm/kaldi-rnnlm.a ../lm/kaldi-lm.a \
					../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a \
					../base/kaldi-base.a

TESTFILES =

cuda-compiled.o: ../kaldi.mk

include ../makefiles/default_rules.mk
