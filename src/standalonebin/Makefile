
all:
	-rm -f arpa2fst

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk


LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

#BINFILES = resourceMonitorARMtest aclattice2lattice decode rnn_lm_rescoring wfst_lm_rescoring extract_probs_B extract_probs_B_accel
#BINFILES = rapl-read resourceMonitorTest aclattice2lattice decode rnn_lm_rescoring wfst_lm_rescoring extract_probs_B extract_probs_B_accel

BINFILES = lattice_compute_metrics check_metric compare_acprobs rapl-read aclattice2lattice decode rnn_lm_rescoring \
           rnn_lm_rescoring_pruned rnn_lm_rescoring_pruned_accel wfst_lm_rescoring extract_probs_B extract_probs_B_accel \
					 resourceMonitorTest
           
           
           
#if(PLATFORM STREQUAL "PLATFORM_INTEL")
#	OBJFILES = resource_monitor_intel.o probeCPU.o probeGPU.o
#elseif(PLATFORM STREQUAL "PLATFORM_ARM")
#	OBJFILES = resource_monitor_ARM.o probeARM.o
#endif()

ifeq ($(PLATFORM), PLATFORM_INTEL)
OBJFILES = resource_monitor_intel.o probeCPU.o probeGPU.o
else
ifeq ($(PLATFORM), PLATFORM_ARM)
#$(warning The following command will probably fail, in that case add -fPIC to your CXXFLAGS and remake all.)
OBJFILES = resource_monitor_ARM.o probeARM.o
else
$(error No platform defined)
endif
endif

OBJFILES += resource_monitor_threaded.o


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
