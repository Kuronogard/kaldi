
all:
	-rm -f arpa2fst
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = baseline


OBJFILES =

ADDLIBS = ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

TESTFILES =

include ../makefiles/default_rules.mk
