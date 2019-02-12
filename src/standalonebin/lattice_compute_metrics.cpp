#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/kaldi-lattice.h"

#include "lat/lattice-functions.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"

#define ABS_(a) (((a)>=0)? (a) : -(a))
#define MAXABS_(a, b)  ((ABS_(a) > ABS_(b))? ABS_(a) : ABS_(b))




int main(int argc, char **argv) {

	using namespace kaldi;
	using namespace kaldi::nnet3;
	typedef kaldi::int32 int32;
	typedef kaldi::int64 int64;
	using fst::VectorFst;
	using fst::StdArc;
	using fst::ReadFstKaldi;

        bool print_header = true;
	const char *usage = 
			"Computes different metrics for the lattice\n"
			"usage: wfst-lm-rescoring [options] <lattice-rspecifier>";

	ParseOptions po(usage);
        po.Register("print-header", &print_header, "Prints the header of the output (yes|no) (default:yes).");
	po.Read(argc, argv);

	if (po.NumArgs() != 1) {
		po.PrintUsage();
		exit(1);
	}


	std::string lats_rspecifier = po.GetArg(1);


  // read lattice (Normal lattice)
	SequentialLatticeReader lattice_reader(lats_rspecifier);
        if (print_header) {
            std::cout << "utterance, numStates, meanArcPerState, maxArcsFromState" << std::endl;
        }

	try {
		for(; !lattice_reader.Done(); lattice_reader.Next()) {
			std::string utt = lattice_reader.Key();
			Lattice lat = lattice_reader.Value();
			lattice_reader.FreeCurrent();
		
			
            // Compute several metrics related to 'lat' and print them 
            // thought std::cout
            int numStates = lat.NumStates();
            double mean = 0;
            int max = 0;
            
            for (int i = 0; i < numStates; i++) {
                int numArcs = lat.NumArcs(i);
            
                mean += numArcs;
                if (numArcs > max) max = numArcs;
            }
            
            mean /= numStates;
            
            //std::cout << "utterance, numStates, meanArcs, maxArcs" << std::endl;
            std::cout << utt << ", " << numStates << ", " << mean << ", " << max << std::endl;

		} // End of the lattice for loop

	} catch (std::exception &e) {
		std::cerr << "Exception: " << e.what() << std::endl;
	} catch (...) {
    std::cerr << "Interrupted by non-std exception" << std::endl;
	}

	return 0;
}
