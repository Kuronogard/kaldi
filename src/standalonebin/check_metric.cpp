/*
 * Extract probabilities from 40-mfcc, cmvn and 60-ivectors
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>

#include "cudamatrix/cu-device.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"

using namespace kaldi;
using namespace kaldi::nnet3;

#define ABS_(a) ((a) < 0)? -(a): (a)

#define NEGLOGLIKE_TO_PROB(a) (1 - exp(a))

double probDensityMetrics(Matrix<BaseFloat> ac_probs) {

    for (int i = 0; i < ac_probs.NumRows(); i++) {
        
        double mean = 0;
        double stdDeviation = 0;
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
            mean += prob;
        }
        mean /= ac_probs.NumCols();
        
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
            stdDeviation += pow(prob - mean, 2);
        }
        stdDeviation = stdDeviation / (ac_probs.NumCols() - 1);
        stdDeviation = sqrt(stdDeviation);
    
        std::cout << "(" << mean << ", " << stdDeviation << ")" << std::endl;
    }
    
    return 0.0;
} 



// Metrics
// Computes, for each frame, the distance from the highest value to
// the next, and averages for the utterance (Confidence of the first choice)
double meanDistanceFromMaxToNext(Matrix<BaseFloat> ac_probs) {
    double mean = 0;
    
    for (int i = 0; i < ac_probs.NumRows(); i++) {
        double max = 0;
        int index = -1;
        
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
						prob = NEGLOGLIKE_TO_PROB(prob);
            if (prob > max) {
                max = prob;
                index = j;
            }
        }
       
        double secMax = 0;
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
						prob = NEGLOGLIKE_TO_PROB(prob);
            if ((j != index) && (prob > secMax)) {
                secMax = prob;
            }
        }
        
        mean += max - secMax;
    }
    
    mean /= ac_probs.NumRows();
    
    return mean;
} 

// Computes, for each frame, the number of values above a given distance
// of the highest value, and averages for the utterance 
// (number of "high" confidence choices)
double numValuesInThresholdOfMax(Matrix<BaseFloat> ac_probs, double threshold) {
    double mean = 0;
    
    for (int i = 0; i < ac_probs.NumRows(); i++) {
        double max = 0;
        int index = -1;
        
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
						prob = NEGLOGLIKE_TO_PROB(prob);
            if (prob > max) {
                max = prob;
                index = j;
            }
        }
       
        int num = 0;
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
						prob = NEGLOGLIKE_TO_PROB(prob);
            if ((j != index) && ((max-prob) < threshold)) {
                num++;
            }
        }
        
        mean += num;
    }
    
    
    mean /= ac_probs.NumRows();
    
    return mean;
} 





// Metrics
// Computes, for each frame, the distance from the highest value to
// the next, and averages for the utterance (Confidence of the first choice)
double meanDistanceFromMinToNext(Matrix<BaseFloat> ac_probs) {
    double mean = 0;
    
    for (int i = 0; i < ac_probs.NumRows(); i++) {
        double min = std::numeric_limits<double>::max();
        int index = -1;
        
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
            if (prob < min) {
                min = prob;
                index = j;
            }
        }
       
        double secMin = std::numeric_limits<double>::max();
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
            if ((j != index) && (prob < secMin)) {
                secMin = prob;
            }
        }
        
        double dist = ABS_(min - secMin);
        mean += dist;
    }
    
    mean /= ac_probs.NumRows();
    
    return mean;
} 



// Computes, for each frame, the number of values above a given distance
// of the highest value, and averages for the utterance 
// (number of "high" confidence choices)
double numValuesInThresholdOfMin(Matrix<BaseFloat> ac_probs, double threshold) {
    double mean = 0;
    
    for (int i = 0; i < ac_probs.NumRows(); i++) {
        double min = std::numeric_limits<double>::max();
        int index = -1;
        
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
            if (prob < min) {
                min = prob;
                index = j;
            }
        }

        int num = 0;
        for (int j = 0; j < ac_probs.NumCols(); j++) {
            double prob = ac_probs(i,j);
            double dist = ABS_(min-prob);
            if ((j != index) && (dist < threshold)) {
                num++;
            }
        }
        
        mean += num;
    }
    
    
    mean /= ac_probs.NumRows();
    
    return mean;
} 



int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		
		
		const char *usage =
				"Reads 1 probs matrix and outputs some metrics\n"
				"Usage: compare_probs <acprobs_filename>\n";

		ParseOptions po(usage);
		std::string out_filename = "";
		bool print_header = true;
		bool print_metric_header = true;

		po.Register("out-filename", &out_filename, "File to save acprobs in csv format.");
		po.Register("print-header", &print_header, "Print the csv header in the 'out-filename' file. Does not has effect if --out-filename is not specified.");
		po.Register("print-metric-header", &print_metric_header, "Print to std output the metrics header");

		po.Read(argc, argv);
		if (po.NumArgs() != 1) {
			po.PrintUsage();
			exit(1);
		}

		std::string probs_filename = po.GetArg(1);


    //std::cerr <<"Before read matrix" << std::endl;		
		Matrix<BaseFloat> utt_ac_probs;		
		{	
			std::ifstream acf(probs_filename);
			assert(acf.is_open());
			unsigned int NumRows;  // num frames
			unsigned int NumCols;	// num pdfs
			acf.read((char*)&NumRows, sizeof(unsigned int));	// nuim frames
			acf.read((char*)&NumCols, sizeof(unsigned int)); // num pdfs	
      //std::cerr << "R: " << NumRows << "C: " << NumCols << std::endl;
			utt_ac_probs.Resize(NumRows, NumCols);
			for (MatrixIndexT i = 0; i < NumRows; i++) {
				for (MatrixIndexT j = 0; j < NumCols; j++) {
					BaseFloat weight;
					acf.read((char*)&weight, sizeof(BaseFloat));	
					utt_ac_probs(i, j) = weight;
                    //std::cout << weight << ", ";
				}
                //std::cout << std::endl;
			}
			acf.close();
		}
		//std::cerr << "After read matrix" << std::endl;

		std::ofstream out_file;
		if (out_filename != "") {
			std::cerr << "Write to " << out_filename << std::endl;
			out_file.open(out_filename);
			if (!out_file.is_open()) {
				std::cout << "Could not open output file " << probs_filename << std::endl;
			}
			else {
				if (print_header) {
					// Print Header
					out_file << "frame" << std::endl;
					for (int i = 0; i < utt_ac_probs.NumCols() ; i++) out_file << ", " << i;
					out_file << std::endl;
				}

				for (int i = 0; i < utt_ac_probs.NumRows(); i++) {
					out_file << i;
    			for (int j = 0; j < utt_ac_probs.NumCols(); j++) {
						double prob = utt_ac_probs(i,j);
						prob = NEGLOGLIKE_TO_PROB(prob);
						out_file << ", " << prob;
     			}
					out_file << std::endl;
				}
				out_file.close();
			}
		}
		
        
    //probDensityMetrics(utt_ac_probs);
    
		if (print_metric_header) {
			std::cout << "meanDistanceFromMaxToNext";
			for (int i = 0; i <= 10; i++) std::cout << ", numValuesNearMax_" << 0.1*i;    
			std::cout << std::endl;
		}

		double dist;
    dist = meanDistanceFromMaxToNext(utt_ac_probs);
    std::cout << dist;
    for (int i = 0; i <= 10; i++) {
      double threshold = 0.1 * i;
			double numVals;
      numVals = numValuesInThresholdOfMax(utt_ac_probs, threshold);
      std::cout << ", " << numVals;
    }
    std::cout << std::endl;

        

		return 0;

	} catch(const std::exception &e) {
		std::cerr << "EXCEPTION :" << e.what() << std::endl;
		return -1;
	}
}
