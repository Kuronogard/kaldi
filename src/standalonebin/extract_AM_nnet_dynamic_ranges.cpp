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


int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		
		// Extract dynamic ranges for AM nnet
		const char *usage =
				"Extract dynamic ranges for nnet weights\n"
				"Usage: extract_nnet_weight_dynamic_range <nnet_filename>\n";

		ParseOptions po(usage);

		po.Read(argc, argv);
		if (po.NumArgs() != 2) {
			po.PrintUsage();
			exit(1);
		}

		std::string nnet_filename = po.GetArg(1);


		kaldi::nnet3::Nnet nnet;
		ReadKaldiObject(nnet_filename, &nnet);

		// Iterate through the nnet components
		// iterate thought the component weights accumulating frequency statistics
		for( int i = 0; i < nnet.NumComponents(); i++) {
			Component* comp = nnet.GetComponent(i);
			
		}

		
		Matrix<BaseFloat> ref_mat;		
		{	
			std::ifstream acf(probs_ref_filename);
			assert(acf.is_open());
			unsigned int NumRows;  // num frames
			unsigned int NumCols;	// num pdfs
			acf.read((char*)&NumRows, sizeof(unsigned int));	// num frames
			acf.read((char*)&NumCols, sizeof(unsigned int)); // num pdfs	
			ref_mat.Resize(NumRows, NumCols);
			for (MatrixIndexT i = 0; i < NumRows; i++) {
				for (MatrixIndexT j = 0; j < NumCols; j++) {
					BaseFloat weight;
					acf.read((char*)&weight, sizeof(BaseFloat));	
					ref_mat(i, j) = weight;
				}
			}
			acf.close();
		}
	
		Matrix<BaseFloat> comp_mat;		
		{	
			std::ifstream acf(probs_comp_filename);
			assert(acf.is_open());
			unsigned int NumRows;
			unsigned int NumCols;
			acf.read((char*)&NumRows, sizeof(unsigned int));	// num frames
			acf.read((char*)&NumCols, sizeof(unsigned int)); // num pdfs
			comp_mat.Resize(NumRows, NumCols);
			for (MatrixIndexT i = 0; i < NumRows; i++) {
				for (MatrixIndexT j = 0; j < NumCols; j++) {
					BaseFloat weight;
					acf.read((char*)&weight, sizeof(BaseFloat));	
					comp_mat(i, j) = weight;
				}
			}
			acf.close();
		}	


		if (ref_mat.NumCols() != comp_mat.NumCols()) {
			std::cout << "WARN: Different number of columns [ref: " << ref_mat.NumCols() << " comp: " << comp_mat.NumCols() << std::endl;
			return 0;
		}
	
		if (ref_mat.NumRows() != comp_mat.NumRows()) {
			std::cout << "WARN: Different number of rows [ref: " << ref_mat.NumRows() << " comp: " << comp_mat.NumRows() << std::endl;
			return 0;
		}
	
		double avg_dif = 0;
		int numValues = ref_mat.NumRows() * ref_mat.NumCols();
		for (int i = 0; i < ref_mat.NumRows(); i++) {
			for (int j = 0; j < ref_mat.NumCols(); j++) {
				float dif = ref_mat(i, j) - comp_mat(i, j);
				avg_dif += abs(dif / ref_mat(i,j));
			}
		}

		avg_dif /= numValues;
		avg_dif *= 100;

		std::cout << "Average Diference: " << avg_dif << "%" << std::endl;

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
