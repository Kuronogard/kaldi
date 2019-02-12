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


#define ABS_(a) ((a) < 0)? -(a): (a)

int main(int argc, char **argv) {

	try {
		using namespace kaldi;
		using namespace kaldi::nnet3;
		
		
		const char *usage =
				"Reads 2 probs matrices and compares them, outputing the mean error\n"
				"Usage: compare_probs <probs_reference_filename> <probs_compare_filename>\n";

		ParseOptions po(usage);

		po.Read(argc, argv);
		if (po.NumArgs() != 2) {
			po.PrintUsage();
			exit(1);
		}

		std::string probs_ref_filename = po.GetArg(1),
								probs_comp_filename = po.GetArg(2);


		
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
                    std::cout << weight;
				}
                std::cout << std::endl;
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
	
		double max_dif = 0;
		double avg_dif = 0;
		double numValues = ref_mat.NumRows();
/*
		for (int i = 0; i < ref_mat.NumRows(); i++) {
			for (int j = 0; j < ref_mat.NumCols(); j++) {
				double ref = ref_mat(i,j);
				double comp = comp_mat(i,j);
				double dif = ABS_((ref - comp)/ref);
				if (dif > max_dif) max_dif = dif;
				std::cout << "ref: " << ref << " comp: " << comp << std::endl;
				std::cout << "Dif: " << dif * 100. << "%" <<  std::endl;
				avg_dif += dif/numValues;
				std::cout << "avg_dif: " << avg_dif << std::endl;
			}
		}
*/
	// TODO: Change distance metric.
	// For each row, compute substraction and frobenius norm
		for (int i = 0; i < ref_mat.NumRows(); i++) {
				SubVector<BaseFloat> ref_row = ref_mat.Row(i);
				SubVector<BaseFloat> cmp_row = comp_mat.Row(i);
        double mod_ref = ref_row.Norm(2);
        double mod_cmp = cmp_row.Norm(2);
/* 
       for( int j = 0; j < ref_row.Dim() ; j++) {
					std::cout << ref_row(j) << " ";
				}
        std::cout << std::endl;

       for( int j = 0; j < cmp_row.Dim() ; j++) {
					std::cout << cmp_row(j) << " ";
				}
        std::cout << std::endl;
*/

				//cmp_row.Scale(-1);
				ref_row.AddVec(-1.0, cmp_row);
        
				double dif = ref_row.Norm(2); // Compute euclidean distance (2-norm)

				if (dif > max_dif) max_dif = dif;
//				std::cout << "ref: " << ref << " comp: " << comp << std::endl;
				std::cout << "( " << mod_ref << ", " << mod_cmp << ") Dif: " << dif <<  std::endl;
				avg_dif += dif/numValues;
				std::cout << "avg_dif: " << avg_dif << std::endl;
		}



		std::cout << "Num values " << numValues << std::endl;
		//avg_dif /= (double)numValues;

		std::cout << "Average Difference: " << avg_dif << std::endl;
		std::cout << "Maximun Difference: " << max_dif << std::endl;

		return 0;

	} catch(const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}
