// nnet3/nnet-simple-component.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Xiaohui Zhang
//                2015  Guoguo Chen
//                2015  Daniel Galvez
//                2016  Yiming Wang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iterator>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "nnet3/nnet-simple-component.h"
#include "nnet3/nnet-simple-quant-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {


// These methods should take care of quantization stats
void QuantAffineComponent::Scale(BaseFloat scale) {
	AffineComponent::Scale(scale);
	SetQuantCorrect(false);
}

void QuantAffineComponent::Resize(int32 input_dim, int32 output_dim) {
	AffineComponent::Resize(input_dim, output_dim);
	SetQuantCorrect(false);
}

void QuantAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
	AffineComponent::Add(alpha, other_in);
	SetQuantCorrect(false);
}

QuantAffineComponent::QuantAffineComponent(const QuantAffineComponent &component):
		AffineComponent(component),
		QuantizableSubComponent(component)
{}

QuantAffineComponent::QuantAffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                                 const CuVectorBase<BaseFloat> &bias_params,
                                 BaseFloat learning_rate):
		AffineComponent(linear_params, bias_params, learning_rate)
{
	SetQuantCorrect(false);
}

void QuantAffineComponent::SetParams(const CuVectorBase<BaseFloat> &bias,
                                const CuMatrixBase<BaseFloat> &linear) {
	AffineComponent::SetParams(bias, linear); 
	SetQuantCorrect(false);
}

void QuantAffineComponent::PerturbParams(BaseFloat stddev) {
	AffineComponent::PerturbParams(stddev);
	SetQuantCorrect(false);
}

Component* QuantAffineComponent::Copy() const {
  QuantAffineComponent *ans = new QuantAffineComponent(*this);
  return ans;
}

BaseFloat QuantAffineComponent::DotProduct(const UpdatableComponent &other_in) const {
	SetQuantCorrect(false);
	return AffineComponent::DotProduct(other_in);
}

void QuantAffineComponent::GetStatistics(ComponentStatistics &statistics) {
	AffineComponent::GetStatistics(statistics);
	GetQuantizationStats(statistics);
}

// TODO: Change this
void* QuantAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {

	//std::cerr << "QuantAffineComponent!!!" << std::endl;
	//KALDI_ASSERT(w_quantized && "QuantAffineComponent not quantized");

	CuMatrix<BaseFloat> quant_in;
	quant_in.Resize(in.NumRows(), in.NumCols());
	quant_in.CopyFromMat(in);

	//Divide inputs by in_quant_factor
	QuantizeInputs(quant_in);

	out->SetZero();

	// perform propagate
	out->AddMatMat(1.0, quant_in, kNoTrans, linear_params_, kTrans, 1.0);

	//multiply outputs by in_quant_factor * w_quant_factor
	UnQuantizeInputs(*out);

	// Add bias
	out->AddVecToRows(1.0, bias_params_);

  // No need for asserts as they'll happen within the matrix operations.
  //out->CopyRowsFromVec(bias_params_); // copies bias_params_ to each row
  // of *out.
  //out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);


  return NULL;
}

void QuantAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // read opening tag and learning rate.
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "</QuantAffineComponent>");
}

void QuantAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "</QuantAffineComponent>");
}

void QuantAffineComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
	AffineComponent::UnVectorize(params);
}


void QuantAffineComponent::QuantizeWeights() {
	if (QuantCorrect()) return;

	QuantizeWeights_(linear_params_);
	SetQuantCorrect(true);
}

void QuantNaturalGradientAffineComponent::Read(std::istream &is, bool binary) {
  ReadUpdatableCommon(is, binary);  // Read the opening tag and learning rate
  ExpectToken(is, binary, "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);

  BaseFloat num_samples_history, alpha;
  int32 rank_in, rank_out, update_period;

  ExpectToken(is, binary, "<RankIn>");
  ReadBasicType(is, binary, &rank_in);
  ExpectToken(is, binary, "<RankOut>");
  ReadBasicType(is, binary, &rank_out);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);

  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  if (PeekToken(is, binary) == 'M') {
    // MaxChangePerSample, long ago removed; back compatibility.
    ExpectToken(is, binary, "<MaxChangePerSample>");
    BaseFloat temp;
    ReadBasicType(is, binary, &temp);
  }
  if (PeekToken(is, binary) == 'I') {
    // for back compatibility; we don't write this here any
    // more as it's written and read in Write/ReadUpdatableCommon
    ExpectToken(is, binary, "<IsGradient>");
    ReadBasicType(is, binary, &is_gradient_);
  }
  if (PeekToken(is, binary) == 'U') {
    ExpectToken(is, binary, "<UpdateCount>");
    // back-compatibility branch (these configs were added and then removed).
    double temp;
    ReadBasicType(is, binary, &temp);
    ExpectToken(is, binary, "<ActiveScalingCount>");
    ReadBasicType(is, binary, &temp);
    ExpectToken(is, binary, "<MaxChangeScaleStats>");
    ReadBasicType(is, binary, &temp);
  }
  std::string token;
  ReadToken(is, binary, &token);
  // the following has to handle a couple variants of
  if (token.find("QuantNaturalGradientAffineComponent>") == std::string::npos)
    KALDI_ERR << "Expected <QuantNaturalGradientAffineComponent> or "
              << "</QuantNaturalGradientAffineComponent>, got " << token;
}


void QuantNaturalGradientAffineComponent::Write(std::ostream &os,
                                           bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "<RankIn>");
  WriteBasicType(os, binary, preconditioner_in_.GetRank());
  WriteToken(os, binary, "<RankOut>");
  WriteBasicType(os, binary, preconditioner_out_.GetRank());
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, preconditioner_in_.GetUpdatePeriod());
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, preconditioner_in_.GetNumSamplesHistory());
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, preconditioner_in_.GetAlpha());
  WriteToken(os, binary, "</QuantNaturalGradientAffineComponent>");
}


void QuantNaturalGradientAffineComponent::QuantizeWeights() {
	if (QuantCorrect()) return;

	QuantizeWeights_(linear_params_);
	SetQuantCorrect(true);
}

// TODO: Change this
void* QuantNaturalGradientAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {
	// Update input statistics

	CuMatrix<BaseFloat> quant_in;
	quant_in.Resize(in.NumRows(), in.NumCols());
	quant_in.CopyFromMat(in);

	//Divide inputs by in_quant_factor
	QuantizeInputs(quant_in);

	out->SetZero();

	// perform propagate
	out->AddMatMat(1.0, quant_in, kNoTrans, linear_params_, kTrans, 1.0);

	//multiply outputs by in_quant_factor * w_quant_factor
	UnQuantizeInputs(*out);

	// Add bias
	out->AddVecToRows(1.0, bias_params_);

  return NULL;
}


Component* QuantNaturalGradientAffineComponent::Copy() const {
  return new QuantNaturalGradientAffineComponent(*this);
}


void QuantNaturalGradientAffineComponent::Scale(BaseFloat scale) {
	NaturalGradientAffineComponent::Scale(scale);
	SetQuantCorrect(false);
}

void QuantNaturalGradientAffineComponent::Add(BaseFloat alpha, const Component &other_in) {
	NaturalGradientAffineComponent::Add(alpha, other_in);
	SetQuantCorrect(false);
}

QuantNaturalGradientAffineComponent::QuantNaturalGradientAffineComponent(
    const QuantNaturalGradientAffineComponent &other) :
    NaturalGradientAffineComponent(other),
		QuantizableSubComponent(other)
	{}

QuantNaturalGradientAffineComponent::QuantNaturalGradientAffineComponent(
    const CuMatrixBase<BaseFloat> &linear_params,
    const CuVectorBase<BaseFloat> &bias_params) :
		NaturalGradientAffineComponent(linear_params, bias_params) {
	SetQuantCorrect(false);
}

void QuantNaturalGradientAffineComponent::GetStatistics(ComponentStatistics &statistics) {
	NaturalGradientAffineComponent::GetStatistics(statistics);
	GetQuantizationStats(statistics);
}


void QuantLinearComponent::QuantizeWeights() {
	if (QuantCorrect()) return;

	QuantizeWeights_(params_);
	SetQuantCorrect(true);
}

void QuantLinearComponent::Read(std::istream &is, bool binary) {
  std::string token = ReadUpdatableCommon(is, binary);
  KALDI_ASSERT(token == "");
  ExpectToken(is, binary, "<Params>");
  params_.Read(is, binary);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OrthonormalConstraint>");
    ReadBasicType(is, binary, &orthonormal_constraint_);
  } else {
    orthonormal_constraint_ = 0.0;
  }
  ExpectToken(is, binary, "<UseNaturalGradient>");
  ReadBasicType(is, binary, &use_natural_gradient_);

  // Read various natural-gradient-related configs.
  int32 rank_in,  rank_out, update_period;
  BaseFloat alpha, num_samples_history;
  ExpectToken(is, binary, "<RankInOut>");
  ReadBasicType(is, binary, &rank_in);
  ReadBasicType(is, binary, &rank_out);
  ExpectToken(is, binary, "<Alpha>");
  ReadBasicType(is, binary, &alpha);
  ExpectToken(is, binary, "<NumSamplesHistory>");
  ReadBasicType(is, binary, &num_samples_history);
  ExpectToken(is, binary, "<UpdatePeriod>");
  ReadBasicType(is, binary, &update_period);

  preconditioner_in_.SetAlpha(alpha);
  preconditioner_out_.SetAlpha(alpha);
  preconditioner_in_.SetRank(rank_in);
  preconditioner_out_.SetRank(rank_out);
  preconditioner_in_.SetNumSamplesHistory(num_samples_history);
  preconditioner_out_.SetNumSamplesHistory(num_samples_history);
  preconditioner_in_.SetUpdatePeriod(update_period);
  preconditioner_out_.SetUpdatePeriod(update_period);

  ExpectToken(is, binary, "</QuantLinearComponent>");
}

void QuantLinearComponent::Write(std::ostream &os,
                            bool binary) const {
  WriteUpdatableCommon(os, binary);  // Write the opening tag and learning rate
  WriteToken(os, binary, "<Params>");
  params_.Write(os, binary);
  if (orthonormal_constraint_ != 0.0) {
    WriteToken(os, binary, "<OrthonormalConstraint>");
    WriteBasicType(os, binary, orthonormal_constraint_);
  }
  WriteToken(os, binary, "<UseNaturalGradient>");
  WriteBasicType(os, binary, use_natural_gradient_);

  int32 rank_in = preconditioner_in_.GetRank(),
      rank_out = preconditioner_out_.GetRank(),
      update_period = preconditioner_in_.GetUpdatePeriod();
  BaseFloat alpha = preconditioner_in_.GetAlpha(),
      num_samples_history = preconditioner_in_.GetNumSamplesHistory();
  WriteToken(os, binary, "<RankInOut>");
  WriteBasicType(os, binary, rank_in);
  WriteBasicType(os, binary, rank_out);
  WriteToken(os, binary, "<Alpha>");
  WriteBasicType(os, binary, alpha);
  WriteToken(os, binary, "<NumSamplesHistory>");
  WriteBasicType(os, binary, num_samples_history);
  WriteToken(os, binary, "<UpdatePeriod>");
  WriteBasicType(os, binary, update_period);
  WriteToken(os, binary, "</QuantLinearComponent>");
}

void* QuantLinearComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                 const CuMatrixBase<BaseFloat> &in,
                                 CuMatrixBase<BaseFloat> *out) const {

	CuMatrix<BaseFloat> quant_in;
	quant_in.Resize(in.NumRows(), in.NumCols());
	quant_in.CopyFromMat(in);

	//Divide inputs by in_quant_factor
	QuantizeInputs(quant_in);

	out->SetZero();

	// perform propagate
	out->AddMatMat(1.0, quant_in, kNoTrans, params_, kTrans, 1.0);

	//multiply outputs by in_quant_factor * w_quant_factor
	UnQuantizeInputs(*out);

  return NULL;
}


Component* QuantLinearComponent::Copy() const {
  return new QuantLinearComponent(*this);
}

QuantLinearComponent::QuantLinearComponent(
    const QuantLinearComponent &other):
		LinearComponent(other),
		QuantizableSubComponent(other)
	{}

QuantLinearComponent::QuantLinearComponent(const CuMatrix<BaseFloat> &params):
		LinearComponent(params) 
	{} 

void QuantLinearComponent::Scale(BaseFloat scale) {
	LinearComponent::Scale(scale);
	SetQuantCorrect(false); 
}

void QuantLinearComponent::Add(BaseFloat alpha, const Component &other_in) {
	LinearComponent::Add(alpha, other_in);
	SetQuantCorrect(false); 
}

void QuantLinearComponent::PerturbParams(BaseFloat stddev) {
	LinearComponent::PerturbParams(stddev);
	SetQuantCorrect(false); 
}


void QuantLinearComponent::UnVectorize(const VectorBase<BaseFloat> &params) {
	LinearComponent::UnVectorize(params);
	SetQuantCorrect(false);
}

BaseFloat QuantLinearComponent::DotProduct(const UpdatableComponent &other_in) const {
	SetQuantCorrect(false); 
	return LinearComponent::DotProduct(other_in);
}

void QuantLinearComponent::GetStatistics(ComponentStatistics &statistics) {
	LinearComponent::GetStatistics(statistics);
	GetQuantizationStats(statistics);
}

QuantFixedAffineComponent::QuantFixedAffineComponent(const AffineComponent &c):
   FixedAffineComponent(c)
	{}


void QuantFixedAffineComponent::QuantizeWeights() {
	if (QuantCorrect()) return;

	QuantizeWeights_(linear_params_);
	SetQuantCorrect(true);
}


void* QuantFixedAffineComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                     const CuMatrixBase<BaseFloat> &in,
                                     CuMatrixBase<BaseFloat> *out) const  {

	// Update input statistics
	CuMatrix<BaseFloat> quant_in;
	quant_in.Resize(in.NumRows(), in.NumCols());
	quant_in.CopyFromMat(in);

	//Divide inputs by in_quant_factor
	QuantizeInputs(quant_in);

	out->SetZero();

	// perform propagate
	out->AddMatMat(1.0, quant_in, kNoTrans, linear_params_, kTrans, 1.0);

	//multiply outputs by in_quant_factor * w_quant_factor
	UnQuantizeInputs(*out);

	// Add bias
	out->AddVecToRows(1.0, bias_params_);

  return NULL;

  //out->CopyRowsFromVec(bias_params_); // Adds the bias term first.
  //out->AddMatMat(1.0, in, kNoTrans, linear_params_, kTrans, 1.0);
 // return NULL;
}

//QuantFixedAffineComponent::QuantFixedAffineComponent(
//		const QuantizableSubComponent &other) :
//		QuantFixedAffineComponent(),
//		QuantizableSubComponent(other) {
//	linear_params_ = other.linear_params_;
//	bias_params_ = bias_params_;
//}

Component* QuantFixedAffineComponent::Copy() const {
	QuantFixedAffineComponent *ans = new QuantFixedAffineComponent();
	//ans((QuantizableSubComponent)*this);
	ans->linear_params_ = linear_params_;
	ans->bias_params_ = bias_params_;
	return ans;
}

void QuantFixedAffineComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<QuantFixedAffineComponent>");
  WriteToken(os, binary, "<LinearParams>");
  linear_params_.Write(os, binary);
  WriteToken(os, binary, "<BiasParams>");
  bias_params_.Write(os, binary);
  WriteToken(os, binary, "</QuantFixedAffineComponent>");
}

void QuantFixedAffineComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<QuantFixedAffineComponent>", "<LinearParams>");
  linear_params_.Read(is, binary);
  ExpectToken(is, binary, "<BiasParams>");
  bias_params_.Read(is, binary);
  ExpectToken(is, binary, "</QuantFixedAffineComponent>");
}


} // namespace nnet3
} // namespace kaldi
