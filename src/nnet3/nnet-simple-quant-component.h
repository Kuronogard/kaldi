// nnet3/nnet-simple-component.h

// Copyright 2011-2013  Karel Vesely
//           2012-2015  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2015  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen
//                2015  Daniel Galvez
//                2015  Tom Ko

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

#ifndef KALDI_NNET3_NNET_SIMPLE_QUANT_COMPONENT_H_
#define KALDI_NNET3_NNET_SIMPLE_QUANT_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include <iostream>
#include <math.h>
#include <float.h>

namespace kaldi {
namespace nnet3 {

/// @file  nnet-simple-component.h
///   This file contains declarations of components that are "simple", meaning
///   they don't care about the indexes they are operating on, produce one
///   output for one input, and return the kSimpleComponent flag in their
///   Properties(): for example, tanh and affine components.  In
///   nnet-general-component.h there are components that don't fit this pattern.
///
///   Some components that do provide the kSimpleComponent flag are not declared
///   here: see also nnet-normalize-component.h.


// This "nnet3" version of the p-norm component only supports the 2-norm.
//class FixedAffineComponent;
//class FixedScaleComponent;
//class PerElementScaleComponent;
//class PerElementOffsetComponent;

// Affine means a linear function plus an offset.
// Note: although this class can be instantiated, it also
// functions as a base-class for more specialized versions of
// AffineComponent.
class QuantAffineComponent: public AffineComponent, public QuantizableSubComponent {
 public:
  QuantAffineComponent() { } // use Init to really initialize.
  virtual std::string Type() const { return "QuantAffineComponent"; }
	virtual void GetStatistics(ComponentStatistics &statistics);
  virtual int32 Properties() const {
		return AffineComponent::Properties()|kQuantizableComponent;
  }


  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

  virtual Component* Copy() const;


  // Some functions from base-class UpdatableComponent.
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);

  // Some functions that are specific to this class.
  virtual void SetParams(const CuVectorBase<BaseFloat> &bias,
                         const CuMatrixBase<BaseFloat> &linear);
  explicit QuantAffineComponent(const QuantAffineComponent &other);
  // The next constructor is used in converting from nnet1.
  QuantAffineComponent(const CuMatrixBase<BaseFloat> &linear_params,
                  const CuVectorBase<BaseFloat> &bias_params,
                  BaseFloat learning_rate);
  // This function resizes the dimensions of the component, setting the
  // parameters to zero, while leaving any other configuration values the same.
  virtual void Resize(int32 input_dim, int32 output_dim);
	virtual void QuantizeWeights();

 protected:
  friend class QuantNaturalGradientAffineComponent;
  // This function Update() is for extensibility; child classes may override
  // this, e.g. for natural gradient update.
  virtual void Update(
      const std::string &debug_info,
      const CuMatrixBase<BaseFloat> &in_value,
      const CuMatrixBase<BaseFloat> &out_deriv) {
    UpdateSimple(in_value, out_deriv);
  }
  // UpdateSimple is used when *this is a gradient.  Child classes may override
  // this if needed, but typically won't need to.

  const QuantAffineComponent &operator = (const QuantAffineComponent &other); // Disallow.
};


/*
  Keywords: natural gradient descent, NG-SGD, naturalgradient.  For
  the top-level of the natural gradient code look here, and also in
  nnet-precondition-online.h.
  NaturalGradientAffineComponent is
  a version of AffineComponent that has a non-(multiple of unit) learning-rate
  matrix.  See nnet-precondition-online.h for a description of the technique.
  It is described, under the name Online NG-SGD, in the paper "Parallel
  training of DNNs with Natural Gradient and Parameter Averaging" (ICLR
  workshop, 2015) by Daniel Povey, Xiaohui Zhang and Sanjeev Khudanpur.

  Configuration values accepted by this component:

  Values inherited from UpdatableComponent (see its declaration in
  nnet-component-itf for details):
     learning-rate
     learning-rate-factor
     max-change

  Values used in initializing the component's parameters:
     input-dim             e.g. input-dim=1024.  The input dimension.
     output-dim            e.g. output-dim=1024.  The output dimension.
     param-stddev          e.g. param-stddev=0.025.  The standard deviation
                           used to randomly initialize the linear parameters
                           (as Gaussian random values * param-stddev).
                           Defaults to 1/sqrt(input-dim), which is Glorot
                           initialization.
     bias-stddev           e.g. bias-stddev=0.0.  The standard deviation
                           used to randomly initialize the bias parameters.
                           Defaults to 1.0 but we usually set it to 0.0
                           in the config.
     bias-mean             e.g. bias-mean=1.0.  Allows you to ininialize the
                           bias parameters with an offset.  Default is 0.0
                           which is normally suitable

     matrix                e.g. matrix=foo/bar/init.mat  May be used as an
                           alternative to (input-dim, output-dim, param-stddev,
                           bias-stddev, bias-mean) to initialize the parameters.
                           Dimension is output-dim by (input-dim + 1), last
                           column is interpreted as the bias.

   Other options:
    orthonormal-constraint=0.0   If you set this to 1.0, then
                           the linear_params_ matrix will be (approximately)
                           constrained during training to have orthonormal rows
                           (or columns, whichever is fewer).. it turns out the
                           real name for this is a "semi-orthogonal" matrix.
                           You can choose a positive nonzero value different
                           than 1.0 to have a scaled semi-orthgonal matrix,
                           i.e. with singular values at the selected value
                           (e.g. 0.5, or 2.0).  This is not enforced inside the
                           component itself; you have to call
                           ConstrainOrthonormal() from the training code to do
                           this.  All this component does is return the
                           OrthonormalConstraint() value.  If you set this to a
                           negative value, it's like saying "for any value",
                           i.e. it will constrain the parameter matrix to be
                           closer to "any alpha" times a semi-orthogonal matrix,
                           without changing its overall norm.


   Options to the natural gradient (you won't normally have to set these,
   the defaults are suitable):

      num-samples-history   Number of frames used as the time-constant to
                            determine how 'up-to-date' the Fisher-matrix
                            estimates are.  Smaller -> more up-to-date, but more
                            noisy.  default=2000.
      alpha                 Constant that determines how much we smooth the
                            Fisher-matrix estimates with the unit matrix.
                            Larger means more smoothing. default=4.0
      rank-in               Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the input space.  default=20.
      rank-out              Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the output-derivative space.  default=80.
      update-period         Determines after with what frequency (in
                            minibatches) we update the Fisher-matrix estimates;
                            making this > 1 saves a little time in training.
                            default=4.
*/
class QuantNaturalGradientAffineComponent: public NaturalGradientAffineComponent, public QuantizableSubComponent {
 public:
  virtual std::string Type() const { return "QuantNaturalGradientAffineComponent"; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual void GetStatistics(ComponentStatistics &statistics);
	
	virtual int32 Properties() const {
		return NaturalGradientAffineComponent::Properties()|kQuantizableComponent;
  }


	virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;


// this constructor does not really initialize, use InitFromConfig() or Read().
  QuantNaturalGradientAffineComponent() { }
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  // copy constructor
  explicit QuantNaturalGradientAffineComponent(
      const QuantNaturalGradientAffineComponent &other);
  QuantNaturalGradientAffineComponent(
      const CuMatrixBase<BaseFloat> &linear_params,
      const CuVectorBase<BaseFloat> &bias_params);

	virtual void QuantizeWeights();
 private:
  // disallow assignment operator.
  QuantNaturalGradientAffineComponent &operator= (
      const QuantNaturalGradientAffineComponent&);
};



/*
  LinearComponent represents a linear (matrix) transformation of its input, with
  a matrix as its trainable parameters.  It's the same as
  NaturalGradientAffineComponent, but without the bias term.

  Configuration values accepted by this component:

  Values inherited from UpdatableComponent (see its declaration in
  nnet-component-itf for details):
     learning-rate
     learning-rate-factor
     max-change

  Values used in initializing the component's parameters:
     input-dim             e.g. input-dim=1024.  The input dimension.
     output-dim            e.g. output-dim=1024.  The output dimension.
     param-stddev          e.g. param-stddev=0.025.  The standard deviation
                           used to randomly initialize the linear parameters
                           (as Gaussian random values * param-stddev).
                           Defaults to 1/sqrt(input-dim), which is Glorot
                           initialization.
     matrix                e.g. matrix=foo/bar/init.mat  May be used as an
                           alternative to (input-dim, output-dim, param-stddev,
                           bias-stddev, bias-mean) to initialize the parameters.
                           Dimension is output-dim by (input-dim + 1), last
                           column is interpreted as the bias.
    orthonormal-constraint=0.0   If you set this to 1.0, then
                           the linear_params_ matrix will be (approximately)
                           constrained during training to have orthonormal rows
                           (or columns, whichever is fewer).. it turns out the
                           real name for this is a "semi-orthogonal" matrix.
                           You can choose a positive nonzero value different
                           than 1.0 to have a scaled semi-orthgonal matrix,
                           i.e. with singular values at the selected value
                           (e.g. 0.5, or 2.0).  This is not enforced inside the
                           component itself; you have to call
                           ConstrainOrthonormal() from the training code to do
                           this.  All this component does is return the
                           OrthonormalConstraint() value.  If you set this to a
                           negative value, it's like saying "for any value",
                           i.e. it will constrain the parameter matrix to be
                           closer to "any alpha" times a semi-orthogonal matrix,
                           without changing its overall norm.

   Options to the natural gradient (you won't normally have to set these,
   the defaults are suitable):

      use-natural-gradient=true   Set this to false to disable the natural-gradient
                            update entirely (it will do regular SGD).
      num-samples-history   Number of frames used as the time-constant to
                            determine how 'up-to-date' the Fisher-matrix
                            estimates are.  Smaller -> more up-to-date, but more
                            noisy.  default=2000.
      alpha                 Constant that determines how much we smooth the
                            Fisher-matrix estimates with the unit matrix.
                            Larger means more smoothing. default=4.0
      rank-in               Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the input space.  default=20.
      rank-out              Rank used in low-rank-plus-unit estimate of Fisher
                            matrix in the output-derivative space.  default=80.
      update-period         Determines after with what frequency (in
                            minibatches) we update the Fisher-matrix estimates;
                            making this > 1 saves a little time in training.
                            default=4.
*/
class QuantLinearComponent: public LinearComponent, public QuantizableSubComponent {
 public:
  virtual std::string Type() const { return "QuantLinearComponent"; }
  virtual int32 Properties() const {
    return LinearComponent::Properties()|kQuantizableComponent;
  }

	virtual void Read(std::istream &is, bool binary);
	virtual void Write(std::ostream &os, bool binary) const;
	virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;
  virtual void GetStatistics(ComponentStatistics &statistics);


  // this constructor does not really initialize, use InitFromConfig() or Read().
  QuantLinearComponent() { }
  virtual Component* Copy() const;
  virtual void Scale(BaseFloat scale);
  virtual void Add(BaseFloat alpha, const Component &other);
  virtual void PerturbParams(BaseFloat stddev);
  virtual BaseFloat DotProduct(const UpdatableComponent &other) const;
  virtual void UnVectorize(const VectorBase<BaseFloat> &params);
  // copy constructor
  explicit QuantLinearComponent(const QuantLinearComponent &other);

  explicit QuantLinearComponent(const CuMatrix<BaseFloat> &params);
	virtual void QuantizeWeights();

 private:

  // disallow assignment operator.
  QuantLinearComponent &operator= (
      const QuantLinearComponent&);
};


/// FixedAffineComponent is an affine transform that is supplied
/// at network initialization time and is not trainable.
class QuantFixedAffineComponent: public FixedAffineComponent, public QuantizableSubComponent {
 public:
  QuantFixedAffineComponent() { }
  virtual std::string Type() const { return "QuantFixedAffineComponent"; }

  // Copy constructor from AffineComponent-- can be used when we're done
  // training a particular part of the model and want to efficiently disable
  // further training.
  QuantFixedAffineComponent(const AffineComponent &c);
//	QuantFixedAffineComponent(const QuantizableSubComponent &c);

  virtual int32 Properties() const { return FixedAffineComponent::Properties()|kQuantizableComponent; }
	virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                         const CuMatrixBase<BaseFloat> &in,
                         CuMatrixBase<BaseFloat> *out) const;

	virtual void GetStatistics(ComponentStatistics& statistics) {
		FixedAffineComponent::GetStatistics(statistics);
		statistics.info = "QuantFixedAffineComponent, rows, cols, weights, size";
		GetQuantizationStats(statistics);
	}

  virtual Component* Copy() const;
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;

	virtual void QuantizeWeights();
};


} // namespace nnet3
} // namespace kaldi


#endif
