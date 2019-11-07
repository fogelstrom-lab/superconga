// ComputeProperty and its derived classes work in tandem with
// the IntegrationIterator class. Functions in this base class are mostly
// 'hooks' which gets called by IntegrationIterator as it integrates over
// energies and angles.


#ifndef _COMPUTEPROPERTY_H
#define _COMPUTEPROPERTY_H

//#include "util/csecond.c"
#include <Parameters.h>
#include <Grid.h>
#include <GeometryGroup.h>
#include <GreensFunctionStatic.h>
#include <IntegrationIterator.h>
#include <ParameterIterator.h>
#include <ContextModule.h>

template <typename> class IntegrationIterator;

template <typename T>
class ComputeProperty : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
						ComputeProperty();
	virtual				~ComputeProperty();

						ComputeProperty( Context<T>* ctx );


//	Parameters<T> *			getParameters() const { return _param; };
	GridGPU*			getGreensCurrentAngle() { return _greensCurrentAngle; }
	void				initGreensAngle( const int N, const int numFields = 1 );
	
	T					getResidual() const { return _residual; }
	void				setResidual( T res ) { _residual = res; }

	bool				isConverged() const { return _isConverged; }
	void				setConverged( bool converged = true ) { _isConverged = converged; }

	// Static Functions
//	template<template <typename> class POTENTIAL>
	static GridGPU		computeSFMomentum( Parameters<T>* param, OrderParameter<T>* delta );


	// Virtual functions
	//
	virtual IntegrationIterator<T>*	createIntegrationIterator( int numEnergiesPerBlock = -1 ) = 0;
	
	// Should return pointer to the computed field (order parameter, current, etc.)
	virtual GridGPU*	getResult() = 0;

	// Gets called (by IntegrationIterator) once for every iteration, before any integration takes place.
	virtual void		initializeIteration( int numFields = 1 ) = 0;
	
	// Gets called once for each angle, before computing greens function.
	virtual void		initializeGreensAngle() { _greensCurrentAngle->setZero(); }
	
	// Computes greens function for current angle. This may be called multiple times
	// as we integrate over many energies.
	virtual void		computeGreensAngle(
							ParameterIterator<int>* iter,
							GridGPU* gamma,
							GridGPU* gammabar,
							GeometryGroup<T>* geom ) = 0;

	// Gets called once for each angle, after all calls (per angle) to computeGreensAngle()
	// are finished. Adds the result from computeGreensAngle() to total.
	virtual void		integrateGreensAngle( GeometryGroup<T>* geom, const T angle ) = 0;

	// Gets called once per iteration, after all angle integration of greens functions
	// are finished. Computes physical property from greens function.
	virtual void		computePropertyFromGreens() = 0;
	
	// We don't always have to compute for every iteration. For example, current and free energy
	// only needs to be computed once the order parameter has converged.
	void				setCompute( bool doCompute ) { _doCompute = doCompute; }
	bool 				doCompute() const { return _doCompute; }


private:

	bool		_isConverged;
	bool		_doCompute;

	T 			_residual;

	GridGPU*	_greensCurrentAngle;
};

#endif

