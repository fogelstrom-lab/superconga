#ifndef _CONTEXT_H
#define _CONTEXT_H

#include <util/csecond.c>
#include <Parameters.h>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <OrderParameterComponent.h>
#include <RiccatiSolver.h>
#include <IntegrationIterator.h>
#include <CoherenceFunctions.h>
#include <GradientAccelerator.h>
#include <ComputeProperty.h>
#include <ComputeOrderParameter.h>
#include <ComputeCurrent.h>
#include <ComputeCurrentDensity.h>
#include <ComputeLDOS.h>

template<typename> class RiccatiSolver;

// CONSIDER MAKING ENERGY ITERATOR A CONTEXT MODULE


template<typename T>
class Context
{
public:
			Context();
			~Context();

	// Setters
	void					set( Parameters<T>* param ) { _param = param; }
	void					set( GeometryGroup<T>* geom ) { _geom = geom; }
	void					set( OrderParameter<T>* delta ) { _delta = delta; }
	void					set( RiccatiSolver<T>* solver ) { _solver = solver; }
	void					set( IntegrationIterator<T>* iter ) { _iter = iter; }
	void					set( GradientAccelerator<T>* acc ) { _acc = acc; }
	int						add( ComputeOrderParameter<T>* compute ) { _compute_delta = compute; return add(static_cast<ComputeProperty<T>*>(compute)); }
	int						add( ComputeCurrent<T>* compute ) { _compute_current = compute; return add(static_cast<ComputeProperty<T>*>(compute)); }
	int						add( ComputeCurrentDensity<T>* compute ) { _compute_current_density = compute; return add(static_cast<ComputeProperty<T>*>(compute)); }
	int						add( ComputeLDOS<T>* compute ) { _compute_ldos = compute; return add(static_cast<ComputeProperty<T>*>(compute)); }
	int						add( ComputeProperty<T>* compute );

	// Functions
	int							initialize( int numEnergiesPerBlock = -1 );
	void						compute();

	// Getters
	Parameters<T>*				getParameters() const { return _param; }
	GeometryGroup<T>*			getGeometry() const { return _geom; }
	OrderParameter<T>*			getOrderParameter() const { return _delta; }
	OrderParameterComponent<T>*	getOrderParameterComponent( int idx ) const { return _delta->getOrderParameterComponent(idx); }
	RiccatiSolver<T>*			getRiccatiSolver() const { return _solver; }
	IntegrationIterator<T>*		getIntegrationIterator() const { return _iter; }
	GradientAccelerator<T>*		getGradientAccelerator() const { return _acc; }

	int 						getNumComputeObjects() const { return _computeObjectList.size(); }
	ComputeProperty<T>*			getComputeObject( int idx ) const { return _computeObjectList[idx]; }

	ComputeOrderParameter<T>*	getComputeOrderParameter() const { return _compute_delta; }
	ComputeCurrent<T>*			getComputeCurrent() const { return _compute_current; }
	ComputeCurrentDensity<T>*	getComputeCurrentDensity() const { return _compute_current_density; }
	ComputeLDOS<T>*				getComputeLDOS() const { return _compute_ldos; }

	T 							getResidual( int idx = 0 ) const;

	double						getComputeTime() const { return _compute_time; }
	int 						getNumIterations() const { return _numIterations; }

	// Helpers
	int 						isConverged() const;

private:

	Parameters<T>*			_param;
	GeometryGroup<T>*		_geom;
	OrderParameter<T>*		_delta;
	RiccatiSolver<T>*		_solver;
	IntegrationIterator<T>*	_iter;
	GradientAccelerator<T>* _acc;

	std::vector<ComputeProperty<T>*> _computeObjectList;

	double					_compute_time;
	int 					_numIterations;

	// Helper pointers
	ComputeOrderParameter<T>*	_compute_delta;
	ComputeCurrent<T>*			_compute_current;
	ComputeCurrentDensity<T>*	_compute_current_density;
	ComputeLDOS<T>*				_compute_ldos;
};

#endif
