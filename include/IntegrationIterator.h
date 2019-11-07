#ifndef _INTEGRATIONITERATOR_H
#define _INTEGRATIONITERATOR_H

//#include <iostream>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <RiccatiSolver.h>
#include <ComputeProperty.h>
#include <ParameterIterator.h>

template <typename T>
class IntegrationIterator : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
				IntegrationIterator();
	virtual		~IntegrationIterator();

				IntegrationIterator(	Context<T>* ctx,
										int numEnergiesPerBlock = -1 );
	
	// Virtual functions
	//
	virtual int 	initialize();

	// Call this once before starting self consistent iteration, or when changing initial condition.
	virtual void	setBoundaryInitialCondition() = 0;

	// Computes all properties added with addComputeProperty(). Iterate this self consistently.
	virtual void	compute() = 0;

	// Used by BoundaryStorage to determine size of storage
	virtual int 	getNumEnergiesInterval() = 0;
	virtual int 	getEnergyIdxOffset_boundaryStorage() = 0;

	// Energies
	const ParameterIterator<int>*	getEnergyIterator() const { return _energy_iter; }
	ParameterIterator<int>* 		getEnergyIterator() { return _energy_iter; }
	void							setEnergyIterator( ParameterIterator<int>* iter ) { if (_energy_iter) delete _energy_iter; _energy_iter = iter; }

	void			initIterators();
	virtual void	initEnergies() = 0;

	// Angles
	int				getCurrentAngleIndex() const { return _currentAngleIndex; }
	T				getCurrentAngleRadians() const;
	
	void			setCurrentAngleIndex( int angle_idx ) { _currentAngleIndex = angle_idx; }


	// Getters
	GridGPU*		getEnergies() const { return _energies; }
	GridGPU*		getResidues() const { return _residues; }

	GridGPU*		getOrderParameterFieldCurrentAngle() const
						{ return ContextModule<T>::getContext()->getOrderParameter()->getOrderParameterFieldRotated(); }					

	int 			getNumIterations() const { return _numIter; }
	int 			getNumEnergiesPerBlock() const { return _numEnergiesPerBlock; }
	int 			getBoundaryIdx_offset() const { return _boundaryIdxOffset; }

	// Setters
	void			setEnergies( GridGPU* en ) { if (_energies) delete _energies; _energies = en; }
	void			setNumEnergiesPerBlock( int num );
	void			setResidues( GridGPU* res ) { if (_residues) delete _residues; _residues = res; }

	// Compute
	int 			increaseIterationCount() { _numIter++; return _numIter; }
//	void			setIntegrationDirection( int dir ) { _direction = dir; }


	// Debug
	void			setDebug( int angle_idx ) { _debug_store_angle = angle_idx; }
	GridGPU *		getDebugGrid() const { return _debug_store_grid; }
	//void			computeDebug();


private:
	// Debug
	int				_debug_store_angle;
	GridGPU *		_debug_store_grid;

	// Energy iterators

	// This iterator loops over all matsubara frequencies when computing gamma
	// functions in parallel, which are summed up to form the greens functions.
	// Used when the storage for gammas needs to be reduced.
	// Has no effect on boundary storage.
//	ParameterIterator<int>	_energyMatsubara_iter;
	
	// This iterator divides the full energy spectrum (array) into several partial
	// blocks. Each partial block is computed until convergence is reached.
	// Used for large sweeps for LDOS, when there is not enough memory to hold
	// all energies in memory at once.
	// Affects boundary storage and gamma storage. Can not be used with matsubara
	// summation.
//	ParameterIterator<int>	_energyBlock_iter;

	GridGPU*				_energies;
	ParameterIterator<int>*	_energy_iter;

	// Factor to multiply summand with in energy sum, i.e. different factor for
	// different energies. In the case of Ozaki integration iterator, these
	// factors are the Ozaki residues. For Matsubara and Keldysh, it will be a
	// factor of 1 for each energy.
	GridGPU*				_residues;

	// Max energies a compute block can have
	int 					_numEnergiesPerBlock;

	int						_boundaryIdxOffset;
	
	// Angle
	int						_numAngles;
	int						_currentAngleIndex;
	
	int						_numIter;
};

#endif

