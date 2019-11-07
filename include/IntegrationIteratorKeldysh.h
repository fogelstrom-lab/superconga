#ifndef _INTEGRATIONITERATORKELDYSH_H
#define _INTEGRATIONITERATORKELDYSH_H

//#include <iostream>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <RiccatiSolver.h>
#include <ComputeProperty.h>
#include <ParameterIterator.h>
#include <Context.h>

template <typename T>
class IntegrationIteratorKeldysh : public IntegrationIterator<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
				IntegrationIteratorKeldysh();
	virtual		~IntegrationIteratorKeldysh();

				IntegrationIteratorKeldysh(	Context<T>* ctx,
											int numEnergiesPerBlock = -1 );

	virtual int		initialize();

	// Configures energy iterators
	void			initIterators( int numEnergiesPerBlock );
	virtual void	initEnergies();

	virtual int 	getNumEnergiesInterval();
	virtual int 	getEnergyIdxOffset_boundaryStorage();

	// Call this once before starting self consistent iteration, or when changing initial condition.
	virtual void	setBoundaryInitialCondition();	

	// Computes all properties added with addComputeProperty(). Iterate this self consistently.
	virtual void	compute();		

private:

	int 	_lastEnergyIndex;
};

#endif

