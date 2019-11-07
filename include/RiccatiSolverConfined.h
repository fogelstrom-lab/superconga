#ifndef _RICCATISOLVERCONFINED_H
#define _RICCATISOLVERCONFINED_H

//#include "util/csecond.c"
//#include <Parameters.h>
//#include <Grid.h>
//#include "BoundaryStorage.h"
#include <RiccatiSolver.h>

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
class RiccatiSolverConfined : public RiccatiSolver<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:

	// Constructors and Destructors

					RiccatiSolverConfined();
	virtual			~RiccatiSolverConfined();

					RiccatiSolverConfined( Context<T>* ctx );

	virtual int 	initialize();

	// Functions
	virtual void	computeInitialCondition(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx );

	virtual void	computeCoherenceFunctions(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx );

	void			gammaSymmetryTransform();

private:

};

#endif

