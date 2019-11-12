#ifndef _RICCATISOLVERDISC_H
#define _RICCATISOLVERDISC_H

#include <RiccatiSolver.h>

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
class RiccatiSolverDisc : public RiccatiSolver<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:

	// Constructors and Destructors

					RiccatiSolverDisc();
	virtual			~RiccatiSolverDisc();

					RiccatiSolverDisc( Context<T>* ctx );

	virtual int 	initialize();

	void			initializeBoundaryStorage() {}

	// Functions
	virtual void	computeInitialCondition(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx ) {}

	virtual void	computeBoundaryCondition() {}

	virtual void	computeCoherenceFunctions(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx ) {}

private:

};

#endif
