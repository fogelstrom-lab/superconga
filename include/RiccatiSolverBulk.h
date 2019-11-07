#ifndef _RICCATISOLVERBULK_H
#define _RICCATISOLVERBULK_H

#include <RiccatiSolver.h>

template <typename T, template <typename> class GAMMA, template <typename> class POTENTIAL>
class RiccatiSolverBulk : public RiccatiSolver<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:

	// Constructors and Destructors

					RiccatiSolverBulk();
	virtual			~RiccatiSolverBulk();

					//RiccatiSolverBulk( Parameters<T> & param, GeometryGroup<T> * geom );
					RiccatiSolverBulk( Context<T>* ctx );

	virtual int 	initialize();

	void			initializeBoundaryStorage() {}

	// Functions
	virtual void	computeInitialCondition(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx ) {}

	virtual void	computeBoundaryCondition() {}

	virtual void	computeCoherenceFunctions(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx );

private:

};

#endif
