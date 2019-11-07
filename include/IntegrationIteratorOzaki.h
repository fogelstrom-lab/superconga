#ifndef _INTEGRATIONITERATOROZAKI_H
#define _INTEGRATIONITERATOROZAKI_H

//#include <iostream>
#include <GeometryGroup.h>
#include <OrderParameter.h>
#include <RiccatiSolver.h>
#include <ComputeProperty.h>
#include <ParameterIterator.h>
#include <Context.h>

/* Wrapper of dsygv from FORTRAN LAPACK
 * Documentation: http://www.netlib.org/lapack/explore-html/d5/d2e/dsygv_8f.html
 * Requires compilation flags:
 *     GCC:       -lgfortran -llapack -lblas
 *     Intel/MKL: -lgfortran -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -DMKL_ILP64 -m64
 */
extern "C" {
	void dsygv_(int64_t* ITYPE,
				char* JOBZ,
				char* UPLO,
				int64_t* N,
				double* A,
				int64_t* LDA,
				double* B,
				int64_t* LDB,
				double* W,
				double* WORK,
				int64_t* LWORK,
				int64_t* INFO
				);
}

template <typename T>
class IntegrationIteratorOzaki : public IntegrationIterator<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
				IntegrationIteratorOzaki();
	virtual		~IntegrationIteratorOzaki();

				IntegrationIteratorOzaki(	Context<T>* ctx,
												int numEnergiesPerBlock = -1 );

	virtual int		initialize();

	// Compute Ozaki poles and residues
	void computeOzakiPoles(	const int numPoles,
							T * poles,
							T * residues );

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
	
};

#endif

