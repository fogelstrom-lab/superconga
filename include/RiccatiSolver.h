//
// Base class for the Riccati equation solver.
//
#ifndef _RICCATISOLVER_H
#define _RICCATISOLVER_H

#include <cplx.h>
#include <Parameters.h>
#include <ParameterIterator.h>
#include <Grid.h>
#include <GeometryGroup.h>
#include <IntegrationIterator.h>
#include <BoundaryStorage.h>
#include <BoundaryCondition.h>

/////////////////////////////////////////////////////////
// MAYBE PUT POINTER TO ENERGIES AND ENERGYITERATOR HERE
/////////////////////////////////////////////////////////

// This function could not be put into RiccatiSolver class, because of template parameter issues
template <typename T, template <typename> class VP>
void computeEnergyShift( ContextModule<T>* mod, typename Grid<T>::GPU* eShift, const int angle_idx );


template <typename T>
class RiccatiSolver : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
	// Constructors and Destructors
					
					RiccatiSolver();
	virtual			~RiccatiSolver();

					RiccatiSolver( Context<T>* ctx );

	// Allocate memory
	virtual int 	initialize();

	// Get pointers to storage of Riccati solutions (gamma/coherence functions).
	// These grids have dimensions (x,y,energy)
	GridGPU*		getGamma() const { return _gamma; }
	GridGPU*		getGammabar() const { return _gammabar; }

	// Functions related to Boundary condition:

	// Get pointer to sparse array where boundary values (for coherence functions) are stored.
	BoundaryStorage<T>*		getBoundaryStorage() const { return _storage; }

	// Add boundary condition.
	void 					add( BoundaryCondition<T>* bc ) { _bc_list.push_back(bc); }
	
	int 					getNumBoundaryConditions() const { return _bc_list.size(); }
	BoundaryCondition<T>* 	getBoundaryCondition(const int idx) { return _bc_list[idx]; }

	// Vector potential
	GridGPU*		getInducedVectorPotential() const { return _ps; }
	GridGPU*		getEnergyShift() const { return _eShift; }

	// Magnetic field
	int				doComputeInducedMagneticField() const { return _computeInduced; }
	void			setComputeInducedMagneticField( int induced ) { _computeInduced = induced; }
	void 			computeInducedMagneticField();

	// Functions
	void			initializeCoherenceFunctions();
	void			initializeBoundaryStorage();
	
	virtual void	computeInitialCondition(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* delta, const int angle_idx ) = 0;

	virtual void	computeBoundaryConditions();
						
	virtual void	computeCoherenceFunctions(
						const ParameterIterator<int>& energy_iter, const GridGPU* energies,
						const GridGPU* deltaRot, const int angle_idx ) = 0;


	// For Computation of Free energy (as in ComputeFreeEnergy2 class)
	T				getOrderParameterMultFactor() const { return _orderParameterMult; }
	void			setOrderParameterMultFactor( const T mult ) { _orderParameterMult = mult; }


	// Helper functions
	void			getKernelDimensions(
					dim3& blocksPerGrid, dim3& threadsPerBlock, const int numEnergies );

	void			getKernelDimensionsOptim(
					dim3& blocksPerGrid, dim3& threadsPerBlock, const int numEnergies, const int angle_idx );
	
private:

	//START ELIMINATING GAMMA AS CLASS MEMBERS!!! USE CoherenceFunctions CLASS INSTEAD!
	
	// Storage for riccati trajectories
	GridGPU*		_gamma;
	GridGPU*		_gammabar;

	// Holds (and reflects) gammas on boundary between interations
	BoundaryStorage<T>* _storage;

	// Boundary conditions
	std::vector<BoundaryCondition<T>*> _bc_list;

	// Functor object for energy shift by the vector potential e -> e + vF.pS (inclusion of magnetic field)
	//POTENTIAL<T>	_vpFunctor;

	// Switch to compute and include internally induced magnetic field
	int 			_computeInduced;

	// Vector potential field
	GridGPU*		_ps;

	// Precomputed field for energy shift
	GridGPU*		_eShift;

	// A flag to indicate that delta should be multiplied with a constant,
	// as required by the free energy computation by J. Sauls et al.
	// For all other computations, this is not used.
	T 				_orderParameterMult;

};

#endif
