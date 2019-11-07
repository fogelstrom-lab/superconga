//
// Collection of functions to set the initial condition of the order parameter.
// All member functions are static so this struct should not be instantiated.
//
#ifndef _INITIAL_H
#define _INITIAL_H

#ifndef PI
#define PI (T)3.1415926535897932384
#endif

#include <math.h>
#include "OrderParameterComponent.h"

#include "Initial_Kernel.cu"

template <typename T>
struct Initial
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
	// Computes bulk value of s-wave order parameter. Can be used as initial condition.
	static double computeDeltaBulk( const double T_Tc, const double deltaConvergence = 1E-9 );
	

	// The following functions set the deltaComponent field to...

	// ...delta bulk
	static void	bulk( OrderParameterComponent<T>* deltaComponent, const T mult_re = 1.0, const T mult_im = 0.0 );
	
	// ...specified complex valued constant 
	static void	constant( OrderParameterComponent<T>* deltaComponent, const T re = 1.0, const T im = 0.0 );

	// ...the input grid argument
	static void grid( OrderParameterComponent<T>* deltaComponent, const GridGPU& grid );

	// ...a given phase winding (one flux vortex has a phase winding of 2pi)
	static void	phaseWinding(	OrderParameterComponent<T>* deltaComponent,
								const T numWinding,
								const T cutOffRadius = 0.0,
								const T phaseShift = 0.0,
								const T center_x = 0.0, const T center_y = 0.0 );

	// ...delta bulk + a perturbation in the phase (to push solution out of meta-stable initial state)
	static void random( OrderParameterComponent<T>* deltaComponent, const T mix = 0.5, const T phase = 0.0 );

	// Mixture between random() and phaseWinding():
	//     phaseWinding with an additional random phase factor
	static void	perturbedPhaseWinding(	OrderParameterComponent<T>* deltaComponent,
										const T numWinding,
										const T mix = 0.5,
										const T cutOffRadius = 0.0,
										const T phaseShift = 0.0,
										const T center_x = 0.0, const T center_y = 0.0 );

	// ...the grid in the input file argument
	static int	read( OrderParameterComponent<T>* deltaComponent, const char* filename );
	static int	read( OrderParameterComponent<T>* deltaComponent, const char* filename, int filenum );
};

#endif

