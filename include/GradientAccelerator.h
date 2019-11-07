#ifndef _GRADIENTACCELERATOR_H
#define _GRADIENTACCELERATOR_H

#include <Grid.h>

template <typename T>
class GradientAccelerator : public ContextModule<T>
{
		typedef typename Grid<T>::GPU GridGPU;
public:
			GradientAccelerator();
			~GradientAccelerator();
			
			GradientAccelerator( Context<T>* ctx );

	virtual int initialize();

	void	reset();
	int		step( T residual, GridGPU* grid );
	
	int		isPreConverged() const { return _criterionMet; }
	int		isConverged() const { return int(_criterionMet && _relaxFinalCount >= _relaxFinal && _numLoops >= _minIter); }
	
	void	resetCount() { _acount = 0; }

	// Setters
	void	setMinIterations( int minIter ) { _minIter = minIter; }
	void	setMaxIterations( int maxIter ) { _maxIter = maxIter; }
	void	setRelaxCount( int relax ) { _relax = relax; }
	void	setRelaxFinal( int relax ) { _relaxFinal = relax; }
	void	setPreConverged( int is_converged ) { _criterionMet = is_converged; }


private:
	
	int		_minIter;			// Minimum # of iterations at start before applying acceleration
	int		_maxIter;			// Maximum # of iterations until applying _relaxFinal iterations, then report convergence
	int		_relax;				// Iterations between applying gradient acceleration
	int		_relaxFinal;		// Iterations after convergence reached
	
	int		_acount;			// Iteration count after latest acceleration
	int		_relaxFinalCount;	// Relaxation iteration count after convergence
	int		_numLoops;			// Total # of iterations
	int		_doAccel;			// Acceleration flag
	int		_criterionMet;		// Is convergence criterion reached?

	T		_residual;			// Residual current iteration
	T		_residualLast;		// Residual last iteration
	T		_convergenceRate;	// Fractional change of residual
	T		_rate[2];			// Stored convergence fractional rates

	GridGPU*	_g1;		// Stored grids from two iterations, to compute gradient
	GridGPU*	_g2;
};

#endif

