#ifndef PI
#define PI 3.1415926535897932384
#endif

#ifndef _ORDERPARAMETERCOMPONENT_H
#define _ORDERPARAMETERCOMPONENT_H

#include <Parameters.h>
#include <Grid.h>

template <typename T>
class OrderParameterComponent
{
	typedef typename Grid<T>::GPU GridGPU;

public:
	// Constructors and Destructors
					OrderParameterComponent();
	virtual			~OrderParameterComponent();

					OrderParameterComponent( const OrderParameterComponent<T>& other );

					OrderParameterComponent( Parameters<T>* param, T Tcthis_Tc = 1.0 );

	virtual OrderParameterComponent<T>* clone() const = 0;

	// Functions
	int 			initialize();

	Parameters<T>*	getParameters() const { return _param; }
	void			setParameters( Parameters<T>* param ) { _param = param; }

	GridGPU*		getOrderParameterField() const { return _deltaComp; }
	T				getRelativeTc() const { return _Tcthis_Tc; }
	void			setRelativeTc( const T Tc ) { _Tcthis_Tc = Tc; }
				
	virtual void	computeMomentumDependence( GridGPU* target, const T angle ) = 0;

	virtual void	initCouplingConstant( const T* energies, const T* residues );
	
	virtual void	computeCouplingConstant( );
	virtual void	computeCouplingConstant( GridGPU* target );

	virtual void	computeCouplingConstant( const T* energies, const T* residues );
	virtual void	computeCouplingConstant( GridGPU* target, const T* energies, const T* residues );

	virtual double	deltaBulk() const;

	char*			getType() const { return _type; }
	void			setType( char* type ) { _type = type; }

private:
	template <typename S>
	friend void		swap( OrderParameterComponent<S>& A, OrderParameterComponent<S>& B );

	// Global model parameters
	Parameters<T>*	_param;

	// Relative critical temperature  Tc_thiscomponent / Tc
	T 				_Tcthis_Tc;

	// Store coupling constant, calculate only once
	// CAUTION!!! Upon changing T_Tc/energies/similar, need to re-calculate
	T				_couplingConstant;
	// True if coupling constant has been initialized, then it never needs to be
	// re-calculated
	int				_initCouplingConstant;

	// Order parameter component
	GridGPU*		_deltaComp;
	
	char*			_type;
};

#endif
