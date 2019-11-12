#include "OrderParameterComponent.h"


template <typename T>
OrderParameterComponent<T>::OrderParameterComponent()
	:	_param(0), _deltaComp(0), _Tcthis_Tc(1.0), _couplingConstant(0), _initCouplingConstant(0)
{}


template <typename T>
OrderParameterComponent<T>::OrderParameterComponent( const OrderParameterComponent<T>& other ) :
	_param(other._param), _Tcthis_Tc(other._Tcthis_Tc), _couplingConstant(other._couplingConstant),  _initCouplingConstant(other._initCouplingConstant)
{
	_type = other._type;
	if ( other._deltaComp ) _deltaComp = new GridGPU(*other._deltaComp);
}


template <typename T>
OrderParameterComponent<T>::OrderParameterComponent( Parameters<T>* param, T Tcthis_Tc )
	:	_param(param), _deltaComp(0), _Tcthis_Tc(Tcthis_Tc), _couplingConstant(0), _initCouplingConstant(0)
{
	// Do an initialization on creation so we can set initial value
	initialize();
}


template <typename T>
OrderParameterComponent<T>::~OrderParameterComponent()
{
	if ( _deltaComp ) delete _deltaComp;
}


template <typename T>
void swap( OrderParameterComponent<T>& A, OrderParameterComponent<T>& B )
{
	std::swap( A._param, B._param );
	std::swap( A._Tcthis_Tc, B._Tcthis_Tc );
	std::swap( A._deltaComp, B._deltaComp );
	std::swap( A._type, B._type );
	std::swap( A._couplingConstant, B._couplingConstant );
	std::swap( A._initCouplingConstant, B._initCouplingConstant );
}



template <typename T>
int OrderParameterComponent<T>::initialize()
{
	int N = _param->getGridResolutionBase();

	if ( _deltaComp && _deltaComp->dimX() != N )
		_deltaComp->resample(N,N);
	else
		GridGPU::initializeGrid( _deltaComp, N, N );

	return 0;
}


template <typename T>
void OrderParameterComponent<T>::initCouplingConstant(
	const T* energies, const T* residues )
{
	const double T_Tc = _param->getTemperature();
	const int numFrequencies = _param->getNumFrequencies();
	
	double energySum = 0.0;

	for ( int i=0; i<numFrequencies; ++i )
		energySum += residues[i] / energies[i];
	
	_couplingConstant = ( M_PI*T_Tc/(log(T_Tc) + log(1.0/_Tcthis_Tc) + 2.0*M_PI*T_Tc*energySum) );
	std::cout << "_couplingConstant: " << _couplingConstant << std::endl;

	_initCouplingConstant = 1;
}


template <typename T>
void OrderParameterComponent<T>::computeCouplingConstant( )
{
	computeCouplingConstant( _deltaComp );
}


template <typename T>
void OrderParameterComponent<T>::computeCouplingConstant( GridGPU* target )
{
	if ( ! _initCouplingConstant )
	{
		const double T_Tc = _param->getTemperature();
		const int numFrequencies = _param->getNumMatsubaraFrequencies();

		double matsubaraSum = 0.0;
		
		for ( int i=0; i<numFrequencies; ++i )
			matsubaraSum += 1.0/(2.0*double(i) + 1.0);
		
		_couplingConstant = ( M_PI*T_Tc/(log(T_Tc) + log(1.0/_Tcthis_Tc) + 2.0*matsubaraSum) );
	}

	(*target) *= _couplingConstant;
}



template <typename T>
void OrderParameterComponent<T>::computeCouplingConstant( const T* energies, const T* residues )
{
	computeCouplingConstant( _deltaComp, energies, residues );
}


template <typename T>
void OrderParameterComponent<T>::computeCouplingConstant(
	GridGPU* target, const T* energies, const T* residues )
{
	if ( ! _initCouplingConstant )
	{
		const double T_Tc = _param->getTemperature();
		const int numFrequencies = _param->getNumFrequencies();
		
		double energySum = 0.0;

		for ( int i=0; i<numFrequencies; ++i )
			energySum += residues[i] / energies[i];
		
		_couplingConstant = ( M_PI*T_Tc/(log(T_Tc) + log(1.0/_Tcthis_Tc) + 2.0*M_PI*T_Tc*energySum) );
	}

	(*target) *= _couplingConstant;
}

template <typename T>
double OrderParameterComponent<T>::deltaBulk() const
{	
	T T_Tc = _param->getTemperature();

	const double deltaConvergence = 1E-9;

	const double gMix = -0.1;
	const int numMatsubara = 1000.0/T_Tc;
	
	// Delta, initial guess
	double delta = 1.76 * sqrt(1.0-T_Tc);

	int isDeltaConverged = 0;
	double matsubaraSum, eps, g, deltaDiff;
	
	while ( !isDeltaConverged )
	{
		matsubaraSum = 0.0;

		for ( int i=0; i<=numMatsubara; ++i )
		{
			eps = M_PI * T_Tc * (2.0*i + 1.0);
			matsubaraSum += 1.0 / sqrt(eps*eps + delta*delta) - 1.0/eps;
		}
		matsubaraSum *= delta;
		
		g = delta * log(T_Tc) - 2.0*PI*T_Tc*matsubaraSum;
		deltaDiff = g * gMix;
		delta += deltaDiff;
		
		if ( abs(deltaDiff/delta) < deltaConvergence )
			isDeltaConverged = 1;
	}
	
	return delta;	
}
