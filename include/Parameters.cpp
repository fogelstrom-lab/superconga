#include <math.h>
#include "Parameters.h"


template <typename T>
Parameters<T>::Parameters( Context<T>* ctx ) :
	_gridWidth( 1.0 ),
	_angularResolution( 0 ),
	_cutOff( 10 ),
	_T_Tc( 0.5 ),
	_magneticField( 0.0 ),
	_convergence( 0.0001 ),
	_numEnergies( 0 ),
	_energyMin( 0.0 ),
	_energyMax( 0.0 ),
	_smearingFactor( 0.01 ),
	_param1(0.0), _param2(100.0), _param3(0.0), _param4(0.0), _param5(0.0),
	_version( 2 ),
	_isValid( 1 )
{
	ContextModule<T>::setContext(ctx);
	ctx->set( this );

	setGridResolution( 101 );
}

template <typename T>
Parameters<T>& Parameters<T>::operator=( const Parameters<T>& right )
{
	operatorEq(right);
	return *this;
}


template <typename T>
template <typename T2>
Parameters<T>& Parameters<T>::operator=( const Parameters<T2>& right )
{
	operatorEq(right);
	return *this;
}


template <typename T>
template <typename T2>
Parameters<T>::operator Parameters<T2>() const
{
	Parameters<T2> other;
	other = *this;
	return other;
}


// Is this necessary?
template <typename T>
template <typename T2>
Parameters<T>& Parameters<T>::operatorEq( const Parameters<T2>& right )
{
	if ( _version != right.getVersion() )
		std::cout << "Parameters<T>::operator=(): Warning, differing versions of Parameters class\n";

	_gridWidth = right.getGridWidth();
	_angularResolution = right.getAngularResolution();
	_cutOff = right.getCutOff();
	_T_Tc = right.getTemperature();
	_magneticField = right.getMagneticFieldStrength();
	_convergence = right.getConvergenceCriterion();
	_numEnergies = right.getNumEnergies();
	_energyMin = right.getEnergyMin();
	_energyMax = right.getEnergyMax();
	_smearingFactor = right.getSmearingFactor();
	_isValid = right.isValid();

	_param1 = right._param1;
	_param2 = right._param2;

	setGridResolution( right.getGridResolutionBase() );

	return *this;
}


template <typename T>
int Parameters<T>::getNumFrequencies() const
{
	if ( _param1 != 0.0 )
		return _param1;
	else
		return (_cutOff / _T_Tc + 2); 
}


template <typename T>
void Parameters<T>::setEnergyRange( T e_min, T e_max, int numEnergies, T smearingFactor )
{
	_energyMin = e_min;
	_energyMax = e_max;
	_smearingFactor = smearingFactor;
	
	if ( e_max == 0.0 || e_max == e_min )
		_numEnergies = 1;
	else	
		_numEnergies = numEnergies;

	if ( numEnergies == 0 )
		_numEnergies = 0;
}


template <typename T>
void Parameters<T>::setGridWidth( T gridWidth )
{
	_gridWidth = gridWidth;
	setGridElementSize();
}


template <typename T>
void Parameters<T>::setGridResolution( const int N_grid_base )
{
	_N_grid_base = N_grid_base;
	_N_grid_coherence = calcGridResolutionRotated(N_grid_base);

	// Set new grid element width
	setGridElementSize();
}


template <typename T>
void Parameters<T>::setGridElementSize()
{
	_gridElementSize = _gridWidth / ((T)_N_grid_base-1);
}


template <typename T>
void Parameters<T>::setAngularResolution( int angularResolution )
{
	_angularResolution = angularResolution;
}



template <typename T>
void Parameters<T>::setCutOff( T cutOff )
{
	_cutOff = cutOff;
}

	
template <typename T>
void Parameters<T>::setTemperature( T T_Tc )
{
	_T_Tc = T_Tc;
}


template <typename T>
void Parameters<T>::setMagneticFieldStrength( T magneticField )
{
	_magneticField = magneticField;
}

	
template <typename T>
void Parameters<T>::setConvergenceCriterion( T convergence )
{
	_convergence = convergence;
}


template <typename T>
void Parameters<T>::setDebug()
{
}


template <typename T>
T Parameters<T>::getAngleIncrement() const
{
	return 2.0*M_PI/(T)_angularResolution;
}


template <typename T>
int Parameters<T>::read( std::string filename )
{
	using namespace std;

	ifstream file( filename.c_str(), ios::in | ios::binary );
	int res = read( file, _version );
	file.close();

	return res;
}


template <typename T>
int Parameters<T>::read( std::string filename, const int version )
{
	using namespace std;

	int res;

	if ( version == _version )
		 return read( filename );
	else
	{
		ifstream file( filename.c_str(), ios::in | ios::binary );
		res = read( file, version );
		file.close();
		return res;
	}
}


template <typename T>
int Parameters<T>::read( std::ifstream& file, const int version )
{
	using namespace std;

	if ( file.is_open() )
	{
		if ( version == _version )
		{
			int numBytes = sizeof(*this);

			file.read( (char*)this, numBytes );

			return 0;
		}
		else
		{
			v1::Parameters<T> pr;
			int numBytes = sizeof(pr);

			file.read( (char*)&pr, numBytes );
			pr.setInvalid();

			setGridResolution( pr.getGridResolutionBase() );
			setGridWidth( pr.getGridWidth() );
			setAngularResolution( pr.getAngularResolution() );
			setCutOff( pr.getCutOff() );
			setMagneticFieldStrength( pr.getMagneticFieldStrength() );
			setTemperature( pr.getTemperature() );
			setConvergenceCriterion( pr.getConvergenceCriterion() );

			T e_min = pr.getLDOSEnergyMin();
			T e_max = pr.getLDOSEnergyMax();
			int numEnergies = pr.getNumEnergies();

			if ( e_min != e_max && numEnergies != pr.getNumMatsubaraFrequencies() )
				setEnergyRange( pr.getLDOSEnergyMin(), pr.getLDOSEnergyMax(), pr.getSmearingFactor(), numEnergies );
			else
				setEnergyRange( 0.0, 0.0, 0.01, 0 );
			return 0;			
		}
	}
	else
	{
		cout << "Could not load file, version: " << version << "\n";
		return 1;
	}	
}


template <typename T>
int Parameters<T>::write( std::string filename ) const
{
	using namespace std;

	int numBytes = sizeof(*this);

	ofstream file( filename.c_str(), ios::out | ios::binary | ios::trunc );

	if ( file.is_open() )
	{
		file.write( (char*)(this), numBytes );
		file.close();
		return 0;
	}
	else
	{
		cout << "Could not save file: " << filename << endl;
		return 1;
	}
}


template <typename T>
int Parameters<T>::calcGridResolutionRotated( const int N_grid_base )
{
	int baseIsOdd = N_grid_base % 2;

	// Set N_grid_rotated
	int N_grid_rotated = sqrt(2.0) * N_grid_base + 1;	// Add 1 to make sure rotation fits inside

	int gammaIsOdd = N_grid_rotated % 2;
	
	if ( baseIsOdd != gammaIsOdd )
		N_grid_rotated += 1;

	return N_grid_rotated;
}


template <typename T>
void Parameters<T>::print() const
{
	using namespace std;

	cout << "Grid Resolution Base: " << _N_grid_base << endl;
	cout << "Grid Width: " << _gridWidth  << endl;
	cout << "Grid Element Size: " << _gridElementSize << endl;
	cout << "Angular Resolution: " << _angularResolution << endl;
	cout << "Temperature: " << _T_Tc << endl;
	cout << "Magnetic Field Strength: " << _magneticField << endl;
	cout << "Spectral energy range: (" << _energyMin << ", " << _energyMax << ")" << endl;
	cout << "Spectral energy levels: " << _numEnergies << endl;
	cout << "Convergence Criterion: " << _convergence << endl;
	cout << "Cutoff parameter: " << _cutOff << endl;
}
