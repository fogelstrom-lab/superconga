#include <math.h>
#include "Parameters.h"
#include <Grid.h>

#ifndef PI
#define PI 3.1415926535897932384
#endif

/*
template <typename T>
Parameters<T>::Parameters() :
	_gridWidth( 1.0 ),
	_angularResolution( 5 ),
	_cutOff( 10 ),
	_T_Tc( 0.5 ),
	_magneticField( 0.0 ),
	_convergence( 0.00000001 ),
	_numFrequencies( 0 ),
	_numEnergies( 0 ),
	_energies( 0 ),
//	_numEnergiesPerBlock( 0 ),
	_signEnergy(1),
	_LDOS_energyMin( 0.0 ),
	_LDOS_energyMax( 0.0 ),
	_smearingFactor( 0.01 ),
	_param1(0.0), _param2(0.0), _param3(0.0), _param4(0.0), _param5(0.0),
	_version( 1 ),
	_isValid( 1 )
{
	setGridResolution( 101 );
//	setMatsubaraFrequencies();
}


template <typename T>
Parameters<T>::~Parameters()
{
	if ( _energies ) delete _energies;
}
*/
namespace v1
{

template <typename T>
Parameters<T>& Parameters<T>::operator=( const Parameters<T>& right )
{
//	std::cout << "A\n";
	operatorEq(right);
	return *this;
}


template <typename T>
template <typename T2>
Parameters<T>& Parameters<T>::operator=( const Parameters<T2>& right )
{
//	std::cout << "B\n";
	operatorEq(right);
	return *this;
}


template <typename T>
template <typename T2>
Parameters<T>::operator Parameters<T2>() const
{
//	std::cout << "C\n";
	Parameters<T2> other;
	other = *this;
	return other;
}


template <typename T>
template <typename T2>
Parameters<T>& Parameters<T>::operatorEq( const Parameters<T2>& right )
{
//	std::cout << "Eq\n";

	if ( _version != right.getVersion() )
		std::cout << "Parameters<T>::operator=(): Warning, differing versions of Parameters class\n";

	_gridWidth = right.getGridWidth();
	_angularResolution = right.getAngularResolution();
	_cutOff = right.getCutOff();
	_T_Tc = right.getTemperature();
	_magneticField = right.getMagneticFieldStrength();
	_convergence = right.getConvergenceCriterion();
	_numFrequencies = right.getNumMatsubaraFrequencies();
	_numEnergies = right.getNumEnergies();
//	_energies
//	_signEnergy(1),
	_LDOS_energyMin = right.getLDOSEnergyMin();
	_LDOS_energyMax = right.getLDOSEnergyMax();
	_smearingFactor = right.getSmearingFactor();
	_isValid = right.isValid();

	setGridResolution( right.getGridResolutionBase() );
//	setLDOSEnergies(_LDOS_energyMin,_LDOS_energyMax,_smearingFactor);
	initEnergies();

	return *this;
}


template <typename T>
void Parameters<T>::setLDOSEnergies( T e_min, T e_max, T smearingFactor, int numEnergies )
{
	_LDOS_energyMin = e_min;
	_LDOS_energyMax = e_max;
	_smearingFactor = smearingFactor;
	
	if ( e_max == 0.0 || e_max == e_min )
		_numEnergies = 1;
	else	
		_numEnergies = numEnergies;

	setLDOSEnergies();
}


template <typename T>
void Parameters<T>::setLDOSEnergies()
{
	T LDOS_energyIncrement;

	if ( _numEnergies > 1 )
		LDOS_energyIncrement = (_LDOS_energyMax - _LDOS_energyMin) / (T)(_numEnergies-1);
	else
		LDOS_energyIncrement = 0.0;
	
	// Enter explicit energy values into array
	if ( _energies ) delete _energies;
	
	_energies = new typename Grid<T>::CPU(_numEnergies,1);
	
	T *energies_re = _energies->getDataPointer(0,Type::real);
	T *energies_im = _energies->getDataPointer(0,Type::imag);

	for ( int i=0; i<_numEnergies; ++i )
	{
		energies_re[i] = _LDOS_energyMin + i * LDOS_energyIncrement;
		energies_im[i] = _smearingFactor;
	}

//	_energies->copyDataToGPU();	
}


template <typename T>
void Parameters<T>::setGridWidth( T gridWidth )
{
	_gridWidth = gridWidth;
	setGridElementSize();
}


template <typename T>
void Parameters<T>::setGridResolution( int N_grid_base )
{
	_N_grid_base = N_grid_base;
	int baseIsOdd = _N_grid_base % 2;

	// Set N_grid_coherence
	_N_grid_coherence = sqrt(2.0) * _N_grid_base + 1;	// Add 1 to make sure rotation fits inside

	int gammaIsOdd = _N_grid_coherence % 2;
	
	if ( baseIsOdd != gammaIsOdd )
		_N_grid_coherence += 1;

	// Set N_grid_boundary
	_N_grid_boundary = sqrt(2.0) * _N_grid_coherence + 1;	// Add 1 to make sure rotation fits inside

	int boundaryIsOdd = _N_grid_boundary % 2;
	
	if ( baseIsOdd != boundaryIsOdd )
		_N_grid_boundary += 1;

	// Set new grid element width
	setGridElementSize();
}


template <typename T>
void Parameters<T>::setGridElementSize()
{
	_gridElementSize = _gridWidth / ((T)_N_grid_base-1);
}


template <typename T>
void Parameters<T>::setMatsubaraFrequencies()
{
	int mult = 1 + (int)(_signEnergy==0);

	_numFrequencies = _cutOff / _T_Tc + 2;
	_numEnergies = mult * _numFrequencies;
		
	// Enter explicit energy value into array
	if ( _energies ) delete _energies;
	
	_energies = new typename Grid<T>::CPU(_numEnergies,1);
	
	T *energies_re = _energies->getDataPointer(0,Type::real);
	T *energies_im = _energies->getDataPointer(0,Type::imag);

	if ( _signEnergy == 0 )
	{
		for ( int i=0; i<_numEnergies; ++i )
		{
			energies_re[i] = 0.0;
			energies_im[i] = (PI*_T_Tc) * (2.0*(i-_numEnergies/2) + 1.0);
		}
	}
	else
	{
		for ( int i=0; i<_numEnergies; ++i )
		{
			energies_re[i] = 0.0;
			energies_im[i] = _signEnergy * (PI*_T_Tc) * (2.0*i + 1.0);
		}
	}
//	_energies->copyDataToGPU();
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
	
	if ( getNumMatsubaraFrequencies() == getNumEnergies() && getLDOSEnergyMin() == getLDOSEnergyMax() )
		setMatsubaraFrequencies();
}

	
template <typename T>
void Parameters<T>::setTemperature( T T_Tc )
{
	_T_Tc = T_Tc;

	if ( getNumMatsubaraFrequencies() == getNumEnergies() && getLDOSEnergyMin() == getLDOSEnergyMax() )
		setMatsubaraFrequencies();		
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
	_numFrequencies = 1;
	_numEnergies = 1;
}


template <typename T>
T Parameters<T>::getAngleIncrement() const
{
	return 2.0*PI/(T)_angularResolution;
}


template <typename T>
void Parameters<T>::initEnergies()
{
	_energies = 0;

	if ( getNumMatsubaraFrequencies() != getNumEnergies() )
		setLDOSEnergies();
	else if ( getNumMatsubaraFrequencies() == getNumEnergies() && getLDOSEnergyMin() != getLDOSEnergyMax() )
		setLDOSEnergies();
	else
		setMatsubaraFrequencies();	
}


template <typename T>
int Parameters<T>::read( std::string filename )
{
	using namespace std;

	int numBytes = sizeof(*this);

	ifstream file( filename.c_str(), ios::in | ios::binary );

	if ( file.is_open() )
	{
		file.read( (char*)this, numBytes );
		file.close();

		initEnergies();
		// _energies = 0;

		// if ( getNumMatsubaraFrequencies() != getNumEnergies() )
		// 	setLDOSEnergies();
		// else if ( getNumMatsubaraFrequencies() == getNumEnergies() && getLDOSEnergyMin() != getLDOSEnergyMax() )
		// 	setLDOSEnergies();
		// else
		// 	setMatsubaraFrequencies();

		return 0;
	}
	else
	{
		cout << "Could not load file\n";
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
void Parameters<T>::print() const
{
	using namespace std;

	cout << "Grid Resolution Base: " << _N_grid_base << endl;
	cout << "Grid Width: " << _gridWidth  << endl;
	cout << "Grid Element Size: " << _gridElementSize << endl;
	cout << "Angular Resolution: " << _angularResolution << endl;
	cout << "Matsubara Frequencies: " << _numFrequencies << endl;
	cout << "Energy levels: " << _numEnergies << endl;
	cout << "Temperature: " << _T_Tc << endl;
	cout << "Magnetic Field Strength: " << _magneticField << endl;
	cout << "LDOS energy range: (" << _LDOS_energyMin << ", " << _LDOS_energyMax << ")" << endl;
	cout << "Convergence Criterion: " << _convergence << endl;
//	cout << "GridData adress: " << _energies << endl;
	cout << "GridData energies: ";
//	if ( _energies != 0 )
//		_energies->printAllData();
	//else
	//	cout << "NIL\n\n";	
}

}