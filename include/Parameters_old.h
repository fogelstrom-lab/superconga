#ifndef _PARAMETERSOLD_H
#define _PARAMETERSOLD_H

#include <string>

template<typename,typename> class GridData;

namespace v1
{
template <typename T>
class Parameters
{
public:
			Parameters() : _version(1), _energies(0) {}
			~Parameters() { if ( _energies && _isValid ) delete _energies; }
	
			Parameters<T>& operator=( const Parameters<T>& right );

			template <typename T2>
			Parameters<T>& operator=( const Parameters<T2>& right );

			template <typename T2>
			Parameters<T>& operatorEq( const Parameters<T2>& right );

			template <typename T2>
			operator Parameters<T2>() const;

	// Getters
	int		getGridResolutionBase() const			{ return _N_grid_base; }
	int		getGridResolutionCoherence() const		{ return _N_grid_coherence; }
	int		getGridResolutionBoundary() const		{ return _N_grid_boundary; }
	int		getGridResolutionDeltaRotated() const	{ return _N_grid_coherence; }

	T		getGridWidth() const				{ return _gridWidth; }
	T		getGridElementSize() const			{ return _gridElementSize; }
	int		getAngularResolution() const		{ return _angularResolution; }
	T		getAngleIncrement() const;

	int		getNumMatsubaraFrequencies() const	{ return _numFrequencies; }
	T		getTemperature() const				{ return _T_Tc; }
	T		getMagneticFieldStrength() const	{ return _magneticField; }

	T		getConvergenceCriterion() const		{ return _convergence; }
	
	T 		getCutOff() const					{ return _cutOff; }
	int		getNumEnergies() const				{ return _numEnergies; }
	
	T		getLDOSEnergyMin() const			{ return _LDOS_energyMin; }
	T		getLDOSEnergyMax() const			{ return _LDOS_energyMax; }
	T 		getSmearingFactor() const			{ return _smearingFactor; }
	
	typename Grid<T>::CPU*	getEnergies() const			{ return _energies; }
	
	int 	isValid() const { return _isValid; }

	
	// Setters
	void	setGridWidth( T gridWidth );
	void	setGridResolution( int N_grid_base );
	void	setAngularResolution( int angularResolution );
	
	void	setCutOff( T cutOff );
	void	setTemperature( T T_Tc );
	void	setMagneticFieldStrength( T magneticField );
	
	void	setSignEnergy( int sign ) { _signEnergy = sign; }
	void	setLDOSEnergies( T e_min, T e_max = 0.0, T smearingFactor = 0.01, int numEnergies = 1 );
	void	incrementLDOSEnergyIndex( const int idx_inc = 1 );
	
	void	setConvergenceCriterion( T convergence );
	
	void	setDebug();

	void	setValid() { _isValid = 1; }
	void	setInvalid() { _isValid = 0; }
	
	
	// File I/O
	int		read( std::string filename );
	int		write( std::string filename ) const;


	// Helpers
	void	setMatsubaraFrequencies();
	void	setLDOSEnergies();
	void	initEnergies();
	int 	getVersion() const { return _version; }
	void	print() const;


private:	
	// Discrete properties
	int		_N_grid_boundary;		// Number of elements for boundary/initial grid
	int		_N_grid_coherence;		// Number of elements for coherence functions (gamma,gammabar) grid
	int		_N_grid_base;			// Number of elements for order parameter solution
	
	T		_gridWidth;				// In natural units: h_bar * vF / (kB*Tc)
	T		_gridElementSize;		// Width of one grid element

	int		_angularResolution;		// Number of angles to solve for
	int		_numFrequencies;		// Number of Matsubara frequencies
	
	int		_numEnergies;			// Number of energies (Matsubara or Real)

	typename Grid<T>::CPU*	_energies;
	
	// Physical properties	
	T		_cutOff;
	T		_T_Tc;					// T/T_c
	T		_magneticField;
	T		_magneticFlux;

	// Spare parameters
	T		_param1, _param2, _param3, _param4, _param5;

	int		_signEnergy;

	T		_LDOS_energyMin;		// Lowest energy to compute LDOS for
	T		_LDOS_energyMax;		// Highest    "     "
	T		_smearingFactor;

	
	// Numerical properties
	T		_convergence;
	
	// Other
	const int	_version;
	int 		_isValid;

	// Helper functions
	void	setGridElementSize();
};
}
#endif

