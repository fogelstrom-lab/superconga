#ifndef _PARAMETERS_H
#define _PARAMETERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <ContextModule.h>

template<typename,typename> class GridData;

#include <Parameters_old.h>

template <typename T>
class Parameters : public ContextModule<T>
{
public:
			Parameters() : _version(2) {}
			~Parameters() {}

			Parameters( Context<T>* ctx );
	
			Parameters& operator=( const Parameters& right );

			template <typename T2>
			Parameters<T>& operator=( const Parameters<T2>& right );

			template <typename T2>
			Parameters<T>& operatorEq( const Parameters<T2>& right );

			template <typename T2>
			operator Parameters<T2>() const;


	virtual int initialize() { return 0; }

	// Getters
	int		getGridResolutionBase() const			{ return _N_grid_base; }
	int		getGridResolutionCoherence() const		{ return _N_grid_coherence; }
	int		getGridResolutionDeltaRotated() const	{ return _N_grid_coherence; }

	T		getGridWidth() const				{ return _gridWidth; }
	T		getGridElementSize() const			{ return _gridElementSize; }
	int		getAngularResolution() const		{ return _angularResolution; }
	T		getAngleIncrement() const;

	int		getNumFrequencies() const;
	int		getNumMatsubaraFrequencies() const	{ return getNumFrequencies(); }
	T		getTemperature() const				{ return _T_Tc; }
	T		getMagneticFieldStrength() const	{ return _magneticField; }

	T		getConvergenceCriterion() const		{ return _convergence; }
	
	T 		getCutOff() const					{ return _cutOff; }
	
	T		getEnergyMin() const				{ return _energyMin; }
	T		getEnergyMax() const				{ return _energyMax; }
	int 	getNumEnergies() const				{ return _numEnergies; }
	T 		getSmearingFactor() const			{ return _smearingFactor; }

	T		getPenetrationDepth() const			{ return _param2; }
	
	int 	isValid() const { return _isValid; }

	
	// Setters
	void	setGridWidth( T gridWidth );
	void	setGridResolution( const int N_grid_base );
	void	setAngularResolution( int angularResolution );
	
	void	setCutOff( T cutOff );
	void	setNumMatsubaraFrequencies( const int Ne ) { _param1 = Ne; }
	void	setNumFrequencies( const int Ne ) { _param1 = Ne; }
	void	setTemperature( T T_Tc );
	void	setMagneticFieldStrength( T magneticField );
	
	void	setEnergyRange( T e_min, T e_max, int numEnergies, T smearingFactor = 0.01 );
	void	setNumEnergies( int numEnergies ) { _numEnergies = numEnergies; }
	
	void	setConvergenceCriterion( T convergence );
	
	void	setPenetrationDepth( T penetrationDepth ) { _param2 = penetrationDepth; }

	void	setDebug();

	void	setValid() { _isValid = 1; }
	void	setInvalid() { _isValid = 0; }
	
	
	// File I/O
	int		read( std::string filename );
	int 	read( std::ifstream& file, const int version );
	int		read( std::string filename, const int version );
	int		write( std::string filename ) const;


	// Helpers
	static int 	calcGridResolutionRotated( const int N_base );
	int 		getVersion() const { return _version; }
	void		print() const;


private:	
	// Discrete properties
	int		_N_grid_boundary;		// Number of elements for boundary/initial grid
	int		_N_grid_coherence;		// Number of elements for coherence functions (gamma,gammabar) grid
	int		_N_grid_base;			// Number of elements for order parameter solution
	
	T		_gridWidth;				// In natural units: h_bar * vF / (kB*Tc)
	T		_gridElementSize;		// Width of one grid element

	int		_angularResolution;		// Number of angles to solve for
	int		_numEnergies;			// Number of spectral energies 

	// Physical properties	
	T		_cutOff;
	T		_T_Tc;					// T/T_c
	T		_magneticField;
	T		_magneticFlux;

	// Spare parameters
	T		_param1, _param2, _param3, _param4, _param5;

//	int		_signEnergy;

	T		_energyMin;			// Lowest energy to compute LDOS for
	T		_energyMax;			// Highest    "     "
	T		_smearingFactor;

	
	// Numerical properties
	T		_convergence;
	
	// Other
	const int	_version;
	int 		_isValid;

	// Helper functions
	void	setGridElementSize();
};

#endif

