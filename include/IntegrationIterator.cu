#include "IntegrationIterator.h"

template <typename T>
IntegrationIterator<T>::IntegrationIterator() :
	_energies(0), _energy_iter(0), _numIter(0), _debug_store_grid(0), _residues(0)
{}


template <typename T>
IntegrationIterator<T>::IntegrationIterator( Context<T>* ctx, int numEnergiesPerBlock ) :
	ContextModule<T>(ctx), _energies(0), _energy_iter(0), _numEnergiesPerBlock(numEnergiesPerBlock),
	_numIter(0), _debug_store_grid(0), _debug_store_angle(-1), _residues(0)
{
	std::cout << "IntegrationIterator<T>::IntegrationIterator( Context<T>* context )\n";
}


template <typename T>
IntegrationIterator<T>::~IntegrationIterator()
{
	std::cout << "~IntegrationIterator()\n";

	if ( _debug_store_grid ) delete _debug_store_grid;
	if ( _energies ) delete _energies;
	if ( _energy_iter ) delete _energy_iter;
	if ( _residues ) delete _residues;
}


template <typename T>
int IntegrationIterator<T>::initialize()
{
	std::cout << "IntegrationIterator::initialize()\n";
	initIterators();
	return 0;
}


template <typename T>
void IntegrationIterator<T>::initIterators()
{
	_numAngles = ContextModule<T>::getContext()->getParameters()->getAngularResolution();
	setCurrentAngleIndex(0);
	_numIter = 0;
}


template <typename T>
T IntegrationIterator<T>::getCurrentAngleRadians() const
{
	return ((T)_currentAngleIndex/(T)_numAngles) * (T)2.0*PI;
}


template <typename T>
void IntegrationIterator<T>::setNumEnergiesPerBlock( int num )
{
	if ( num != _numEnergiesPerBlock )
	{
		_numEnergiesPerBlock = num;
		initialize();
	}
}