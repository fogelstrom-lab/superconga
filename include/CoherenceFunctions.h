#ifndef _COHERENCEFUNCTIONS_H
#define _COHERENCEFUNCTIONS_H

#include <Grid.h>
#include <Context.h>
#include <ContextModule.h>

template <typename T>
class CoherenceFunctions : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
				CoherenceFunctions();
	virtual 	~CoherenceFunctions();

				CoherenceFunctions( Context<T>* ctx, Type::coherence cf_type );

	virtual int initialize();

	int 			getNumComponents() const {return _Nrow*_Ncol;}
	Type::coherence getCoherenceFunctionSymmetry() const {return _cf_type;}

	GridGPU*	getPositive(int row=0, int col=0) const {return _pos[row*_Ncol+col];}
	GridGPU*	getNegative(int row=0, int col=0) const {return _neg[row*_Ncol+col];}

	GridGPU*	getPositive(int idx) const {return _pos[idx];}
	GridGPU*	getNegative(int idx) const {return _neg[idx];}

private:
	Type::coherence 		_cf_type;

	// 1x1 for singlet (scalar), 2x2 for triplet (2x2 matrix)
	const int 				_Nrow;
	const int 				_Ncol;

	std::vector<GridGPU*>	_pos;		// gamma
	std::vector<GridGPU*>	_neg;		// gammabar
};

template <typename T>
CoherenceFunctions<T>::CoherenceFunctions()
	:	ContextModule<T>()
{}


template <typename T>
CoherenceFunctions<T>::CoherenceFunctions(Context<T>* ctx, Type::coherence cf_type )
	:	ContextModule<T>(ctx), _cf_type(cf_type)
{
	if ( cf_type == Type::singlet )
	{
		_Nrow = 1;
		_Ncol = 1;
	}

	if ( cf_type == Type::triplet )
	{
		_Nrow = 2;
		_Ncol = 2;
	}

	const int components = getNumComponents();

	_pos = std::vector<GridGPU*>( components );
	_neg = std::vector<GridGPU*>( components );
}


template <typename T>
CoherenceFunctions<T>::~CoherenceFunctions()
{
	for (int i=0; i<_pos.size(); ++i)
	{
		if (_pos[i]) delete _pos[i];
		if (_neg[i]) delete _neg[i];
	}
}


// This had no return type before, not even void-
template <typename T>
int CoherenceFunctions<T>::initialize()
{
	std::cout << "CoherenceFunctions<T>::initialize()\n";

	typedef ContextModule<T> C;

	const int N1 = C::getParameters()->getGridResolutionCoherence();		
	const int Ne = C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

    const int components = getNumComponents();
	for ( int i=0; i<components; ++i )
	{
		GridGPU::initializeGrid( _pos[i], N1, N1, Ne, Type::complex );
		GridGPU::initializeGrid( _neg[i], N1, N1, Ne, Type::complex );

		_pos[i]->setZero();
		_neg[i]->setZero();
	}

	long long bytes = N1*N1*Ne*2*sizeof(T)*2*components;
	std::cout << "Coherence functions storage data: " << bytes << " bytes.\n";
	std::cout << "numFields = " << Ne << "\n\n";
	return 0;
}


#endif
