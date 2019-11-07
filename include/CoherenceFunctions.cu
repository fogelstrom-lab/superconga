#include <CoherenceFunctions.h>

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


template <typename T>
CoherenceFunctions<T>::initialize()
{
	std::cout << "CoherenceFunctions<T>::initialize()\n";

	typedef ContextModule<T> C;

	const int N1 = C::getParameters()->getGridResolutionCoherence();		
	const int Ne = C::getIntegrationIterator()->getEnergyIterator()->getNumInterval();

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
}
