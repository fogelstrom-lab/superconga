#include "Context.h"

template<typename T>
Context<T>::Context() :
	_param(0), _geom(0), _delta(0), _solver(0), _iter(0), /*_cf(0),*/ _acc(0), _compute_current(0),
	_compute_delta(0), _compute_time(0.0), _numIterations(0)
{}


template<typename T>
Context<T>::~Context()
{
	using namespace std;
	cout << "~Context()\n";

	if (_param)
	{
		delete _param;
		cout << "~Context(): Deleted Parameters\n";
	}

	if (_geom) 
	{
		delete _geom;
		cout << "~Context(): Deleted Geometry\n";
	}
	if (_delta)
	{
		delete _delta;
		cout << "~Context(): Deleted OrderParameter\n";
	}
	if (_solver)
	{	
		delete _solver;
		cout << "~Context(): Deleted RiccatiSolver\n";
	}
	if (_iter)
	{
		delete _iter;
		cout << "~Context(): Deleted IntegrationIterator\n";
	}
//	if (_cf)
//	{
//		delete _cf;
//		cout << "~Context(): Deleted CoherenceFunctions\n";
//	}
	if (_acc)
	{
		delete _acc;
		cout << "~Context(): Deleted GradientAccelerator\n";
	}

	for ( int i=0; i<getNumComputeObjects(); ++i )
	{
		delete _computeObjectList[i];
		cout << "~Context(): Deleted ComputeObject " << i << "\n";
	}

	Grid<float>::CPU::printAllocatedMemTotal();
	Grid<double>::CPU::printAllocatedMemTotal();
	Grid<int>::CPU::printAllocatedMemTotal();
	Grid<char>::CPU::printAllocatedMemTotal();

	Grid<float>::GPU::printAllocatedMemTotal();
	Grid<double>::GPU::printAllocatedMemTotal();
	Grid<int>::GPU::printAllocatedMemTotal();
	Grid<char>::GPU::printAllocatedMemTotal();
}


template <typename T>
int Context<T>::add( ComputeProperty<T>* property )
{
	_computeObjectList.push_back( property );
		
	return getNumComputeObjects();
}


template <typename T>
int Context<T>::initialize( int numEnergiesPerBlock )
{
	using namespace std;

	if ( _iter == 0 && getNumComputeObjects() > 0 )
	{
		cout << "Context::initialize() --> Creating IntegrationIterator\n";
		_computeObjectList[0]->createIntegrationIterator( numEnergiesPerBlock ); // sets _iter member
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): createIntegrationIterator" );
	}

	if ( _geom )
	{
		cout << "Context::initialize() --> Initializing Geometry\n";
		_geom->initialize();
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): GeometryGroup" );
	}

	if ( _delta )
	{
		cout << "Context::initialize() --> Initializing OrderParameter\n";
		_delta->initialize();
		//_delta->createCoherenceFunctions(); // sets _cf member
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): OrderParameter" );
	}

	if ( _iter )
	{
		cout << "Context::initialize() --> Initializing IntegrationIterator\n";
		_iter->initialize();
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): IntegrationIterator" );
	}

	// RiccatiSolver requires IntegrationIterator to be initialized first
	if ( _solver )
	{
		cout << "Context::initialize() --> Initializing RiccatiSolver\n";
		_solver->initialize();
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): RiccatiSolver" );
	}

	if ( _acc )
	{
		cout << "Context::initialize() --> Initializing GradientAccelerator\n";
		_acc->initialize();
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): GradientAccelerator" );
	}

	for ( int i=0; i<getNumComputeObjects(); ++i )
	{
		cout << "Context::initialize() --> Initializing ComputeObject " << i << ":\n";
		_computeObjectList[i]->initialize();
		ContextModule<T>::checkCudaError( "Context<T>::initialize(): ComputeProperty" );
	}

	_numIterations = 0;

	cout << "\n\nContext::initialize() --> Initialization complete!\n";

	Grid<float>::CPU::printAllocatedMemTotal();
	Grid<double>::CPU::printAllocatedMemTotal();
	Grid<int>::CPU::printAllocatedMemTotal();
	Grid<char>::CPU::printAllocatedMemTotal();
	
	Grid<float>::GPU::printAllocatedMemTotal();
	Grid<double>::GPU::printAllocatedMemTotal();
	Grid<int>::GPU::printAllocatedMemTotal();
	Grid<char>::GPU::printAllocatedMemTotal();

	return 0;
}


template <typename T>
void Context<T>::compute()
{
	_compute_time = csecond();

	_iter->compute();

	// Only use accelerator if we are computing order parameter
	ComputeOrderParameter<T>* compute_delta = getComputeOrderParameter();
	if (_acc && compute_delta )
	{
		_acc->step( compute_delta->getResidual(), _delta->getOrderParameterComponent(0)->getOrderParameterField() );
	}

	_compute_time = csecond() - _compute_time;
	_numIterations++;
}


template <typename T>
T Context<T>::getResidual( int idx ) const
{
	if ( idx <= getNumComputeObjects() )
		return _computeObjectList[idx]->getResidual();
	else
		return 1.0;
}


template <typename T>
int Context<T>::isConverged() const
{
	// If we are computing order parameter self-consistently, use it to
	// determine convergence.
	if ( getComputeOrderParameter() )
	{
		if ( _acc )
			return _acc->isConverged();
		else
			return getComputeOrderParameter()->isConverged();
	}
	else
	// If no order parameter, use first compute object to determine convergence
	{
		return getComputeObject(0)->isConverged();
	}
}
