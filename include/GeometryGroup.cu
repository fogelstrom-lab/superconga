#include "GeometryGroup.h"
#include "GeometryGroup_Kernel.cu"

template <typename T>
GeometryGroup<T>::GeometryGroup() :
	_domainLabels(0), _normals(0), _boundaryDistance(0), _domain_ref(0)
{
	setGeomOpMatrix();
}


template <typename T>
GeometryGroup<T>::GeometryGroup( const GeometryGroup<T>& other ) :
	ContextModule<T>(other),
	_domainLabels(0), _normals(0), _boundaryDistance(0), _domain_ref(0)
{
	if ( other._domainLabels )		_domainLabels = new Grid<char>::GPU( *other._domainLabels );
	if ( other._normals )			_normals = new GridGPU( *other._normals );
	if ( other._boundaryDistance )	_boundaryDistance = new GridGPU( *other._boundaryDistance );
	if ( other._domain_ref )		_domain_ref = new Grid<char>::GPU( *other._domain_ref );

	const int Nc = other.getNumGeometryComponents();

	if ( Nc > 0 )
	{
		//_geomComponentList = new GeometryComponent<T>*[_numGeomComponents];
		for ( int i=0; i<Nc; ++i )
		{
			_geomComponentList.push_back( other._geomComponentList[i]->clone() );
			_geomOpList.push_back( other._geomOpList[i] );
		}
	}

	setGeomOpMatrix();
}


template <typename T>
GeometryGroup<T>::GeometryGroup( Context<T>* ctx ) :
	ContextModule<T>(ctx),
	_domainLabels(0), _normals(0), _boundaryDistance(0), _domain_ref(0), _geomOpList(0)
{
	std::cout << "GeometryGroup<T>::GeometryGroup( Context<T>* )\n";

	ctx->set( this );
	
	// Set operation matrices
	setGeomOpMatrix();
}


template <typename T>
GeometryGroup<T>::~GeometryGroup()
{
	std::cout << "~GeometryGroup()\n";
	
	if ( _domainLabels ) delete _domainLabels;
	if ( _normals ) delete _normals;
	if ( _boundaryDistance ) delete _boundaryDistance;
	if ( _domain_ref ) delete _domain_ref;
	if ( _geomOpAdd ) delete _geomOpAdd;
	if ( _geomOpRemove ) delete _geomOpRemove;
	
	for ( int i=0; i<getNumGeometryComponents(); ++i )
		delete _geomComponentList[i];
}


template <typename T>
GeometryGroup<T>& GeometryGroup<T>::operator=( GeometryGroup<T> other )
{
	swap(&this,other);
	return *this;
}


template <typename T>
void swap( GeometryGroup<T>& A, GeometryGroup<T>& B )
{
	std::swap(A._geomComponentList, B._geomComponentList);
	std::swap(A._geomOpList, B._geomOpList );
	std::swap(A._geomOpAdd, B._geomOpAdd );
	std::swap(A._geomOpRemove, B._geomOpRemove );
	std::swap(A._currentAngle, B._currentAngle );
	std::swap(A._domainLabels, B._domainLabels );
	std::swap(A._normals, B._normals );
	std::swap(A._boundaryDistance, B._boundaryDistance );
	std::swap(A._domain_ref, B._domain_ref );
}


template <typename T>
int GeometryGroup<T>::initialize()
{
	std::cout << "GeometryGroup<T>::initialize()\n";
	int N = ContextModule<T>::getParameters()->getGridResolutionDeltaRotated();
	
	// Initialize grids with data for ONE angle
	GridGPU::initializeGrid( _normals, N, N, 2, Type::real );
	GridGPU::initializeGrid( _boundaryDistance, N, N, 1, Type::real );
	Grid<char>::GPU::initializeGrid( _domainLabels, N, N, 1, Type::real );

	if ( _domain_ref )
		delete _domain_ref;

	_domain_ref = createDomain(N);

	return 0;
}


template <typename T>
void GeometryGroup<T>::setGeomOpMatrix()
{
	//std::cout << "GeometryGroup<T>::setGeomOpMatrix()\n";
	char op_add[16] 	= {	0,1,2,3,
							1,1,1,1,
							2,1,2,1,
							3,1,1,3	};

	char op_remove[16]	= {	0,0,0,0,
							1,0,3,2,
							2,0,0,2,
							3,0,3,0	};
	
	_geomOpAdd = new Grid<char>::GPU(16,1,1,Type::real);
	_geomOpRemove = new Grid<char>::GPU(16,1,1,Type::real);
	
	_geomOpAdd->setData( op_add, 16 );
	_geomOpRemove->setData( op_remove, 16 );
}


template <typename T>
void GeometryGroup<T>::create( const T angle )
{
	_currentAngle = angle;

	// Initialize compound geom grids
	_domainLabels->setZero();
	_normals->setZero();
	_boundaryDistance->setZero();
		
	GeometryComponent<T>* tmpGeom;
	
	// Perform operations for each element
	for ( int i=0; i<getNumGeometryComponents(); ++i )
	{
		tmpGeom = _geomComponentList[i];
		tmpGeom->create( angle );
		
		if ( _geomOpList[i] == Type::add )
			geomOpAdd( tmpGeom );

		if ( _geomOpList[i] == Type::remove )
			geomOpRemove( tmpGeom );
	}
}


template <typename T>
Grid<char>::GPU* GeometryGroup<T>::createDomain( const int N )
{
	const int N_r = ContextModule<T>::getParameters()->calcGridResolutionRotated(N);

	// Initialize compound geom grids
	Grid<char>::GPU* dom_r = new Grid<char>::GPU(N_r,N_r,1,Type::real);
	dom_r->setZero();

	// Perform operations for each element
	for ( int i=0; i<getNumGeometryComponents(); ++i )
	{
		Grid<char>::GPU* dom_local = _geomComponentList[i]->createDomain( N_r );
		
		if ( _geomOpList[i] == Type::add )
			geomOpAdd( dom_r, dom_local );

		if ( _geomOpList[i] == Type::remove )
			geomOpRemove( dom_r, dom_local );

		delete dom_local;
	}

	Grid<char>::GPU* dom = new Grid<char>::GPU(N,N,1,Type::real);
	dom->setZero();
	dom->insertGrid(dom_r);

	delete dom_r;

	return dom;
}

template <typename T>
int GeometryGroup<T>::add( GeometryComponent<T>* newComponent, Type::type operation )
{
	// Add new Geometry component
	//
	newComponent->setParameters( ContextModule<T>::getParameters() );

	_geomComponentList.push_back( newComponent );
	
	// Set geometry operation
	_geomOpList.push_back( int(operation) );

	return getNumGeometryComponents();
}



template <typename T>
void GeometryGroup<T>::set( GeometryComponent<T>* newComponent, Type::type operation, int geomIdx )
{
	newComponent->setParameters( ContextModule<T>::getParameters() );

	if ( geomIdx >= 0 && geomIdx < getNumGeometryComponents() )
	{
		delete _geomComponentList[geomIdx];
		
		_geomComponentList[geomIdx] = newComponent;
		_geomOpList[geomIdx] = int(operation);
	}
	else
	{
		std::cout << "GeometryGroup<T>::setGeometryComponent(): Nothing added, index out of range\n";
	}
}


template <typename T>
GeometryComponent<T>* GeometryGroup<T>::getGeometryComponent( const int comp ) const
{
	int compLim = min(getNumGeometryComponents()-1,comp);
	compLim = max(0,compLim);
	
	return _geomComponentList[compLim];
}


template <typename T>
void GeometryGroup<T>::geomOp( GeometryComponent<T>* newGeom, Grid<char>::GPU* geomOpMatrix )
{
	int Nx, Ny;
	getGridDimensions( Nx, Ny );

	dim3 bpg, tpb;
	_domainLabels->getKernelDimensions(bpg,tpb);
	
	geomOp_kernel<T><<<bpg,tpb>>>(
		getDomainLabels(), getNormals(0), getNormals(1), getBoundaryDistance(),
		newGeom->getDomainLabels(), newGeom->getNormals(0), newGeom->getNormals(1),
		newGeom->getBoundaryDistance(), geomOpMatrix->getDataPointer(),
		Nx, Ny );
}


template <typename T>
void GeometryGroup<T>::geomOpAdd( GeometryComponent<T> * newGeom )
{
	geomOp( newGeom, _geomOpAdd );
}


template <typename T>
void GeometryGroup<T>::geomOpRemove( GeometryComponent<T>* newGeom )
{
//	geomOp( newGeom, _geomOpRemove );
	int Nx, Ny;
	getGridDimensions( Nx, Ny );

	dim3 bpg, tpb;
	_domainLabels->getKernelDimensions(bpg,tpb);

	Grid<char>::GPU* domResult = _domainLabels;//getDomainLabelsGrid();

	Grid<char>::GPU* domTmp(0);
	Grid<char>::GPU::initializeGrid( domTmp, domResult );
	domTmp->setZero();
	
	geomOpRemove_kernel<T><<<bpg,tpb>>>(
		getDomainLabels(), getNormals(0), getNormals(1), getBoundaryDistance(),
		newGeom->getDomainLabels(), newGeom->getNormals(0), newGeom->getNormals(1), newGeom->getBoundaryDistance(),
		domTmp->getDataPointer(),
		_geomOpRemove->getDataPointer(),
		Nx, Ny );		
	
	_domainLabels = domTmp;
	delete domResult;
}


template <typename T>
void GeometryGroup<T>::geomOpAdd( Grid<char>::GPU* dom, Grid<char>::GPU* dom_local ) const
{
	int N = dom->dimX();

	dim3 bpg, tpb;
	Grid<T>::GPU::getKernelDimensions(bpg,tpb,N,N);
	
	geomOpAdd_kernel<T><<<bpg,tpb>>>(
		dom->getDataPointer(), 
		dom_local->getDataPointer(), 
		N );
}


template <typename T>
void GeometryGroup<T>::geomOpRemove(  Grid<char>::GPU* dom, Grid<char>::GPU* dom_local ) const
{
	int N = dom->dimX();

	dim3 bpg, tpb;
	Grid<T>::GPU::getKernelDimensions(bpg,tpb,N,N);

	geomOpRemove_kernel<T><<<bpg,tpb>>>(
		dom->getDataPointer(), 
		dom_local->getDataPointer(), 
		N );
}



template <typename T>
T GeometryGroup<T>::area( int discrete )
{
	create(0.0);

	typename Vector<char>::CPU data_cpu = *(_domainLabels->getVector());
	char* data_ptr = thrust::raw_pointer_cast(data_cpu.data());
	
	int area = 0, border = 0;
	char node;

	for ( int i=0; i<data_cpu.size(); ++i )
	{
		node = data_ptr[i];
		
		if ( node == 1 )
			area++;
		if ( node == 2 || node == 3 )
			border++;
	}			
	
	T h = ContextModule<T>::getParameters()->getGridElementSize();

	if ( discrete )
		h = 1.0;
		
	return ((T)area + (T)border * 1.0) * h*h;
//	return ((T)area) * h*h;
}


// Compute field aveage, only accounting for nodes inside geometry
template <typename T>
cplx<T> GeometryGroup<T>::fieldAverage( GridGPU* data, const int field )
{
	create(0.0);
	
//	_domainLabels->copyDataToCPU();
//	data->copyDataToCPU();

	typename Vector<T>::CPU vec_cpu = *(_domainLabels->getVector());
	typename Grid<T>::CPU data_cpu = *data;

	const int complex = data->isComplex();
	
	// Pointers
	T *data_re = data_cpu->getDataPointer( field, Type::real );	
	T *data_im = data_cpu->getDataPointer( field, Type::imag );	
	
	char *dom = thrust::raw_pointer_cast(vec_cpu.data());

	// Dimensions
	int Nx_data = data->dimX();
	int Ny_data = data->dimY();
	
	int Nx_dom = _domainLabels->dimX();
	int Ny_dom = _domainLabels->dimY();

	int domOffsetX = (Nx_dom-Nx_data)/2;
	int domOffsetY = (Ny_dom-Ny_data)/2;

	cplx<T> sum(0.0, 0.0);

	int idx_data, idx_dom, num_discrete = 0;
	
	for ( int j=0; j<Ny_data; ++j )
	{
		for ( int i=0; i<Nx_data; ++i )
		{
			idx_data = j*Nx_data + i;
			idx_dom = (j+domOffsetY)*Nx_dom + (i+domOffsetX);
	
			if ( dom[idx_dom] == 1 )
			{
				sum.x += data_re[idx_data];
				if ( complex )
					sum.y += data_im[idx_data];
				num_discrete++;
			}
		}
	}

	return sum / (T)num_discrete;
}


template <typename T>
void GeometryGroup<T>::countBoundariesAlongTrajectory( int Nx, int Ny,
	char* dom_ptr, int* idx_in_ptr, int* idx_out_ptr, int* numBndIsect_ptr, int& sum_idx_in, int& sum_idx_out )
{
	int i, j, idx;
	
	int *num_in = new int[Nx];
	int *num_out = new int[Nx];
	
	// Initialize entry/exit counters for current angle
	for ( i=0; i<Nx; ++i )
	{
		num_in[i] = 0;
		num_out[i] = 0;
	}

	// Trace along trajectories to find entry/exit points
	for ( i=0; i<Nx; ++i )
	{
		for ( j=0; j<Ny; ++j )
		{
			idx = j*Nx + i;

			if ( dom_ptr[idx] == 2 )
				num_in[i] += 1;
				
			if ( dom_ptr[idx] == 3 )
				num_out[i] += 1;
		}

		// Number of times this trajectory terminates at a boundary (for reflection)
		numBndIsect_ptr[i] = num_out[i];
	}

	// Calculate indices to sparse storage
	for ( i=0; i<Nx; ++i )
	{
		if ( num_in[i] == 0 )
			idx_in_ptr[i] = -1;
		else
			idx_in_ptr[i] = sum_idx_in;
		
		sum_idx_in += num_in[i];

		if ( num_out[i] == 0 )
			idx_out_ptr[i] = -1;
		else
			idx_out_ptr[i] = sum_idx_out;
		
		sum_idx_out += num_out[i];
	}
	
	delete[] num_in;
	delete[] num_out;
}


template <typename T>
void GeometryGroup<T>::countBoundariesAlongNegativeTrajectory( int Nx, int Ny,
	char* dom_ptr, int* idx_in_ptr, int* idx_out_ptr, int* numBndIsect_ptr, int& sum_idx_in, int& sum_idx_out )
{
	int i, j, idx;
	
	int *num_in = new int[Nx];
	int *num_out = new int[Nx];
	
	// Initialize entry/exit counters for current angle
	for ( i=0; i<Nx; ++i )
	{
		num_in[i] = 0;
		num_out[i] = 0;
	}

	// Trace along trajectories to find entry/exit points
	for ( i=0; i<Nx; ++i )
	{
		for ( j=Ny-1; j>=0; --j )
		{
			idx = j*Nx + i;

			if ( dom_ptr[idx] == 3 )
				num_in[i] += 1;
				
			if ( dom_ptr[idx] == 2 )
				num_out[i] += 1;
		}

		numBndIsect_ptr[i] = num_out[i];
	}

	// Calculate indices to sparse storage
	for ( i=0; i<Nx; ++i )
	{
		if ( num_in[i] == 0 )
			idx_in_ptr[i] = -1;
		else
			idx_in_ptr[i] = sum_idx_in;
		
		sum_idx_in += num_in[i];

		if ( num_out[i] == 0 )
			idx_out_ptr[i] = -1;
		else
			idx_out_ptr[i] = sum_idx_out;
		
		sum_idx_out += num_out[i];
	}
	
	delete[] num_in;
	delete[] num_out;
}


template <typename T>
void GeometryGroup<T>::findBoundaryCrossingsAlongTrajectory( int Nx, int Ny, char *dom_p, int *& coord_ptr )
{
	int i, j, idx;

	for ( i=0; i<Nx; ++i )
	{
		for ( j=0; j<Ny; ++j )
		{
			idx = j*Nx + i;

			if ( dom_p[idx] == 3 )
			{
				*coord_ptr = j;
				coord_ptr++;
			}
		}
	}
}


template <typename T>
void GeometryGroup<T>::findBoundaryCrossingsAlongNegativeTrajectory( int Nx, int Ny, char *dom_p, int *& coord_ptr )
{
	int i, j, idx;

	for ( i=0; i<Nx; ++i )
	{
		for ( j=Ny-1; j>=0; --j )
		{
			idx = j*Nx + i;

			if ( dom_p[idx] == 2 )
			{
				*coord_ptr = j;
				coord_ptr++;
			}
		}
	}
}




template <typename T>
void GeometryGroup<T>::extendValuesAtBoundary( GridGPU* grid, const int numExtend ) const
{
	char *dom_ptr = getDomainLabels();

	int Nx_geom, Ny_geom, Nx_target, Ny_target, Nx_geom_offset, Ny_geom_offset;

	getGridDimensions( Nx_geom, Ny_geom );
	
	Nx_target = grid->dimX();
	Ny_target = grid->dimY();
	
	Nx_geom_offset = (Nx_geom - Nx_target) / 2;
	Ny_geom_offset = (Ny_geom - Ny_target) / 2;
	
	int Nf = grid->numFields();
	
	// Set kernel dimensions
	dim3 bpg, tpb;

	grid->getKernelDimensionsFields( bpg, tpb );

	if ( grid->isComplex() )
		extendValuesAtBoundary_kernel<T><<<bpg,tpb>>>(
			dom_ptr,
			grid->getDataPointer( 0, Type::real ),
			grid->getDataPointer( 0, Type::imag ),
			Nx_geom, Ny_geom,
			Nx_target, Ny_target,
			Nx_geom_offset, Ny_geom_offset,
			numExtend );
	else
		extendValuesAtBoundary_kernel<T><<<bpg,tpb>>>(
			dom_ptr,
			grid->getDataPointer( 0, Type::real ),
			Nx_geom, Ny_geom,
			Nx_target, Ny_target,
			Nx_geom_offset, Ny_geom_offset,
			numExtend );

}


template <typename T>
void GeometryGroup<T>::zeroValuesOutsideDomain( GridGPU* target )
{
	if ( getCurrentAngle() > (T)0.0 || getCurrentAngle() < (T)0.0 )
		create(0.0);
	
	char *data_dom = _domainLabels->getDataPointer( 0, Type::real );
	
	int Nx_dom = _domainLabels->dimX();
	int Ny_dom = _domainLabels->dimY();
	
	int Nx_img = target->dimX();
	int Ny_img = target->dimY();

	int Nx_diff = (Nx_dom - Nx_img)/2;
	int Ny_diff = (Ny_dom - Ny_img)/2;
	
	dim3 grid, block;
	target->getKernelDimensionsFields( grid, block );
		
	zeroValuesOutsideDomain_kernel<<<grid,block>>>(
		_domainLabels->getDataPointer( 0, Type::real ),
		target->getDataPointer( 0, Type::real ),
		Nx_dom, Ny_dom, Nx_img, Ny_img, Nx_diff, Ny_diff, int(target->isComplex()) );
		
	if ( target->isComplex() )
		zeroValuesOutsideDomain_kernel<<<grid,block>>>(
			_domainLabels->getDataPointer( 0, Type::real ),
			target->getDataPointer( 0, Type::imag ),
			Nx_dom, Ny_dom, Nx_img, Ny_img, Nx_diff, Ny_diff, int(target->isComplex()) );
}


template <typename T>
void GeometryGroup<T>::getGridDimensions( int & Nx, int & Ny ) const
{
	Nx = _domainLabels->dimX();
	Ny = _domainLabels->dimY();
}


template <typename T>
int GeometryGroup<T>::getNumAngles() const
{
	return ContextModule<T>::getParameters()->getAngularResolution();
}


template <typename T>
char* GeometryGroup<T>::getDomainLabels() const
{
	return _domainLabels->getDataPointer();
}


template <typename T>
T* GeometryGroup<T>::getNormals( const int dim ) const 
{
	return _normals->getDataPointer( dim );
}


template <typename T>
T* GeometryGroup<T>::getBoundaryDistance() const
{
	return _boundaryDistance->getDataPointer();
}
