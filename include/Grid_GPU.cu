#include "Grid_GPU_kernel.cu"

///////////////////////////////////////////
// Code path for GPU (device) specific code
///////////////////////////////////////////

template<typename,typename> class GridData;

template <typename T>
struct GridFunctions<T,thrust::device_vector<T> >
{
	typedef GridData<T,thrust::device_vector<T> > GridGPU;
	typedef thrust::device_vector<T> Vector;

	GridGPU* 	grid;

	GridFunctions( GridGPU* ptr ) : grid(ptr) {}

	template <typename T2>
	void	typeConvert( GridData<T2,thrust::device_vector<T2> >& other ) const;

	// IN place transponate
	GridGPU&	transpose(const int field);

	// Compute the complex magnitude squared, of specified field.
	void		complexMagnitudeSq( GridGPU* result ) const;

	// Compute complex phase of this (complex) grid.
	void		complexPhase( GridGPU* result ) const;

	// Compute gradient of complex phase of this grid.
	void		complexPhaseGradient( GridGPU* result, const T h ) const;

	// Compute the magnitude of the real part of this vector field.
	void		fieldMagnitude( GridGPU* result ) const;

	// Compute divergence
	void 		div( GridGPU* result, const T h ) const;

	// Compute curl
	void 		curl( GridGPU* result, const T h ) const;

	// Compute curl
	void 		del2( GridGPU* result, const T h ) const;

	// Sum all fields.
	void		sumFieldComponents( GridGPU* result ) const;

	// Insert other grid into this. <x,y>_offset are offset in discrete elements from center.
	// x_offset = y_offset = 0 results in a centered grid.
	void		insertGrid(	const GridGPU* other, const int x_offset, const int y_offset );

	// Insert other grid into this, with applied rotation (to other grid).
	void		insertRotatedGrid( const GridGPU* other, const T phi,
					const int x_offset, const int y_offset, Type::type part );

	// Add other grid to this, with applied rotation (to other grid).
	void		insertRotatedGridAdditive( const GridGPU* other, const T phi, int field,
					const int x_offset, const int y_offset, Type::type part );

	// Change resolution of grid, with linear interpolation of existing data.
	void		resample( const int nx_new, const int ny_new );
	void		resample( GridGPU*& target, const int nx_new, const int ny_new ) const;

	// Find min and max value of range
	static void	minmax( const GridGPU& grid, T& minv, T& maxv, int field, Type::type part );

	// Helper functions to create kernel dimensions
	void		getKernelDimensions( dim3& blocksPerGrid, dim3& threadsPerBlock ) const;
	void		getKernelDimensionsFields( dim3& blocksPerGrid, dim3& threadsPerBlock ) const;
	static void	getKernelDimensions( dim3& bpg, dim3& tpb, int nx, int ny, int nf=1 );

	// Check for latest error
	static int	checkCudaError( std::string msg = "" );
};


template <typename T>
template <typename T2>
void GridFunctions<T,thrust::device_vector<T> >::typeConvert( GridData<T2,thrust::device_vector<T2> >& other ) const
{
	dim3 tpb, bpg;
	getKernelDimensionsFields( bpg, tpb );

	int complex = int(grid->isComplex());
	
	typeConvert_kernel<T,T2><<<bpg, tpb>>>(
		grid->getDataPointer( 0, Type::real ),
		other.getDataPointer( 0, Type::real ),
		grid->_dimX, grid->_dimY, complex );

	if ( complex )
		typeConvert_kernel<T,T2><<<bpg, tpb>>>(
			grid->getDataPointer( 0, Type::imag ),
			other.getDataPointer( 0, Type::imag ),
			grid->_dimX, grid->_dimY, complex );
}


template <typename T>
GridData<T,thrust::device_vector<T> >& GridFunctions<T,thrust::device_vector<T> >::transpose(const int field)
{
	if ( grid->_dimX != grid->_dimY )
	{
		std::cout << "GridFunctions<T,thrust::device_vector<T> >::transpose(): Error, dimensions not equal\n";
		return *grid;
	}

	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );

	int f_start = 0, f_end = grid->_numFields-1;

	if ( field != -1 )
	{
		f_start = field;
		f_end = field;
	}

	for ( int i=f_start; i<=f_end; ++i )
	{
		transpose_kernel<<<bpg,tpb>>>( grid->getDataPointer(i,Type::real), grid->_dimX );
		if ( grid->isComplex() )
			transpose_kernel<<<bpg,tpb>>>( grid->getDataPointer(i,Type::imag), grid->_dimX );
	}

	return *grid;
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::complexMagnitudeSq( GridGPU* result ) const
{
	dim3 tpb, bpg;
	getKernelDimensionsFields( bpg, tpb );
	
	complexMagnitudeSq_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		grid->getDataPointer( 0, Type::imag ),
		result->getDataPointer( 0, Type::real ),
		grid->_dimX, grid->_dimY );

	checkCudaError("GridData::complexMagnitudeSq()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::complexPhase( GridGPU* result ) const
{
	dim3 tpb, bpg;
	getKernelDimensionsFields( bpg, tpb );

	complexPhase_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		grid->getDataPointer( 0, Type::imag ),
		result->getDataPointer( 0, Type::real ),
		grid->_dimX, grid->_dimY );
		
	checkCudaError("GridData::complexPhase()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::complexPhaseGradient( GridGPU* result, const T h ) const
{
	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );

	complexPhaseGradient_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		grid->getDataPointer( 0, Type::imag ),
		result->getDataPointer( 0, Type::real ),
		result->getDataPointer( 1, Type::real ),
		h, grid->_dimX, grid->_dimY );
	
	checkCudaError("GridData<T>::complexPhaseGradient()");	
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::fieldMagnitude( GridGPU* result ) const
{
	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );
	
	fieldMagnitude_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		result->getDataPointer( 0, Type::real ),
		grid->_dimX, grid->_dimY, grid->_numFields, int(grid->isComplex()) );
		
	checkCudaError("GridData<T>::fieldMagnitude()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::div( GridGPU* result, const T h ) const
{
	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );
	
	div_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		grid->getDataPointer( 1, Type::real ),
		result->getDataPointer( 0, Type::real ),
		h, grid->dimX(), grid->dimY() );

	checkCudaError("GridData<T>::div()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::curl( GridGPU* result, const T h ) const
{
	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );
	
	curl_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		grid->getDataPointer( 1, Type::real ),
		result->getDataPointer( 0, Type::real ),
		h, grid->dimX(), grid->dimY() );
		
	checkCudaError("GridData<T>::curl()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::del2( GridGPU* result, const T h ) const
{
	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );

	const T inv_h2 = 1.0/(h*h);
	
	for ( int c=0; c<result->numFields(); ++c)
	{
		del2_kernel<<<bpg, tpb>>>( 
			result->getDataPointer( c, Type::real ),
			grid->getDataPointer( c, Type::real ),
			inv_h2, grid->dimX(), grid->dimY() );
	}

	checkCudaError("GridData<T>::del2()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::sumFieldComponents( GridGPU* result ) const
{
	dim3 tpb, bpg;
	getKernelDimensions( bpg, tpb );

	sumFieldComponents_kernel<<<bpg, tpb>>>( 
		grid->getDataPointer( 0, Type::real ),
		result->getDataPointer( 0, Type::real ),
		grid->_dimX, grid->_dimY, grid->_numFields, int(grid->isComplex()) );
	
	if ( grid->isComplex() )
		sumFieldComponents_kernel<<<bpg, tpb>>>( 
			grid->getDataPointer( 0, Type::imag ),
			result->getDataPointer( 0, Type::imag ),
			grid->_dimX, grid->_dimY, grid->_numFields, int(grid->isComplex()) );

	checkCudaError("GridData<T>::sumFieldComponents()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::insertGrid(
	const GridGPU* other, const int x_offset, const int y_offset )
{
	// insert 'other' grid into 'this'
	const int Nx_other = other->dimX();
	const int Ny_other = other->dimY();
	const int Nf_other = other->numFields();

	int x_start_this=0, y_start_this=0, x_start_other=0, y_start_other=0, tpb, bpg;

	int x_diff = (grid->_dimX - Nx_other)/2;
	int y_diff = (grid->_dimY - Ny_other)/2;
	
	if ( x_diff < 0 )	// Other grid bigger than this
	{
		x_start_other = -x_diff + x_offset;
		tpb = grid->_dimX;
	} else				// Other grid smaller (or equal) than this
	{
		x_start_this = x_diff + x_offset;
		tpb = Nx_other;
	}

	if ( y_diff < 0 )
	{
		y_start_other = -y_diff + y_offset;
		bpg = grid->_dimY;
	} else
	{
		y_start_this = y_diff + y_offset;
		bpg = Ny_other;
	}
	
	dim3 threadsPerBlock, blocksPerGrid;
	getKernelDimensions( blocksPerGrid, threadsPerBlock, tpb, bpg );

	for ( int c=0; c<grid->_numFields; ++c )
	{
		int c_other = c >= Nf_other ? Nf_other-1 : c;

		insertGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>(
			grid->getDataPointer( c, Type::real ),
			other->getDataPointer( c_other, Type::real ),
			grid->_dimX, grid->_dimY,
			Nx_other, Ny_other,
			x_start_this, y_start_this,
			x_start_other, y_start_other );
		
		if ( grid->isComplex() )
		{
			insertGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>(
				grid->getDataPointer( c, Type::imag ),
				other->getDataPointer( c_other, Type::imag ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				x_start_this, y_start_this,
				x_start_other, y_start_other );
		}
	}

	checkCudaError("GridData<T>::insertGrid()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::insertRotatedGrid(
	const GridGPU* other, const T phi, const int x_offset, const int y_offset, Type::type part )
{
	checkCudaError("GridData<T>::insertRotatedGrid() b4 ");

	// Grid dimensions of grid to rotate and read from (lookup grid)
	const int Nx_other = other->dimX();
	const int Ny_other = other->dimY();

	const bool complex_this = grid->isComplex();
	const bool complex_other = other->isComplex();

	// Kernel launch
	dim3 threadsPerBlock, blocksPerGrid;
	getKernelDimensions( blocksPerGrid, threadsPerBlock );
	
	for ( int c=0; c<grid->_numFields; ++c )
	{
		int field_other = c;
		
		if ( other->numFields() <= c )
			field_other = 0;	

		if ( complex_this && complex_other )
		{
			insertRotatedGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::real ),
				other->getDataPointer( field_other, Type::real ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );

			insertRotatedGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::imag ),
				other->getDataPointer( field_other, Type::imag ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );			
		}
		else if ( !complex_this && !complex_other )
		{
			insertRotatedGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::real ),
				other->getDataPointer( field_other, Type::real ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );
		}
		else if ( !complex_this )
		{
			insertRotatedGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::real ),
				other->getDataPointer( field_other, part ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );		
		}
		else
		{
			insertRotatedGrid_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, part ),
				other->getDataPointer( field_other, Type::real ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );		
		}
	}
	
	checkCudaError("GridData<T>::insertRotatedGrid()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::insertRotatedGridAdditive(
	const GridGPU* other, const T phi, int field, const int x_offset, const int y_offset, Type::type part )
{
	int fieldStart, fieldEnd;

	if ( field == -1 )
	{
		fieldStart = 0;
		fieldEnd = grid->_numFields;
	} else if ( field < grid->_numFields ) {
		fieldStart = field;
		fieldEnd = field + 1;	
	} else {
		std::cout << "GridData() : Field out of bounds, operation terminated.\n";
		return;
	}
	
	// Grid dimensions of grid to rotate and read from (lookup grid)
	const int Nx_other = other->dimX();
	const int Ny_other = other->dimY();

	const bool complex_this = grid->isComplex();
	const bool complex_other = other->isComplex();

	// Kernel launch
	dim3 threadsPerBlock, blocksPerGrid;
	getKernelDimensions( blocksPerGrid, threadsPerBlock );

	for ( int c=fieldStart; c<fieldEnd; ++c )
	{
		int field_other = c;
		
		if ( other->numFields() <= c )
			field_other = 0; // Maybe put this to other->numFields()-1 ??
		
		if ( complex_this && complex_other )
		{
			insertRotatedGridAdditive_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::real ),
				other->getDataPointer( field_other, Type::real ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );

			insertRotatedGridAdditive_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::imag ),
				other->getDataPointer( field_other, Type::imag ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );			
		}
		else if ( !complex_this && !complex_other )
		{
			insertRotatedGridAdditive_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::real ),
				other->getDataPointer( field_other, Type::real ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );
		}
		else if ( !complex_this )
		{
			insertRotatedGridAdditive_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, Type::real ),
				other->getDataPointer( field_other, part ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );		
		}
		else
		{
			insertRotatedGridAdditive_kernel<<<blocksPerGrid,threadsPerBlock>>>( 
				grid->getDataPointer( c, part ),
				other->getDataPointer( field_other, Type::real ),
				grid->_dimX, grid->_dimY,
				Nx_other, Ny_other,
				phi, x_offset, y_offset );		
		}
	}
	
	checkCudaError("GridData<T>::insertRotatedGridAdditive()");
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::resample( const int nx_new, const int ny_new )
{
	const int isComplex = int(grid->isComplex());
	const int numFields = grid->numFields();
	const int numElements = nx_new * ny_new * numFields * (1+isComplex);

	// Allocate vector for new grid
	Vector* vec_tmp = new Vector(numElements);

	const int nx = grid->dimX();
	const int ny = grid->dimY();

	T* data_tmp = thrust::raw_pointer_cast(vec_tmp->data());
	
	// Kernel launch
	dim3 threadsPerBlock, blocksPerGrid;
	getKernelDimensions( blocksPerGrid, threadsPerBlock, nx_new, ny_new );
	
	for ( int c=0; c<numFields; ++c )
	{
		int fieldOffset = (nx_new * ny_new * (1+isComplex)) * c;
		
		resample_kernel<<<blocksPerGrid,threadsPerBlock>>>(
			grid->getDataPointer( c, Type::real ),
			data_tmp + fieldOffset,
			grid->_dimX, grid->_dimY,
			nx_new, ny_new );
			
		if ( isComplex )
		{
			resample_kernel<<<blocksPerGrid,threadsPerBlock>>>(
				grid->getDataPointer( c, Type::imag ),
				data_tmp + fieldOffset + (nx_new * ny_new),
				grid->_dimX, grid->_dimY,
				nx_new, ny_new );
		}
	}
	
	// Free old vector and reassign
	delete grid->_data;
	grid->_data = vec_tmp;

	// Change affected class member variables
	grid->_dimX = nx_new;
	grid->_dimY = ny_new;
	
	grid->_allocatedMemTotal -= nx*ny*numFields*(1+isComplex)*sizeof(T);
	grid->_allocatedMemTotal += numElements*sizeof(T);
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::resample( GridGPU*& target, const int nx_new, const int ny_new ) const
{
	GridGPU::initializeGrid(target, nx_new, ny_new, grid->numFields(), grid->representation());

	const int isComplex = int(grid->isComplex());
	const int numFields = grid->numFields();

	const int nx = grid->dimX();
	const int ny = grid->dimY();

	
	// Kernel launch
	dim3 threadsPerBlock, blocksPerGrid;
	getKernelDimensions( blocksPerGrid, threadsPerBlock, nx_new, ny_new );
	
	for ( int c=0; c<numFields; ++c )
	{
		resample_kernel<<<blocksPerGrid,threadsPerBlock>>>(
			grid->getDataPointer( c, Type::real ),
			target->getDataPointer( c, Type::real ),
			nx, ny,
			nx_new, ny_new );
			
		if ( isComplex )
		{
			resample_kernel<<<blocksPerGrid,threadsPerBlock>>>(
				grid->getDataPointer( c, Type::imag ),
				target->getDataPointer( c, Type::imag ),
				nx, ny,
				nx_new, ny_new );
		}
	}
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::minmax( const GridGPU& grid, T& minv, T& maxv, int field, Type::type part )
{
	Vector& vec = *(grid.getVector());

	size_t start = grid.getFieldComponentOffset( field, part );
	size_t end = start + grid.dimX() * grid.dimY();

	typedef typename thrust::detail::normal_iterator<thrust::device_ptr<T> > iter;

	iter begin = vec.begin();

	thrust::pair<iter,iter> result = thrust::minmax_element( begin+start, begin+end );

	minv = *result.first;
	maxv = *result.second;
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::getKernelDimensions( dim3& blocksPerGrid, dim3& threadsPerBlock ) const
{
	getKernelDimensions( blocksPerGrid, threadsPerBlock, grid->_dimX, grid->_dimY, 1 );
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::getKernelDimensionsFields( dim3& blocksPerGrid, dim3& threadsPerBlock ) const
{
	getKernelDimensions( blocksPerGrid, threadsPerBlock, grid->_dimX, grid->_dimY, grid->_numFields );
}


template <typename T>
void GridFunctions<T,thrust::device_vector<T> >::getKernelDimensions( dim3& blocksPerGrid, dim3& threadsPerBlock, int Nx, int Ny, int Nf )
{
	int tpb_x, bpg_x;
	
	const int threadmax = 64;
	
	if ( Nx <= threadmax )
	{
		tpb_x = Nx;
		bpg_x = 1;
	} else {
		tpb_x = threadmax;
		bpg_x = Nx / threadmax;
	
		if ( tpb_x * bpg_x != Nx )
			bpg_x++;	
	}
	
	threadsPerBlock = dim3(tpb_x,1,1);
	blocksPerGrid = dim3(bpg_x,Ny,Nf);
}


template <typename T>
int GridFunctions<T,thrust::device_vector<T> >::checkCudaError( std::string msg )
{
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess )
	{
		std::cout << msg << ": " << cudaGetErrorString( err ) << std::endl;
		return 1;
	}
	else
		return 0;
}

