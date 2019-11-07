#include "Grid.h"
#include "Grid_functors.h"
#include "Grid_CPU.cpp"
#include "Grid_file.cpp"

#ifdef __CUDACC__
#include "Grid_GPU.cu"
#endif


template <typename T, typename DEV>
size_t GridData<T,DEV>::_allocatedMemTotal = 0;


template <typename T, typename DEV>
GridData<T,DEV>::GridData()
	:	GridFunctions<T,DEV>(this), _dimX(0), _dimY(0), _numFields(0), _data(0)
{}


template <typename T, typename DEV>
GridData<T,DEV>::GridData( const GridData<T,DEV>& other ) :
	GridFunctions<T,DEV>(this),
	_representation(other._representation),
	_dimX(other._dimX),	_dimY(other._dimY),
	_numFields(other._numFields),
	_data(0)
{
	if ( other._data )
		_data = new DEV(*other._data);

	_allocatedMemTotal += numElements() * sizeof(T);
}


template <typename T, typename DEV>
GridData<T,DEV>::GridData( const GridData<T,DEV>& other, const int numFields ) :
	GridFunctions<T,DEV>(this),
	_representation(other._representation),
	_dimX(other._dimX),	_dimY(other._dimY),
	_numFields(numFields),
	_data(0)
{
	_data = new DEV(numElements());
	_allocatedMemTotal += numElements() * sizeof(T);
}


template <typename T, typename DEV>
GridData<T,DEV>::GridData( int nx, int ny, int numFields, Type::type representation ) :
	GridFunctions<T,DEV>(this),
	_representation(representation),
	_dimX(nx), _dimY(ny),
	_numFields(numFields),
	_data(0)
{
	_data = new DEV( numElements() );
	_allocatedMemTotal += numElements() * sizeof(T);
}


template <typename T, typename DEV>
GridData<T,DEV>::~GridData()
{
	if (_data)
	{
		delete _data;
		_allocatedMemTotal -= numElements() * sizeof(T);
	}
}


template <typename T, typename DEV>
bool GridData<T,DEV>::operator==( const GridData<T,DEV>& right ) const
{
	return (	_dimX == right._dimX &&
				_dimY == right._dimY &&
				_numFields == right._numFields &&
				_representation == right._representation );
}


template <typename T, typename DEV>
bool GridData<T,DEV>::operator!=( const GridData<T,DEV>& right ) const
{
	return !operator==(right);
}


template <typename T, typename DEV>
GridData<T,DEV>& GridData<T,DEV>::operator=( GridData<T,DEV> right )
{
	swap(*this,right);
	return *this;
}


template <typename T, typename DEV>
GridData<T,DEV>& GridData<T,DEV>::operator=( const cplx<T> right )
{
	if ( isComplex() )
	{
		setValue(right.x, Type::real);
		setValue(right.y, Type::imag);
	}
	else
		std::cout << "Warning: Attempted to fill a real valued grid to a complex valued constant! Ignored.\n";

	return *this;
}


template <typename T, typename DEV>
GridData<T,DEV>& GridData<T,DEV>::operator=( const T right )
{
	setValue(right);
	return *this;
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator+=( const GridData<T,DEV>& right )
{
	if ( *this == right )
	{
		thrust::transform( _data->begin(), _data->end(), right._data->begin(), _data->begin(), thrust::plus<T>() );
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator+=( const cplx<T> right )
{
	// This assumes that the calling object is complex, rightwise new memory for the imaginary
	// part would have to be allocated, and it would not be coallesced properly
	if ( isComplex() )
	{
		for ( int c=0; c<numFields(); ++c )
		{
			thrust::for_each(thrust::make_zip_iterator( thrust::make_tuple( _data->begin() + getFieldComponentOffset( c, Type::real, Type::begin ),
																			_data->begin() + getFieldComponentOffset( c, Type::imag, Type::begin ) ) ),
							 thrust::make_zip_iterator( thrust::make_tuple( _data->begin() + getFieldComponentOffset( c, Type::real, Type::end ),
							 												_data->begin() + getFieldComponentOffset( c, Type::imag, Type::end ) ) ),
							 addComplexScalar_functor( right ) );
		}
	}
	else
	{
		std::cout << "Warning: Attempted to add real valued grid with complex valued constant! Ignored.\n";
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator+=( const T right )
{
	if ( !isComplex() )
	{
		thrust::constant_iterator<T> value(right);
		thrust::transform( value, value+_data->size(), _data->begin(), _data->begin(), thrust::plus<T>() );
	}
	else
	{
		cplx<T> temp_cplx;
		temp_cplx.x = right;
		temp_cplx.y = T(0);
		operator+=(temp_cplx);
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator-=( const GridData<T,DEV>& right )
{
	if ( *this == right )
	{
		thrust::transform( _data->begin(), _data->end(), right._data->begin(), _data->begin(), thrust::minus<T>() );
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator-=( const T right )
{
	operator+=(-right);
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator-=( const cplx<T> right )
{
	operator+=(-right);
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator*=( const GridData<T,DEV>& right )
{
	if ( *this == right )
	{
		// Check if both vectors are complex
		if ( isComplex() )
		{ 
			// Complex valued multiplication
			for ( int c=0; c<numFields(); ++c )
			{
				thrust::for_each(thrust::make_zip_iterator( thrust::make_tuple( _data->begin() + getFieldComponentOffset( c, Type::real, Type::begin ),
																				_data->begin() + getFieldComponentOffset( c, Type::imag, Type::begin ),
																				right._data->begin() + right.getFieldComponentOffset( c, Type::real, Type::begin ),
																				right._data->begin() + right.getFieldComponentOffset( c, Type::imag, Type::begin ) ) ),
								 thrust::make_zip_iterator( thrust::make_tuple( _data->begin() + getFieldComponentOffset( c, Type::real, Type::end ),
								 												_data->begin() + getFieldComponentOffset( c, Type::imag, Type::end ),
																				right._data->begin() + right.getFieldComponentOffset( c, Type::real, Type::end ),
																				right._data->begin() + right.getFieldComponentOffset( c, Type::imag, Type::end ) ) ),
								 complexVectorProduct_functor() );
			}
		}
		else
		{
			// Real valued multiplication 
			thrust::transform( _data->begin(), _data->end(), right._data->begin(), _data->begin(), thrust::multiplies<T>() );
		}
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator*=( const cplx<T> right )
{
	// Only callable if this grid is complex valued
	if ( isComplex() )
	{
		for ( int c = 0; c < numFields(); ++c )
		{
			thrust::for_each(thrust::make_zip_iterator( thrust::make_tuple( _data->begin() + getFieldComponentOffset( c, Type::real, Type::begin ),
																			_data->begin() + getFieldComponentOffset( c, Type::imag, Type::begin ) ) ),
							 thrust::make_zip_iterator( thrust::make_tuple( _data->begin() + getFieldComponentOffset( c, Type::real, Type::end ),
							 												_data->begin() + getFieldComponentOffset( c, Type::imag, Type::end ) ) ),
							 scaleComplexVector_functor( right ) );
		}
	}
	else
	{
		std::cout << "Warning: Attempted to multiply real valued grid with complex valued constant! Ignored.\n";
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::operator*=( const T right )
{
	// This will be valid even if the calling object is complex, because re and im scale the same with real scalar
	thrust::constant_iterator<T> value(right);
	thrust::transform( value, value+_data->size(), _data->begin(), _data->begin(), thrust::multiplies<T>() );
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator+( GridData<T,DEV> right ) const
{
	right += *this;
	return right;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator+( const cplx<T> right ) const
{
	GridData<T,DEV> tmp( *this );
	tmp += right;
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator+( const T right ) const
{
	GridData<T,DEV> tmp( *this );
	tmp += right;
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator-( GridData<T,DEV> right ) const
{
	right -= *this;
	return -right;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator-( const cplx<T> right ) const
{
	GridData<T,DEV> tmp( *this );
	tmp -= right;
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator-( const T right ) const
{
	GridData<T,DEV> tmp( *this );
	tmp -= right;
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator*( GridData<T,DEV> right ) const
{
	right *= *this;
	return right;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator*( const cplx<T> right ) const
{
	// Not optimal, but arithmetic operators are typically not performance critical:
	GridData<T,DEV> tmp( *this );	// One read (this) and one write (tmp)
	tmp *= right;					// One read (tmp) and one write (tmp)
	// --> Total two reads and two writes

	// Optimal implentation would be something like:

	// Allocate tmp grid with correct dimensions
	//GridData<T,DEV> tmp( dimX(), dimY(), numFields(), representation() ); // No read or write
	// Then:
	//"some op that will read data from *this and multiplies with right,
	// then writes to tmp. All in one go." // One read (this) and one write (tmp)
	// --> Total one read and one write of an entire grid.

	// The above comment goes for all arithmetic operators starting with a call
	// to the copy constructor, i.e.: GridData<T,DEV> tmp( *this );

	// Return result
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator*( const T right ) const
{
	GridData<T,DEV> tmp( *this );
	tmp *= right;
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator-() const
{
	GridData<T,DEV> tmp( *this );
	tmp *= T(-1.0);
	return tmp;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::operator[]( const int field ) const
{
	GridData<T,DEV> tmp( *this, 1 );
	copyFieldTo( &tmp, field, 0 );
	return tmp;
}


template <typename T, typename DEV>
template <typename T2, typename DEV2>
GridData<T,DEV>::operator GridData<T2,DEV2>()
{
	GridData<T2,DEV2> other(_dimX, _dimY, _numFields, _representation);
	*(other.getVector()) = *_data;
	return other;
}


template <typename T, typename DEV>
template <typename T2>
GridData<T,DEV>::operator GridData<T2,thrust::device_vector<T2> >()
{
	GridData<T2,thrust::device_vector<T2> > other(_dimX, _dimY, _numFields, _representation);
	GridFunctions<T,DEV>::typeConvert( other );
	return other;
}


template <typename T, typename DEV>
void swap( GridData<T,DEV>& grid1, GridData<T,DEV>& grid2 )
{
	std::swap( grid1._representation,	grid2._representation );
	std::swap( grid1._dimX,				grid2._dimX );
	std::swap( grid1._dimY,				grid2._dimY );
	std::swap( grid1._numFields,		grid2._numFields );
	std::swap( grid1._data,				grid2._data );
}


template <typename T, typename DEV>
void GridData<T,DEV>::setValue( const T val, Type::type part, const int field )
{
	if ( _data ) // Make sure we don't try to set elements when no memory is allocated
	{
		if ( field == -1 )
		{
			// Set all elements
			if ( part == Type::complex )
				thrust::fill( _data->begin(), _data->end(), val );
			else
			{
				// Set real or imag parts for all fields
				for ( int c=0; c<_numFields; ++c )
				{
					thrust::fill(	_data->begin()+getFieldComponentOffset(c,part,Type::begin),
									_data->begin()+getFieldComponentOffset(c,part,Type::end), val );
				}
			}
		}
		else
		{
			// Set real part, if requested, for specified field
			if ( part == Type::real || part == Type::complex )
				thrust::fill(	_data->begin()+getFieldComponentOffset(field,Type::real,Type::begin),
								_data->begin()+getFieldComponentOffset(field,Type::real,Type::end), val );

			// Set imag part, if requested and if existing, for specified field
			if ( isComplex() && (part == Type::imag || part == Type::complex) )
				thrust::fill(	_data->begin()+getFieldComponentOffset(field,Type::imag,Type::begin),
								_data->begin()+getFieldComponentOffset(field,Type::imag,Type::end), val );
		}
	}
}


template <typename T, typename DEV>
void GridData<T,DEV>::setZero()
{
	setValue(T(0.0));
}


template <typename T, typename DEV>
int GridData<T,DEV>::setData( const T* p_host_data, size_t numElements )
{
	size_t numElements_this = this->numElements();
	
	if ( numElements > numElements_this )
		numElements = numElements_this;

	thrust::host_vector<T> data_tmp(numElements);
	T* data_ptr = thrust::raw_pointer_cast(data_tmp.data());

	for ( int i=0; i<numElements; ++i )
		data_ptr[i] = p_host_data[i];

	*_data = data_tmp;

	return 0;
}


template <typename T, typename DEV>
GridData<T,DEV>& GridData<T,DEV>::conjugate()
{
	if ( isComplex() )
	{
		for (int i=0; i<numFields(); ++i)
		{
			thrust::transform(	_data->begin()+getFieldComponentOffset(i,Type::imag,Type::begin),
								_data->begin()+getFieldComponentOffset(i,Type::imag,Type::end),
								_data->begin()+getFieldComponentOffset(i,Type::imag,Type::begin),
								thrust::negate<T>() );
		}
	}
	return *this;
}


template <typename T, typename DEV>
GridData<T,DEV>& GridData<T,DEV>::transpose(const int field)
{
	return GridFunctions<T,DEV>::transpose(field);
}


template <typename T, typename DEV>
GridData<T,DEV>& GridData<T,DEV>::sqrt()
{
	thrust::for_each_n( _data->begin(), numElements(), GridFunctors::square_root<T>() );
	return *this;
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::abs() const
{
	GridData<T,DEV> result;
	
	if ( isComplex() )
		result = complexMagnitudeSq().sqrt();
	else
	{
		result = *this;
		thrust::for_each_n( result.getVector()->begin(), result.numElements(), GridFunctors::absVal<T>() );
	}

	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::complexMagnitudeSq( GridData<T,DEV>* result ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) && isComplex() )
		GridFunctions<T,DEV>::complexMagnitudeSq( result );
	else
		std::cout << "GridData::complexMagnitudeSq() ERROR: Grids not of equal dimensions, or source field not complex.\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::complexMagnitudeSq() const
{
	GridData<T,DEV> result( _dimX, _dimY, _numFields, Type::real );
	complexMagnitudeSq( &result );
	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::complexPhase( GridData<T,DEV>* result ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) && isComplex() )
		GridFunctions<T,DEV>::complexPhase( result );
	else
		std::cout << "GridData::complexPhase(): ERROR: Grids not of equal dimensions, or source grid not complex\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::complexPhase() const
{
	GridData<T,DEV> result( _dimX, _dimY, _numFields, Type::real );
	complexPhase( &result );
	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::complexPhaseGradient( GridData<T,DEV>* result, const T h ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) && isComplex() && result->numFields() == 2 )
		GridFunctions<T,DEV>::complexPhaseGradient( result, h );
	else
		std::cout << "GridData::complexPhaseGradient(): ERROR: Grids not of equal dimensions, or source grid not complex\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::complexPhaseGradient( const T h ) const
{
	GridData<T,DEV> result( _dimX, _dimY, _numFields*2, Type::real );
	complexPhaseGradient( &result, h );
	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::fieldMagnitude( GridData<T,DEV>* result ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) )
		GridFunctions<T,DEV>::fieldMagnitude( result );
	else
		std::cout << "GridData::fieldMagnitude(): ERROR: Grids not of equal dimensions.\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::fieldMagnitude() const
{
	GridData<T,DEV> result( _dimX, _dimY, 1, Type::real );
	fieldMagnitude( &result );
	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::div( GridData<T,DEV>* result, const T h ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) && _numFields == 2 )
		GridFunctions<T,DEV>::div( result, h );
	else
		std::cout << "GridData::fieldMagnitude(): ERROR: Grids not of equal dimensions.\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::div( const T h ) const
{
	GridData<T,DEV> result( _dimX, _dimY, 1, Type::real );
	result.setZero();
	div( &result, h );
	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::curl( GridData<T,DEV>* result, const T h ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) && _numFields == 2 )
		GridFunctions<T,DEV>::curl( result, h );
	else
		std::cout << "GridData::fieldMagnitude(): ERROR: Grids not of equal dimensions.\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::curl( const T h ) const
{
	GridData<T,DEV> result( _dimX, _dimY, 1, Type::real );
	curl( &result, h );
	return result;
}

template <typename T, typename DEV>
void GridData<T,DEV>::del2( GridData<T,DEV>* result, const T h ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) )
		GridFunctions<T,DEV>::del2( result, h );
	else
		std::cout << "GridData::fieldMagnitude(): ERROR: Grids not of equal dimensions.\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::del2( const T h ) const
{
	GridData<T,DEV> result( _dimX, _dimY, _numFields, Type::real );
	del2( &result, h );
	return result;
}


template <typename T, typename DEV>
void GridData<T,DEV>::sumFieldComponents( GridData<T,DEV>* result ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *result) && isComplex() == result->isComplex() )
		GridFunctions<T,DEV>::sumFieldComponents( result );
	else
		std::cout << "GridData::sumFieldComponents() ERROR: Grids not of equal dimensions.\n";
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::sumFieldComponents() const
{
	GridData<T,DEV> result( *this, 1 );
	sumFieldComponents( &result );
	return result;	
}


template <typename T, typename DEV>
T GridData<T,DEV>::sumField( const int field ) const
{
	T sum = thrust::reduce(	_data->begin()+getFieldComponentOffset(field,Type::real,Type::begin),
							_data->begin()+getFieldComponentOffset(field,Type::real,Type::end),
							(T)0, thrust::plus<T>() );
	return sum;
}


template <typename T, typename DEV>
void GridData<T,DEV>::sumField( T& sum_re, T& sum_im, const int field ) const
{
	sum_re = thrust::reduce(	_data->begin()+getFieldComponentOffset(field,Type::real,Type::begin),
								_data->begin()+getFieldComponentOffset(field,Type::real,Type::end),
								(T)0, thrust::plus<T>() );

	if ( isComplex() )
	{
		sum_im = thrust::reduce(	_data->begin()+getFieldComponentOffset(field,Type::imag,Type::begin),
									_data->begin()+getFieldComponentOffset(field,Type::imag,Type::end),
									(T)0, thrust::plus<T>() );
	}
}


template <typename T, typename DEV>
GridData<T,DEV> GridData<T,DEV>::sumFields() const
{
	typename Grid<T>::CPU result( numFields(), 1, 1, representation() );

	T re_tmp, im_tmp, *re, *im;

	re = result.getDataPointer(0,Type::real);

	if ( isComplex() )
		im = result.getDataPointer(0,Type::imag);

	for ( int c=0; c<numFields(); ++c )
	{
		sumField(re_tmp,im_tmp,c);

		re[c] = re_tmp;

		if ( isComplex() )
			im[c] = im_tmp;
	}

	return GridData(result);
}


template <typename T, typename DEV>
void GridData<T,DEV>::insertGrid( const GridData<T,DEV>* other, const int x_offset, const int y_offset )
{
//	if ( isComplex() == other->isComplex() )
		GridFunctions<T,DEV>::insertGrid( other, x_offset, y_offset );
//	else
//		std::cout << "GridData::insertGrid() ERROR: Both grids must be real or complex.\n";
}


template <typename T, typename DEV>
void GridData<T,DEV>::insertRotatedGrid( const GridData* other, const T phi, const int x_offset, const int y_offset, Type::type part )
{
//	if ( isComplex() == other->isComplex() )
		GridFunctions<T,DEV>::insertRotatedGrid( other, phi, x_offset, y_offset, part );
//	else
//		std::cout << "GridData::insertRotatedGrid() ERROR: Both grids must be real or complex.\n";
}


template <typename T, typename DEV>
void GridData<T,DEV>::insertRotatedGridAdditive( const GridData* other, const T phi, int field, const int x_offset, const int y_offset, Type::type part )
{
//	if ( isComplex() == other->isComplex() )
		GridFunctions<T,DEV>::insertRotatedGridAdditive( other, phi, field, x_offset, y_offset, part );
//	else
//		std::cout << "GridData::insertRotatedGrid() ERROR: Both grids must be real or complex.\n";
}


template <typename T, typename DEV>
void GridData<T,DEV>::resample( const int nx_new, const int ny_new )
{
	GridFunctions<T,DEV>::resample(nx_new, ny_new);
}


template <typename T, typename DEV>
void GridData<T,DEV>::resample( GridData*& target, const int nx_new, const int ny_new ) const
{
	GridFunctions<T,DEV>::resample(target, nx_new, ny_new);
}


template <typename T, typename DEV>
void GridData<T,DEV>::copyFieldTo( GridData<T,DEV>* target, int thisField, int otherField ) const
{
	if ( GridData<T,DEV>::dimEq(*this, *target) && thisField < numFields() && otherField < target->numFields() )
	{
		size_t src_begin = getFieldComponentOffset(thisField);
		size_t src_end = getFieldComponentOffset(thisField+1);
		size_t dst_begin = getFieldComponentOffset(otherField);

		thrust::copy(_data->begin()+src_begin, _data->begin()+src_end, target->getVector()->begin()+dst_begin);
	}
	else
		std::cout << "GridData<T,DEV>::copyFieldTo(): ERROR: x and y dimensions are not equal, or field out of range.\n";
}


template <typename T, typename DEV>
int GridData<T,DEV>::initializeGrid( GridData<T,DEV>*& grid, int nx, int ny, int numFields, Type::type representation )
{
	int init = 0;

	if	( grid && (	grid->dimX() != nx ||
					grid->dimY() != ny ||
					grid->numFields() != numFields ||
					grid->representation() != representation )
		)
	{
		delete grid;
		grid = 0;
	}

	if ( grid == 0 )
	{
		grid = new GridData<T,DEV>(nx,ny,numFields,representation);
		init = 1;
	}

	return init;
}


template <typename T, typename DEV>
template <typename T2, typename DEV2>
int GridData<T,DEV>::initializeGrid( GridData<T,DEV>*& grid, const GridData<T2,DEV2>* other )
{
	return initializeGrid(	grid, other->dimX(), other->dimY(),
							other->numFields(), other->representation() );
}


template <typename T, typename DEV>
void GridData<T,DEV>::deleteGrid( GridData<T,DEV>*& grid )
{
	if ( grid )
	{
		delete grid;
		grid = 0;
	}
}


template <typename T, typename DEV>
bool GridData<T,DEV>::dimEq( const GridData<T,DEV>& A, const GridData<T,DEV>& B )
{
	return ( A._dimX == B._dimX && A._dimY == B._dimY );
}


template <typename T, typename DEV>
T GridData<T,DEV>::differenceMagnitude( const GridData<T,DEV>& A, const GridData<T,DEV>& B )
{
	if ( A != B )
	{
		std::cout << "GridData::differenceMagnitude(): Error: Grids have different dimensions.\n";
		return 1000000.0;
	}

	T sumDiff, sumOld;

	if ( A.isComplex() )
	{
		sumDiff = (A-B).complexMagnitudeSq().sumField();
		sumOld = A.complexMagnitudeSq().sumField();
	}
	else if (A.numFields() > 1)
	{
		sumDiff = (A-B).fieldMagnitude().sumField();
		sumOld = A.fieldMagnitude().sumField();
	}
	else
	{
		sumDiff = (A-B).sumField();
		sumOld = A.sumField();
	}

	if ( sumOld < 0.000001 )
		sumOld = 0.000001;

	// Compiler dependent namespace issues?
	return std::sqrt(sumDiff) / std::sqrt(sumOld);
}


template <typename T, typename DEV>
void GridData<T,DEV>::minmax( const GridData<T,DEV>& grid, T& minv, T& maxv, int field, Type::type part )
{
	GridFunctions<T,DEV>::minmax(grid,minv,maxv,field,part);
}


template <typename T, typename DEV>
void GridData<T,DEV>::print( int field, Type::type representation ) const
{
	thrust::host_vector<T> vec = *_data;
	
	T* p_data = thrust::raw_pointer_cast(vec.data()) + getFieldComponentOffset(field,representation);
	
	const int Nx = dimX();
	const int Ny = dimY();

	int i, j, idx;

	for ( j=0; j<Ny; ++j )
	{
		for ( i=0; i<Nx; ++i )
		{	
			idx = j*Nx + i;
			if ( i < Nx-1 )
				std::cout << p_data[idx] << ", ";
			else
				std::cout << p_data[idx];
		}
		std::cout << std::endl;
	}
}	


template <typename T, typename DEV>
void GridData<T,DEV>::print( int field  ) const
{
	std::cout << "Field = " << field << "\nReal:\n";
	print( field, Type::real );

	if ( representation() == Type::complex )
	{
		std::cout << "\nImaginary:\n";
		print( field, Type::imag );
	}
	std::cout << std::endl;
}


template <typename T, typename DEV>
void GridData<T,DEV>::print() const
{
	for (int i=0; i<numFields(); ++i )
		print( i );
}


template <typename T, typename DEV>
void GridData<T,DEV>::printDimensions() const
{
	using namespace std;
	cout << "Nx = " << dimX() << ", Ny = " << dimY();
	cout << ", Num fields = " << numFields() << ", Complex = " << int(isComplex()) << endl;	
}
