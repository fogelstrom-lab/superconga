#ifndef _GRIDDATA_H
#define _GRIDDATA_H

#ifdef __CUDACC__
#define _DEV_ __host__ __device__
#else
#define _DEV_
#endif

//#include <string.h>
#include <cmath>

#include <Type.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <cplx.h>

// Forward declarations. These can be used when we want to refer to another
// class without including the .h file. GridData is forward declared since 
// it referred to in the wrapper structs below. 
template<typename,typename> class GridData;
template<typename,typename> class GridFunctions;
template<typename> class Parameters;


// Wrapper structs/typedefs for the GridData class. This makes it simpler to
// read and write GridData variable declarations. For example:
// GridData<double,thrust::host_vector<double> >    --->  Grid<double>::CPU
// GridData<double,thrust::device_vector<double> >  --->  Grid<double>::GPU
// thrust::host_vector<double>    --->  Vector<double>::CPU
// thrust::device_vector<double>  --->  Vector<double>::GPU
template <typename T>
struct Vector
{
	typedef thrust::host_vector<T> CPU;

#ifdef __CUDACC__
	typedef thrust::device_vector<T> GPU;
#else
	typedef thrust::host_vector<T> GPU;
#endif
};

template <typename T>
struct Grid
{
	typedef GridData<T,thrust::host_vector<T> > CPU;

#ifdef __CUDACC__
	typedef GridData<T,thrust::device_vector<T> > GPU;
#else
	typedef GridData<T,thrust::host_vector<T> > GPU;
#endif
};


// Diverging codepaths (CPU and GPU) are put in GridFunctions class, with template specialization telling them apart
template <typename T, typename DEV>
class GridData : public GridFunctions<T,DEV>
{
public:
		GridData();
		~GridData();
			
		// Copy constructor
		GridData( const GridData& other );					// Copies entire object

		// Allocates memory only, no field data values are set
		GridData( const GridData& other, int numFields );	// Allocates with field dimensions from other, but number of fields as specified
		GridData( int nx, int ny, int numFields = 1, Type::type representation = Type::complex );

	// Operators
	bool		operator==( const GridData& right ) const;
	bool		operator!=( const GridData& right ) const;

	GridData&	operator=( GridData right );
	GridData&	operator=( const cplx<T> right );
	GridData&	operator=( const T right );

	// Try thrust::for_each instead of thrust::transform for better performance for operators?

	// IMPORTANT NOTE on operators:
	// GridData does NOT represent matrices. As such the operator* does NOT perform a matrix-matrix multiplication,
	// but rather an element-element multiplication; for all elements in the array: c_i = a_i * b_i

	void		operator+=( const GridData& right );
	void		operator+=( const cplx<T> right );
	void		operator+=( const T right );
	void		operator-=( const GridData& right );
	void		operator-=( const cplx<T> right );
	void		operator-=( const T right );
	void		operator*=( const GridData& right );
	void		operator*=( const cplx<T> right );
	void		operator*=( const T right );

	GridData	operator+( GridData right ) const;
	GridData	operator+( const cplx<T> right ) const;
	GridData	operator+( const T right ) const;
	GridData	operator-( GridData right ) const;
	GridData	operator-( const cplx<T> right ) const;
	GridData	operator-( const T right ) const;
	GridData	operator*( GridData right ) const;
	GridData	operator*( const cplx<T> right ) const;
	GridData	operator*( const T right ) const;

	GridData	operator-() const;
	
	// Returns a copy of the specified field (somewhat dubious use of operator[])
	GridData	operator[]( const int field ) const;

	// operator() is identical to getDataPointer()
	T*			operator()( const int field = 0, Type::type representation = Type::real ) const {
					return thrust::raw_pointer_cast(_data->data()) + getFieldComponentOffset(field,representation);}	

	// Type conversion operator
	template <typename T2, typename DEV2>
	operator GridData<T2,DEV2>();

	template <typename T2>
	operator GridData<T2,thrust::device_vector<T2> >();


	//// Query functions ////

	int			dimX() const { return _dimX; }
	int			dimY() const { return _dimY; }
	int			numFields() const { return _numFields; }
	Type::type 	representation() const { return _representation; }

	bool		isComplex() const { return _representation == Type::complex; }

	DEV*		getVector() const { return _data; }
	T*			getDataPointer( const int field = 0, Type::type representation = Type::real ) const {
					return thrust::raw_pointer_cast(_data->data()) + getFieldComponentOffset(field,representation);}

	//// Modifying basic functions ////

	// Set elements of all or specified field(s) to scalar value.
	void		setValue( const T val, Type::type representation = Type::complex, const int field = -1 );

	// Set all elements to zero.
	void		setZero();

	// Copy raw data from p_data to this grid.
	int			setData( const T* p_host_data, size_t numElements );

	// In-place complex conjugation of field data.
	GridData&	conjugate();

	// In-place transponate of field data. (Not implemented for CPU!!!)
	GridData&	transpose( const int field = -1 );

	// In-place square root (Only per-element. Does NOT perform complex-valued square-root!)
	GridData&	sqrt();


	//// Non-modifying utility functions ////

	// In all cases where the argument is of type GridData*, the object to which the pointer
	// refers must be initialized and have the correct dimensions.

	// Compute absolute value for grid (works for complex as well)
	GridData	abs() const;

	// Compute the complex magnitude squared
	void		complexMagnitudeSq( GridData* result ) const;
	GridData	complexMagnitudeSq() const;

	// Compute complex phase of this (complex) grid
	void		complexPhase( GridData* result ) const;
	GridData	complexPhase() const;

	// Compute complex phase gradient of this grid
	void		complexPhaseGradient( GridData* result, const T h = 1.0 ) const;
	GridData	complexPhaseGradient( const T h = 1.0 ) const;

	// Compute the magnitude of this (real valued) vector field
	void		fieldMagnitude( GridData* result ) const;
	GridData	fieldMagnitude() const;

	// Compute divergence of vector field (Not implemented for CPU!!!)
	void 		div( GridData* result, const T h = 1.0 ) const;
	GridData 	div( const T h = 1.0 ) const;

	// Compute curl of vector field (Not implemented for CPU!!!)
	// Only returns z-component, since x- and y-components will be zero for a 2d-field
	void 		curl( GridData* result, const T h = 1.0 ) const;
	GridData 	curl( const T h = 1.0 ) const;

	// Compute laplacian (div(grad)) of field (Not implemented for CPU!!!)
	void		del2( GridData* result, const T h = 1.0 ) const;
	GridData 	del2( const T h = 1.0 ) const;

	// Sum all field components.
	void		sumFieldComponents( GridData* result ) const;
	GridData	sumFieldComponents() const;

	// Sum all elements of specified fields. Result returned (real) or put in reference variables (complex).
	T			sumField( const int field = 0 ) const;							// Use for real valued
	void		sumField( T& sum_re, T& sum_im, const int field = 0 ) const;	// Use for real or complex valued

	// Sum all elements in a field, for all fields
	GridData 	sumFields() const;

	// Copy single field from this to target
	void		copyFieldTo( GridData* target, int thisField, int otherField ) const;


	//// Modifying utility functions ////

	// Insert other grid into this. <x,y>_offset are offset in discrete elements from center.
	// x_offset = y_offset = 0 results in a centered grid.
	void		insertGrid(	const GridData* other, const int x_offset = 0, const int y_offset = 0 );

	// Insert other grid into this, with applied rotation (to other grid).
	void		insertRotatedGrid( const GridData* other, const T phi,
					const int x_offset = 0, const int y_offset = 0, Type::type part = Type::complex );

	// Add other grid to this, with applied rotation (to other grid).
	void		insertRotatedGridAdditive( const GridData* grid_insert, const T phi, int field = -1,
					const int x_offset = 0, const int y_offset = 0, Type::type part = Type::complex );

	// Change resolution of grid, with linear interpolation of existing data.
	void		resample( const int nx_new, const int ny_new );
	void		resample( GridData*& target, const int nx_new, const int ny_new ) const;  //(CPU NOT IMPLEMENTED)


	//// Helper query functions ////

	// Returns the pointer offset (==number of data elements) to the specified field.
	size_t		getFieldComponentOffset( int field, const Type::type representation = Type::real, Type::location pos = Type::start ) const {
					const int complex = int(isComplex());
					return _dimX * _dimY * (field * (1+complex) + int(representation==Type::imag && complex) + int(pos==Type::end)); }

	// Returns the number of data elements in the entire array.
	size_t		numElements() const { return numElementsField() * _numFields; }

	// Returns the number of elements of one field.
	size_t		numElementsField() const { return size_t(_dimX) * size_t(_dimY) * size_t(1+int(isComplex())); }


	//// File I/O ////

	int			read( const char* filename, Parameters<T>* param = 0 );
	int			read( const char* filename, int num, Parameters<T>* param = 0 );

	int			write( const char* filename, Parameters<T>* param = 0 );
	int			write( const char* filename, int num, Parameters<T>* param = 0 );

	int			readNumpyArray( const char* filename, int numFields = 1 );	// Can only read real valued data
	int			writeNumpyArray( const char* filename, Type::type representation = Type::complex );

	// Print data to terminal
	void		print( int field, Type::type representation ) const;
	void		print( int field ) const;
	void		print() const;

	void 		printDimensions() const;


	//// Static functions ////
	static int	readParameters( const char* filename, Parameters<T>* param );
	static int	readParameters( const char* filename, int num, Parameters<T>* param );

	// Safely initialize or delete GridData pointers, without copying values.
	// *grid must point to properly initialized GridData, or == 0.
	static int		initializeGrid( GridData*& grid, int Nx, int Ny, int numFields=1, Type::type representation = Type::complex );

	template <typename T2, typename DEV2>
	static int		initializeGrid( GridData<T,DEV>*& grid, const GridData<T2,DEV2>* other );

	static void		deleteGrid( GridData*& grid );

	// Returns True if A.dimX * A.dimY == B.dimX * B.dimY
	static bool		dimEq( const GridData& A, const GridData& B );

	// Used to compute residual
	static T 		differenceMagnitude( const GridData& A, const GridData& B );

// FUNCTIONS BELOW SHOULD NOT BE STATIC MEMBERS!! //////////
	static void		minmax( const GridData& grid, T& minv, T& maxv, int field=0, Type::type part = Type::real );
// TO HERE //////////////////////////////////////////////////////

	static size_t	allocatedMemTotal() { return _allocatedMemTotal; }
	static void		printAllocatedMemTotal() { std::cout << _allocatedMemTotal << std::endl; }

	// GridFunctions is a class used to contain the code which is CPU or GPU specific.
	// By declaring it a friend class it can access the private members of this class.
	friend class GridFunctions<T,DEV>;


	// Functors for complex valued multiplication
	//

	// Complex vector dot product
	struct complexVectorProduct_functor
	{
		template <typename Tuple>
		_DEV_ void operator()( Tuple vecs )
		{
			const T temp_re = thrust::get<0>(vecs);
			const T temp_im = thrust::get<1>(vecs);
			thrust::get<0>(vecs) = ( temp_re * thrust::get<2>(vecs) ) - ( temp_im * thrust::get<3>(vecs) );
			thrust::get<1>(vecs) = ( temp_re * thrust::get<3>(vecs) ) + ( temp_im * thrust::get<2>(vecs) );
		}
	};


	// Scale complex vector with complex scalar
	struct scaleComplexVector_functor
	{
		const cplx<T> scalar;

		scaleComplexVector_functor( const cplx<T> _scalar ) : scalar(_scalar) {}

		template <typename Tuple>
		_DEV_ void operator()( Tuple vec )
		{
			const T temp_re = thrust::get<0>(vec);
			const T temp_im = thrust::get<1>(vec);

			thrust::get<0>(vec) = ( temp_re * scalar.x ) - ( temp_im * scalar.y );
			thrust::get<1>(vec) = ( temp_re * scalar.y ) + ( temp_im * scalar.x );
		}
	};


	// Scale complex vector with complex scalar
	struct addComplexScalar_functor
	{
		const cplx<T> scalar;

		addComplexScalar_functor( const cplx<T> _scalar ) : scalar(_scalar) {}

		template <typename Tuple>
		_DEV_ void operator()( Tuple vec )
		{

			thrust::get<0>(vec) += scalar.x;
			thrust::get<1>(vec) += scalar.y;
		}
	};	

private:

	// Specifies if GridData contains real (Type::real) or complex (Type::complex) valued data
	Type::type	_representation;
	
	// Number of elements in x and y
	int		_dimX;
	int		_dimY;

	// Number of fields
	int		_numFields;
	
	// DEV can be thrust::host_vector<T> or thrust::device_vector<T>. Contains data array.
	DEV*	_data;

	// Total allocated memory for ALL instantiated Grid<T>::DEV objects
	static size_t	_allocatedMemTotal;

	// Swap function implemented according to the copy-and-swap c++ idiom (very useful!)
	template <typename S, typename DEVS>
	friend void swap( GridData<S,DEVS>& grid1, GridData<S,DEVS>& grid2 );
};

#endif
