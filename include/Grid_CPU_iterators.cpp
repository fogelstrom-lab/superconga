// Iterators
//
template <typename T, typename DEV, template <typename> class F>
void iterOne( const GridData<T,DEV>& dst, F<T> functor, int field = -1, Type::type part = Type::complex )
{
	const int Nx = dst.dimX();
	const int Ny = dst.dimY();
	
	const int doReal = int( part==Type::real || part==Type::complex);
	const int doImag = int((part==Type::imag || part==Type::complex) && dst.isComplex());

	const Type::type parts[2] = {Type::real,Type::imag};

	const int c_begin = field==-1 ? 0 : field;
	const int c_end = field==-1 ? dst.numFields() : field+1;

	int i,j,p,c,idx;
	T *data_dst;

	for ( c=c_begin; c<c_end; ++c )
	{
		for ( p=1-doReal; p<1+doImag; ++p )
		{
			data_dst = dst.getDataPointer(c,parts[p]);

			//#pragma omp parallel for private(i,j,idx) shared(c,p,data_src,data_dst,functor)
			for ( j=0; j<Ny; ++j )
			{
				for ( i=0; i<Nx; ++i )
				{
					idx = j*Nx + i;
					data_dst[idx] = functor( data_dst[idx] );
				}
			}
		}
	}
}


template <typename T, typename DEV, template <typename> class F>
void iterTwo( const GridData<T,DEV>& src, GridData<T,DEV>& dst, F<T> functor, Type::type part = Type::complex, int fieldSrc = -1, int fieldDst = -1 )
{
	const int Nx = dst.dimX();
	const int Ny = dst.dimY();

	const int numFieldsSrc = src.numFields();
	const int numFieldsDst = dst.numFields();

	const Type::type parts[2] = {Type::real,Type::imag};

	const int doReal = int( part==Type::real || part==Type::complex);
	const int doImag = int((part==Type::imag || part==Type::complex) && src.isComplex());

	int c_end = 1, doAllFields = 0;

	if ( fieldSrc == -1 )
	{
		c_end = numFieldsSrc;
		doAllFields = 1;
	}

	int i,j,p,c,idx;
	T *data_src, *data_dst;

	for ( c=0; c<c_end; ++c )
	{
		fieldSrc = doAllFields ? c : fieldSrc;
		fieldDst = doAllFields ? c : fieldDst;

		fieldDst = fieldDst >= numFieldsDst ? numFieldsDst-1 : fieldDst;	

		for ( p=1-doReal; p<1+doImag; ++p )
		{
			data_src = src.getDataPointer(fieldSrc,parts[p]);
			data_dst = dst.getDataPointer(fieldDst,parts[p]);

			//#pragma omp parallel for private(i,j,idx) shared(c,p,data_src,data_dst,functor)
			for ( j=0; j<Ny; ++j )
			{
				for ( i=0; i<Nx; ++i )
				{
					idx = j*Nx + i;
					data_dst[idx] = functor( data_src[idx], data_dst[idx] );
				}
			}
		}
	}
}


template <typename T, typename DEV, template <typename> class F>
void iterTwoComplexToReal( const GridData<T,DEV>& src, GridData<T,DEV>& dst, F<T> functor )
{
	const int Nx = dst.dimX();
	const int Ny = dst.dimY();

	int i,j,c,idx;
	T *data_src_re, *data_src_im, *data_dst;

	for ( c=0; c<dst.numFields(); ++c )
	{
		data_src_re = src.getDataPointer(c,Type::real);
		data_src_im = src.getDataPointer(c,Type::imag);
		
		data_dst = dst.getDataPointer(c,Type::real);

		//#pragma omp parallel for private(i,j,idx) shared(c,p,data_src,data_dst,functor)
		for ( j=0; j<Ny; ++j )
		{
			for ( i=0; i<Nx; ++i )
			{
				idx = j*Nx + i;
				data_dst[idx] = functor( data_src_re[idx], data_src_im[idx], data_dst[idx] );
			}
		}
	}
}


// Functors
//
template <typename T>
struct EqFuncScalar
{
	const T a;
	EqFuncScalar(T _a) : a(_a) {}
	T operator()( const T& dst ) { return a; }
};

template <typename T>
struct EqFuncGrid
{
	EqFuncGrid() {}
	T operator()( const T& src, const T& dst ) { return src; }
};

template <typename T>
struct MultFuncOne
{
	const T a;
	MultFuncOne(T _a) : a(_a) {}
	T operator()( const T& dst ) { return a*dst; }
};

template <typename T>
struct MultFuncTwo
{
	MultFuncTwo() {}
	T operator()( const T& src, const T& dst ) { return src*dst; }
};

template <typename T>
struct AddFuncOne
{
	const T a;
	AddFuncOne(T _a) : a(_a) {}
	T operator()( const T& dst ) { return a+dst; }
};

template <typename T>
struct AddFuncTwo
{
	AddFuncTwo() {}
	T operator()( const T& src, const T& dst ) { return src+dst; }
};

template <typename T>
struct SubFuncTwo
{
	SubFuncTwo() {}
	T operator()( const T& src, const T& dst ) { return dst-src; }
};

template <typename T>
struct ComplexMagnitudeSqFunc
{
	ComplexMagnitudeSqFunc() {}
	T operator()( const T& src_re, const T& src_im, const T& dst ) { return src_re*src_re + src_im*src_im; }
};

template <typename T>
struct ComplexPhaseFunc
{
	ComplexPhaseFunc() {}
	T operator() ( const T& x, const T& y, const T& dst )
	{
		T phase = (T)0.0;

		// Interior of quadrants
		if ( x > (T)0.0 && y > (T)0.0 )
			phase = atan(y/x);
		else if ( x < (T)0.0 && y > (T)0.0 )
			phase = M_PI + atan(y/x);
		else if ( x < (T)0.0 && y < (T)0.0 )
			phase = M_PI + atan(y/x);
		else if ( x > (T)0.0 && y < (T)0.0 )
			phase = (T)2.0*M_PI + atan(y/x);
		// Boundaries
		else if ( x == (T)0.0 )
		{
			if ( y > (T)0.0 )
				phase = (T)0.5 * M_PI;
			if ( y < (T)0.0 )
				phase = (T)1.5 * M_PI;
		}
		else if ( y == (T)0.0 )
		{
			if ( x > (T)0.0 )
				phase = (T)0.0;
			if ( x < (T)0.0 )
				phase = M_PI;
		}
		return phase;
	}
};

template <typename T>
struct FieldMagnitudeFunc
{
	FieldMagnitudeFunc() {}
	T operator()( const T& src, const T& dst ) { return dst + src*src; }
};

template <typename T>
struct SqrtFunc
{
	SqrtFunc() {}
	T operator()( const T& dst ) { return sqrt(dst); }
};

template <typename T>
struct LogFunc
{
	const T a;
	LogFunc(T _a) : a(_a) {}
	T operator()( const T& dst ) { return log(dst+a); }
};

