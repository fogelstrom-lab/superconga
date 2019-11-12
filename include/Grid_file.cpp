#include <iostream>
#include <fstream>
#include <sstream>
#include <Parameters.h>

template <typename T, typename DEV>
int GridData<T,DEV>::read( const char* filename, int num, Parameters<T>* param )
{
	char buf[256];
	sprintf(buf,"%s.%d.grd",filename,num);

	return read( buf, param );
}


template <typename T, typename DEV>
int GridData<T,DEV>::write( const char* filename, int num, Parameters<T>* param )
{
	char buf[256];
	sprintf(buf,"%s.%d.grd",filename,num);

	return write( buf, param );
}



template <typename T, typename DEV>
int GridData<T,DEV>::read( const char* filename, Parameters<T>* param )
{
	using namespace std;
	
	// Data header structure: (all int datatype)
	// size of datatype in bytes, isComplex, num_x, num_y, num_z, num_fields
	size_t numBytesHeader = 32 * sizeof(int);
	size_t numBytesData_old = numElements() * sizeof(T);
	size_t numBytesData;

	Parameters<T> param_read;

	int storeParameters, parametersVersion; //,numBytesParameters

	ifstream file( filename, ios::in | ios::binary );
	
	if ( file.is_open() )
	{
		int* header = new int[32];
		
		file.read( (char*)header, 32*sizeof(int) );
		
		int sizeOfInputDatatype = header[0];
		
		if ( sizeOfInputDatatype != sizeof(T) )
		{
			std::cout << "Error: datatype to be read is not the same as instantiated class\n";
			file.close();
			return 1;
		}

		int isComplex = header[1];
		_dimX = header[2];
		_dimY = header[3];
		_numFields = header[5];

		storeParameters = header[6];
		parametersVersion = header[7];
//		numBytesParameters = header[8];

		if ( isComplex == 0 )
			_representation = Type::real;
		if ( isComplex == 1 )
			_representation = Type::complex;
			
		numBytesData = numElements() * sizeof(T);

		GridData<T,DEV>::_allocatedMemTotal -= numBytesData_old;
		GridData<T,DEV>::_allocatedMemTotal += numBytesData;

		// Read data to host memory
		thrust::host_vector<T> data_tmp(numElements());
		T* data_ptr = thrust::raw_pointer_cast( data_tmp.data() );

		// Go past header
		file.seekg( numBytesHeader );
			
		// Read parameters
		param_read.setInvalid();
	
		if ( storeParameters == 1 )
		{
			param_read.read( file, parametersVersion );
			param_read.setValid();
		}

		file.read( (char*)data_ptr, numBytesData );
		
		file.close();
		std::cout << "Read: " << filename << "\n";
		
		if ( param )
			*param = param_read;

		if ( _data ) delete _data;
		_data = new DEV(numElements());

		*_data = data_tmp;
		
		return 0;
	} else {
		std::cout << "Could not open file: " << filename << "\n";
		return 1;
	}
}


template <typename T, typename DEV>
int GridData<T,DEV>::write( const char* filename, Parameters<T>* param )
{
	using namespace std;
	
	// Data header structure: (all int datatype)
	// size of datatype in bytes, isComplex, num_x, num_y, num_z, num_fields
	size_t numBytesHeader = 32 * sizeof(int);

	size_t numBytesData, numBytesParameters;//, numBytesPerField;
	numBytesData = numElements() * sizeof(T);
	numBytesParameters = sizeof(Parameters<T>);

	int storeParameters = 0, parametersVersion = 0;
	if ( param != 0 )
	{
		storeParameters = 1;
		parametersVersion = param->getVersion();
	}

	int isComplex_out = int(isComplex());

	// Open file for writing
	ofstream file( filename, ios::out | ios::binary | ios::trunc );
	
	if ( file.is_open() )
	{
		//copyDataToCPU();
		thrust::host_vector<T> data_tmp = *_data;
		T* data_ptr = thrust::raw_pointer_cast(data_tmp.data());
		
		int sizeOfDatatype = sizeof(T);
		int _dimZ = 1;
		// Write header info
		file.write( (char*)(&sizeOfDatatype), sizeof(int) );
		file.write( (char*)(&isComplex_out), sizeof(int) );
		file.write( (char*)(&_dimX), sizeof(int) );
		file.write( (char*)(&_dimY), sizeof(int) );
		file.write( (char*)(&_dimZ), sizeof(int) ); // legacy stuff
		file.write( (char*)(&_numFields), sizeof(int) );

		file.write( (char*)(&storeParameters), sizeof(int) );
		file.write( (char*)(&parametersVersion), sizeof(int) );
		file.write( (char*)(&numBytesParameters), sizeof(int) );
		
		file.seekp( numBytesHeader );
		
		// Write parameters
		if ( storeParameters )
			file.write( (char*)param, numBytesParameters );

		// Write grid data
		file.write( (char*)data_ptr, numBytesData );
		
		file.close();
		std::cout << "Wrote binary: " << filename << "\n";
		return 0;
	} else {
		std::cout << "Could not open file: " << filename << "\n";
		return 1;
	}
}


template <typename T, typename DEV>
int GridData<T,DEV>::readNumpyArray( const char* filename, int numFields )
{
	using namespace std;
	
	const int MAX_TOKENS_PER_LINE = 16384;

	// create a file-reading object
	ifstream fin;
	fin.open( filename );

	if ( !fin.good() )
		return 1;
  
	// Determine grid dimensions (Nx)
	int Nx, Ny, Nf;
	
	int n_delimit = 0, found = 0, found_last = 1;
	
	string line;
	getline( fin, line );
	
	while ( line.find('#', 0) <= line.length() || line.length() == 0 )
		getline( fin, line );
		
	while ( found != found_last )
	{
		found_last = found;	
		found = line.find(' ', found_last+1);

		if ( found >= line.length() )
			break;

		n_delimit++;
	}

	Nx = n_delimit;

	// Determine grid dimensions (Ny,Nf)	
	int lines_data = 0;

	fin.clear();
	fin.seekg(0);

	while ( !fin.eof() )
	{
		getline( fin, line );
		
		if ( line.find('#',0) >= line.length() )
		{
			if ( (int)line[0] > 0 )
				lines_data++;
		}			
	}

	if ( numFields > 0 )
	{
		Ny = lines_data / numFields;
		Nf = numFields;
	}
	else
	{
		Ny = 1;
		Nf = lines_data;
	}

	// Create grid
	size_t numBytesData_old = numElements() * sizeof(T);

	_dimX = Nx;
	_dimY = Ny;
	_numFields = Nf;
	_representation = Type::real;

	thrust::host_vector<T> data_tmp( numElements() );

	// Rewind
	fin.clear();
	fin.seekg(0);

	// Read file contents
	T* data_ptr = thrust::raw_pointer_cast(data_tmp.data());
	int j = 0;
	
	while ( !fin.eof() )
	{
		getline( fin, line );
	
		int idx_delimit[MAX_TOKENS_PER_LINE];
		n_delimit = 0;
		
		found = 0;
		found_last = 1;
		
		if ( line.find('#', 0) >= line.length() && line.length() > 0 )
		{
			idx_delimit[n_delimit++] = -1;
		
			while ( found != found_last )
			{
				found_last = found;	
				found = line.find(' ', found_last+1);

				if ( found >= line.length() )
					break;

				idx_delimit[n_delimit++] = found;
			}

			T subfloat;
			for ( int i=0; i<n_delimit-1; ++i )
			{
				int start = idx_delimit[i]+1;
				int num = idx_delimit[i+1] - start;
				string sub = line.substr(start, num);

				stringstream tmp;
				tmp << sub;
				tmp >> subfloat;
				data_ptr[j++] = subfloat;
			}		
		}
	}
	
	fin.close();

	*_data = data_tmp;
	GridData<T,DEV>::_allocatedMemTotal -= numBytesData_old;
	GridData<T,DEV>::_allocatedMemTotal += numElements() * sizeof(T);
		
	return 0;
}


template <typename T, typename DEV>
int GridData<T,DEV>::writeNumpyArray( const char* filename, Type::type representation )
{
	using namespace std;

	// Open file for writing
	ofstream file( filename, ios::out | ios::binary | ios::trunc );
	
	if ( file.is_open() )
	{
		typename Grid<T>::CPU grid_tmp = *this;

		T *data_re, *data_im, val_re, val_im;

		int n, idx;
		char tmp[64];
		char* sig;
		
		int writeComplex = int(isComplex() && representation == Type::complex);

		n = sprintf(tmp,"# Nx=%d, Ny=%d, Nf=%d, complex=%d\n", _dimX, _dimY, _numFields, writeComplex );
		file.write( tmp, n );
		
		for ( int c=0; c<_numFields; ++c )
		{
			if ( writeComplex )
			{
				data_re = grid_tmp.getDataPointer( c, Type::real );
				data_im = grid_tmp.getDataPointer( c, Type::imag );
				
				for ( int j=0; j<_dimY; ++j )
				{
					for ( int i=0; i<_dimX; ++i )
					{
						idx = j*_dimX + i;

						val_re = data_re[idx];
						val_im = data_im[idx];
						
						if ( val_im >= (T)0.0 )
							sig = (char*)"+";
						else
							sig = (char*)"";

						n = sprintf(tmp,"(%.10e%s%.10ej) ", double(val_re), sig, double(val_im));
						file.write( tmp, n );
					}
					file.write( (char*)"\n", 1 );
				}
			} else {
				data_re = grid_tmp.getDataPointer( c, Type::real );
				
				for ( int j=0; j<_dimY; ++j )
				{
					for ( int i=0; i<_dimX; ++i )
					{
						idx = j*_dimX + i;

						val_re = data_re[idx];
				
						n = sprintf(tmp,"%.10e ",double(val_re));
						file.write( tmp, n );
					}
					file.write( (char*)"\n", 1 );
				}
			}
			file.write( (char*)"\n", 1 );
		}			
		
		file.close();
		std::cout << "Wrote Ascii in numpy format: " << filename << "\n";
		return 0;
	} else {
		std::cout << "Could not open file: " << filename << "\n";
		return 1;
	}
}


template <typename T, typename DEV>
int GridData<T,DEV>::readParameters( const char* filename, int num, Parameters<T>* param )
{
	char buf[256];
	sprintf(buf,"%s.%d.grd",filename,num);

	return readParameters( buf, param );
}


template <typename T, typename DEV>
int GridData<T,DEV>::readParameters( const char* filename, Parameters<T>* param )
{
	using namespace std;
	
	// Data header structure: (all int datatype)
	// size of datatype in bytes, isComplex, num_x, num_y, num_z, num_fields
	size_t numBytesHeader = 32 * sizeof(int);
	int storeParameters, parametersVersion; //,numBytesParameters;

	Parameters<T> param_read;

	ifstream file( filename, ios::in | ios::binary );
	
	if ( file.is_open() )
	{
		int* header = new int[32];
		
		file.read( (char*)header, 32*sizeof(int) );
		
		int sizeOfInputDatatype = header[0];
		
		if ( sizeOfInputDatatype != sizeof(T) )
		{
			std::cout << "Error: datatype to be read is not the same as instantiated class\n";
			file.close();
			return 1;
		}

		storeParameters = header[6];
		parametersVersion = header[7];

		// Go past header
		file.seekg( numBytesHeader );
			
		// Read parameters
		param_read.setInvalid();

		if ( storeParameters == 1 )
		{
			param_read.read( file, parametersVersion );
			param_read.setValid();
		}
		
		file.close();
		
		if ( param )
			*param = param_read;

		return 0;
	} else {
		std::cout << "Could not open file: " << filename << "\n";
		return 1;
	}
}
