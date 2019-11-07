#include <iostream>
#include <string>
#include <fstream>
#include <boost/filesystem.hpp>

#include <Grid.cpp>
#include <Parameters.cpp>


typedef double T;
//typedef float T;

// Typedef to simplify things
typedef typename Grid<T>::CPU GridCPU;
typedef typename Grid<T>::GPU GridGPU;

// Main function
int main( int argc, char *argv[] )
{
	using namespace std;
	using namespace boost::filesystem;

	string filename = "";
	int filenumMin = -1, filenumMax = -1;

	Parameters<T>* param = new Parameters<T>;
	
	// Parse arguments
	//
	if ( argc == 1 )
	{
		cout << "Usage:\ncomputeB \033[1mfilename\033[00m\n";
		return 0;
	}
	
	filename = argv[i+1];
	
	if ( filename == "" )
	{
		cout << "No filename provided.\n";
		return 0;
	}

	

	// Write result
	path p(filename);

	GridCPU tmp;
	string fileOut;
	
	if ( tmp.read( filename.c_str(), param ) )
		return 0;

	fileOut = p.stem().string() + ".txt";
	tmp.writeNumpyArray( fileOut.c_str() );
	//cout << "out: " << fileOut << endl;

	return 0;	
}
