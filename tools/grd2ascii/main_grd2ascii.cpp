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
		cout << "Usage:\ngrd2ascii -f \033[1mfilename\033[00m [-min \033[1m#\033[00m] [-max \033[1m#\033[00m]\n";
		cout << "\n--> Filename is specified without numbering and without .grd, e.g.\n";
		cout << "grd2ascii -f file -min 0 -max 7\n will read file.0.grd through file.7.grd, and\n";
		cout << "grd2ascii -f file.1\n will read file.1.grd\n\n";
	}
	
	for ( int i=1; i<argc; i+=2 )
	{
		string arg = argv[i];

		if ( arg == "-f" )
		{
			filename = argv[i+1];
		}
		else if ( arg == "-min" )
		{
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> filenumMin;
		}
		else if ( arg == "-max" )
		{
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> filenumMax;		
		}
	}
	
	if ( filename == "" )
	{
		cout << "No filename provided.\n";
		return 0;
	}

	// Read files and write ascii data
	//
	path p(filename);

	GridCPU tmp;
	string fileOut;
	
	// One file
	if ( filenumMin == -1 || filenumMax == -1 )
	{
		if ( tmp.read( filename.c_str(), param ) )
			return 0;

		fileOut = p.stem().string() + ".txt";
		tmp.writeNumpyArray( fileOut.c_str() );
		//cout << "out: " << fileOut << endl;
	}
	else // Multiple files in a sequence
	{
		for ( int i=filenumMin; i<=filenumMax; ++i )
		{
			if ( tmp.read( filename.c_str(), i ) )
				return 0;

			fileOut = filename + "." + to_string(i) + ".txt";
			tmp.writeNumpyArray( fileOut.c_str() );
			//cout << "out: " << fileOut << endl;
		}
	}

	return 0;	
}
