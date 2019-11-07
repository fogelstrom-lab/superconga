#include <iostream>
#include <boost/filesystem.hpp>

#include <Grid.cpp>
#include <Parameters.cpp>

typedef double T;

int main( int argc, char *argv[] )
{
	using namespace std;

	// Parse arguments
	//
	string filename = "";
	double mult = 1.0;

	if ( argc < 3 )
	{
		cout << "No filename or mult factor provided.\n";
		cout << "Usage:\n";
		cout << "gridmult [filename] [mult factor]\n";
		return 0;
	}
	
	filename = argv[1];
	
	stringstream sstream;
	sstream << argv[2];
	sstream >> mult;


	// Read grid and multiply
	Parameters<T> param;

	Grid<T>::CPU* b = new Grid<T>::CPU;

	if ( b->read(filename.c_str(),&param) )
		return 0;

	*b *= mult;


	// Write result
	boost::filesystem::path p(filename);

	string parent = p.parent_path().string();
	
	if ( parent != "" )
		parent = parent + "/";

	string fn_out = parent + p.stem().string() + "_mult.grd";

	if ( param.getAngularResolution() > 0 )
		b->write(fn_out.c_str(),&param);
	else
		b->write(fn_out.c_str());


	// Clean up and return
	delete b;

	return 0;
}


