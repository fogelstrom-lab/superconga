#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <Grid.cpp>
#include <Parameters.cpp>
//#include <Parameters_old.cu>
//#include <Visualization.cu>

#ifdef __CUDACC__
#include <GeometryGroup.cu>
#include <GeometryComponent.cu>
#endif

typedef double T;

// Typedefs to simplify things
typedef typename Grid<T>::GPU GridGPU;
typedef typename Grid<T>::CPU GridCPU;


int main( int argc, char *argv[] )
{
	using namespace std;
	using namespace boost::filesystem;

	if ( argc == 1 )
	{
		cout << "No filename provided.\n";

		cout << "\nUsage:\nsum_dos -f [filename] [FLAGS]\n";
		cout << "\nFlags:\n -a  Output dos per unit area\n";
		return 0;
	}

	string filename;

	string filename_base;
	string filename_in;
	string filename_out;

	T norm_mult = 1.0;
	T norm_area = 1.0;
	//T norm_area = 0.96*0.96;

	for ( int i=1; i<argc; i+=2 )
	{
		string arg = argv[i];

		if ( arg == "-f" )
		{
			filename = argv[i+1];

			path p(filename);

			if ( !(p.has_extension()) || (p.has_extension() && p.extension() != ".grd") )
				filename_base = filename;

			if ( p.has_extension() && p.extension() == ".grd" )
			{
				string ppath = p.parent_path().string();
				
				if ( ppath == "" )
					filename_base = p.stem().string();
				else
					filename_base = p.parent_path().string() + "/" + p.stem().string();
			}

			filename_in = filename_base + ".grd";
			filename_out = filename_base + ".txt";
		}
		else if ( arg == "-a" )
		{
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> norm_area;
		}
		else if ( arg == "-n" )
		{
			norm_mult = 2.0*2.0*M_PI;
		}
	}

	GridCPU ldos;
	Parameters<T> param;

	if ( ldos.read(filename_in.c_str(),&param) )
		return 0;
	
	param.print();

	const int Nx = ldos.dimX();
	const int Ny = ldos.dimY();

	const int Ne = param.getNumEnergies();
	const T e_min = param.getEnergyMin();
	const T e_max = param.getEnergyMax();

	// Field 0: energy levels, Field 1: dos
	GridCPU dos(Ne,1,2,Type::real);

	T* en_p = dos.getDataPointer(0);
	T* dos_p = dos.getDataPointer(1);

	const T dos_mult = 1.0/(Nx*Ny*norm_area*norm_mult);

	for ( int e=0; e<Ne; ++e )
	{
		en_p[e] = e_min + (e_max-e_min) * (T(e)/T(Ne-1));
		dos_p[e] = ldos.sumField(e) * dos_mult;
	}

	dos.writeNumpyArray( filename_out.c_str() );

	return 0;
}
