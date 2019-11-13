//*******************************************************
// Standard libraries
//*******************************************************
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

//*******************************************************
// API libraries
//*******************************************************
#include <cplx.h>	// Complex number datatype
#include <Grid.h> // Data type for observables defined on the lattice
#include <Parameters.h> // Parameters specifying which the superconducting system to simulate
#include <ParameterIterator.h> // Used for sweeping parameters (e.g. energy, temperature)
#include <GeometryGroup.h> // The geometry group stores a list of polygons/discs/... that togheter define the sample geometry (container class for GeometryComponent)
#include <GeometryComponent.h> // Pure virtual geometry object class - base class for discs, polygons etcetera
#include <Disc.h> // Used for adding/subtracting discs from sample geometry
#include <Polygon.h> // Used for adding/subtracting polygons from sample geometry (!! currently limited to convex hull !!)
#include <PolygonGeometry.h>
#include <DiscGeometry.h>
#include <OrderParameter.h> // Container class for order parameter components 
#include <OrderParameterComponent.h> // Pure virtul for s-wave, d-wave, ... components
#include <OrderParameterComponent_S.h> // s-wave
//#include <OrderParameterComponent_D.h> // d-wave
//#include <OrderParameterComponent_P.h> // p-wave (p_x part)
#include <RiccatiSolver.h> // Pure virtual for different kinds of Riccati solvers
//#include <RiccatiSolverBulk.h> // Riccati solver for bulk
#include <RiccatiSolverConfined.h> // Riccati solver for confined geometries
#include <ComputeProperty.h> // Pure virtual for compute properties (observables etc). The observables to be computed are added to a list of compute objects.
#include <ComputeOrderParameter.h> // Compute order parameter
#include <ComputeCurrent.h> // Compute charge current
#include <ComputeFreeEnergy.h> // Compute free energy (Eilenberger functional)
#include "ComputeVectorPotential.h"
//#include <ComputeLDOS.h> // Compute Local Density of States (!! note !! more convenient to run in a separate program using the Keldysh iterator, reading a fully converged solution as start file)
#include <BoundaryStorage.h> // Storage for coherence functions on the system edges - we save memory by only saving on boundaries between self-consistent iterations. These are the start point for each new energy and angle.
#include <BoundaryConditionSpecular.h> // Boundary condition for perfect specular reflection
#include <IntegrationIterator.h> // Pure virtual for different energy integration schemes (Ozaki, Matsubara, Keldysh, ...)
#include <IntegrationIteratorOzaki.h> // Calculate energy integral through Ozaki summation
//#include <IntegrationIteratorMatsubara.h> // Matsubara sum, deprecated by Ozaki method
//#include <IntegrationIteratorKeldysh.h> // Used for calculating observables on the real energy axsis (e.g. DOS)
#include <GreensFunctionStatic.h> // Misc. Green function calculations
#include <Context.h> // The context that connects all classes (helpful since enacpsulation is used)
#include <Initial.h> // Used for generating initial guess for order parameter
#include <Visualization.h> // Live visualization of desired observables through OpenGL
#include <GradientAccelerator.h> // Naive method for speeding up self-consistent iterations. Should be replaced by Anderson accelerator or similar.
#include <ContextModule.h> // Pure virtual for context modules (e.g. Parameters, ComputeProperty, GradientAccelerator, RiccatiSolver, OrderParameter, GeometryGroup, IntegrationIterator)


//*******************************************************
// Defines
//*******************************************************
// Used for rough comparison between floating point numbers
#ifndef MYEPSILON
#define MYEPSILON (double)1E-7
#endif


//*******************************************************
// Namespaces
//*******************************************************
using namespace std;


//*******************************************************
// Forward declarations
//*******************************************************
int readCommandLineArguments( int ___argc, char *const *___argv );
void runInTerminal();
int saveParamsMetaData( const char* fileName );
bool startFileExists( const char* startFile );
void createProjectDirectory( const char* dirname );
void myKeyboardFunc( unsigned char key, int x, int y );
void constructFullFilename( char* fileName );
void getFileName( char* fileName );
void getFilePath( char* filePath );
void saveInducedB();
void updateBfromFile();


//*******************************************************
// Typedefs
//*******************************************************
// Run with double or single floating point precision, double is highly recommended
typedef double T;


//*******************************************************
// Global objects
//*******************************************************
Context<T>* ctx = new Context<T>();
Visualization<T>* vis;
GradientAccelerator<T>* acc;
ComputeFreeEnergy<T>* freeEnergyObj;
ComputeCurrent<T>* current;


//*******************************************************
// Misc. definitions
//*******************************************************
char baseFileName[512];
char startBaseFileName[512];
char runPath[512];
char baseFilePath[512];
char startPath[512];
int numLoops = 0;
double totalTime = 0.0;
int g_forceSave = 0;
T B_external;
T radius;


//*******************************************************
// Parameters
//*******************************************************
const int saveData = 0;				// Save data to file
const int saveDataInterval = 200;	// Number of self-consistent iterations

const int runVisual = 1;			// Run in OpenGL mode or terminal mode
// Sample will cover 2*outerRadius (percentage) of available simulation space.
// Make sure to leave some extra space (used for interpolation scheme).
const double outerRadius = 0.48;

// Save path is homedir/projectPath/projectName/.../datafile.data
// where ... might be additional directories to sort according to system size,
// temperature, order parameter basis function etcetera.
const char *homedir;
char* projectPath = (char*)
"projects/EilenbergerRiccati/data/demo/";
const char* projectName = (char*)"square_VL";

// Used to construct path for start file
char* start_base = (char*) "projects/EilenbergerRiccati/data/demo/";
const char* start_project = (char*)"square_VL";

//const T latticeOrientation = 0.25*M_PI; // Crystal ab-axis rotation

const int Nx = 401;			// Number of lattice points
const int Na = 60;			// Number of discrete trajectory angles
const T convergence = 1E-8; // Convergence criterion
const int Ne = 10;			// Number of Ozaki poles

const int MAXLOOPS = 25000;//1000;//5000;
const int MINLOOPS = 150;//300;
const int RELAXCOUNT = 5;//30;
const int RELAXFINAL = 50;//30;

T T_Tc		 = 0.5;	// Temperature: T/T_c
T fluxQuanta = 90.0;	// Number of flux quanta per unit area
T coherenceLengths = 60.0;  // Grain size in coherence lengths (hbar*v_F/2*pi*k_BT_c)
T area = 91.6646;			// Area in internal units


//*******************************************************
// MAIN
//*******************************************************
int main( int argc, char *argv[] )
{
	// Construct homedirectory path
	if ((homedir = getenv("HOME")) == NULL)
	{
	    homedir = getpwuid(getuid())->pw_dir;
	}

	// Update parameters from command line
	if ( ( argc > 1 ) )
	{
		readCommandLineArguments( argc, argv );
	}

	// Storage for system simulation parameters (e.g. temperature, lattice size)
	Parameters<T>* param = new Parameters<T>( ctx );

	// Calculate sidelength in internal length units
	radius = coherenceLengths/(2.0*M_PI);

	param->setGridResolution( Nx );					// Set number of lattice sites
	param->setGridWidth( radius /					// Set system sidelength
						 (2.0*outerRadius) );
	param->setAngularResolution( Na );				// Number of discrete steps in momentum angle
	param->setConvergenceCriterion( convergence );	// Set convergence criterion
    param->setNumMatsubaraFrequencies( Ne );		// Set number of Ozaki poles

	
	B_external = fluxQuanta / area; // Strength of external magnetic field (normal to the sample) in units of flux quanta per unit area
	param->setMagneticFieldStrength( B_external ); // Set external field
	param->setTemperature( T_Tc );				   // Set temperature

	// Create geometry, add/remove geometry objects (e.g. discs, polygons) to this group
	GeometryGroup<T>* geom = new GeometryGroup<T>( ctx );

	// Parameters used to set up a hexagon
	const T hexScale = 0.7;
	//const T octScale = 0.75;
	const T side_c = outerRadius / sqrt(T(2.0));
	const T side_bleed = 0.5 / sqrt(T(2.0)); // Bleeding edge size, make sure to use smaller sidelength than this, or interpolation will explode

	geom->add( new PolygonGeometry<T>( 6, outerRadius*hexScale, M_PI/6.0 ), Type::add );
	//geom->add( new DiscGeometry<T>( outerRadius ), Type::add );

	RiccatiSolver<T>* solver; // Create Riccati solver
	// Specify external field gauge/orientation (use VectorPotential::None for never evaluating vector potential)
	solver = new RiccatiSolverConfined<T, Gamma::General, VectorPotential::Radial>( ctx );
	solver->add( new BoundaryConditionSpecular<T> ); // Add specular BC

	// Initialize order parameter, add s-wave order parameter
	OrderParameter<T>* delta = new OrderParameter<T>( ctx );
	OrderParameterComponent_S<T>* deltaComponent = new OrderParameterComponent_S<T>( param );
	//deltaComponent->setLatticeOrientation( latticeOrientation ); // Set crystal ab-axes rotation (used for d-wave)

	//srand (time(NULL)); // Randomize seed

	// Construct directory path for saving files
	sprintf( baseFilePath, "%s/%s", homedir, projectPath );
	sprintf( baseFileName, "%s%s/s-wave", baseFilePath, projectName );
	sprintf( startBaseFileName, "%s/%s%s/s-wave", homedir, start_base, start_project );

	// Full path for save file
	sprintf( runPath, "%s/R%2.3f_T%2.4f_n%d",
			 baseFileName, coherenceLengths, param->getTemperature(),
			 int(fluxQuanta) );

	// Create directory if not existing
	createProjectDirectory( runPath );

	// Full path for start file 
	sprintf( startPath, "%s/R%2.3f_T%2.4f_n%d",
			 startBaseFileName, coherenceLengths, param->getTemperature(),
			 int(fluxQuanta) );

	// Order parameter file paths
	char runFile[512];
	char startFile[512];
	sprintf(runFile, "%s/delta.grd", runPath);
	sprintf(startFile, "%s/delta_start.grd", startPath);


	cout << "\n================================================================================" << endl;
	cout << "New run" << endl;
	cout << "flux quanta = " << int(fluxQuanta) << endl;
	cout << "T/Tc = " << T_Tc << endl;
	cout << "NEW FILE: " << runPath << endl;
	cout << "================================================================================\n" << endl;

	// Read and initialize OP from file
	if ( startFileExists(runFile) || startFileExists(startFile) ) {
		cout << "Initial::read" << endl;
		if ( startFileExists(startFile) )
			Initial<T>::read( deltaComponent, startFile );
		else
			Initial<T>::read( deltaComponent, runFile );
	// Start with phase winding guess (good for finding Abrikosov vortices)
	} else {
		cout << "Initial::phaseWinding" << endl;
		Initial<T>::phaseWinding( deltaComponent, fluxQuanta, 0.0, 0.0 );
	}

	// Add the initialized order parameter component
	delta->add( deltaComponent );

	// Create compute object(s) for observable(s)
	new ComputeOrderParameter<T>( ctx );
	current = new ComputeCurrent<T>( ctx );
	freeEnergyObj = new ComputeFreeEnergy<T>( ctx );
	freeEnergyObj->setCompute(0);

	// Convergence acceleration
	acc = new GradientAccelerator<T>( ctx );
	acc->setRelaxCount( RELAXCOUNT );
	acc->setRelaxFinal( RELAXFINAL );

	// Number of plots in visualization = numPlotsY * numPlotsX
	int numPlotsY = 2;
	int numPlotsX = 3;

	// Create visualization object
	Grid<T>::GPU* delta_vis = deltaComponent->getOrderParameterField();
	vis = new Visualization<T>( delta_vis, numPlotsX, numPlotsY );
	vis->setGeometry( geom );
	vis->setPrintParameters( param );

	// Initialize run
	ctx->initialize();

	// Output data
	cout << endl << "##################################################" << endl;
	cout << "Area analytical square: R = " << coherenceLengths*coherenceLengths << endl;
	cout << "Area compute = " << geom->area() << endl;
	cout << "Area lattice nodes = " << geom->area(1) << endl;
	cout << "h = " << param->getGridElementSize() << endl;
	cout << "Delta bulk = " << Initial<T>::computeDeltaBulk( param->getTemperature() ) << endl;
	cout << "N0 (gridResolutionBase): " << param->getGridResolutionBase() << endl;
	cout << "N1 (gridResolutionCoherence): " << param->getGridResolutionCoherence() << endl;
	cout << "numFreq: " << param->getNumMatsubaraFrequencies() << "\n";
	cout << "##################################################" << endl;

	// Save parameters to file
	char paramFile[512];
	sprintf(paramFile, "%s/params.dat", runPath);
	saveParamsMetaData(paramFile);

	// Start main program loop
	if ( runVisual )
	{
		// Run in OpenGL mode
		initGL( argc, argv, vis->getNumPixelsWindowX(), vis->getNumPixelsWindowY() );
		glutKeyboardFunc( myKeyboardFunc );
		glutMainLoop();
	}
	else
		runInTerminal(); // Run in terminal mode

	std::cout << "Successful termination!\n";
	return 0;
}


// Use Boost to check if start file exists and is a proper file
bool startFileExists( const char* startFile )
{
	boost::filesystem::path p ( startFile );	// Create Boost filepath object

	// Check if startFile exists
	if ( boost::filesystem::exists(p) )
	{
		// Check if startFile is file, directory or something else
		if ( boost::filesystem::is_regular_file(p) )
			return true;
		else if ( boost::filesystem::is_directory(p) )
			return false;
		else
			return false;
	}
	else
		return false;
}


// Use boost to create directory if not existing
void createProjectDirectory( const char* dirname )
{
	boost::filesystem::path rootPath ( dirname );
	//boost::system::error_code returnedError;

	boost::filesystem::create_directories( rootPath );//, returnedError );
}


// Output status to terminal
void printIterationStatus()
{
	std::cout << "T = " << ctx->getParameters()->getTemperature() << ", ";
	std::cout << "Iteration = " << ctx->getNumIterations() << ", ";
	std::cout << "Res = " << ctx->getResidual(0) << ", ";
	std::cout << "Time = " << ctx->getComputeTime() << ", ";
	std::cout << "fluxQuanta = " << fluxQuanta << ", ";
	std::cout << "B = " << ctx->getParameters()->getMagneticFieldStrength() << endl;
}


// Append data to text file
int appendLineToTextFile( const char* filename, ostream& os,
						  ios_base::openmode mode = ios_base::app )
{
	ofstream outfile;

	stringstream oss;
	oss << os.rdbuf();
	outfile.open(filename, mode);
	outfile << oss.str() << endl; 
	outfile.close();

	return 0;
}


// Save free energy to text file
int saveFreeEnergy( const char* filename )
{
	// Add header if file does not exist
	if ( !boost::filesystem::exists( filename ) )
	{
		stringstream ss_header;
		ss_header << "# run res freeE fluxQuanta area internalLatticePoints";
		//ss_header << "# run res freeE";
		appendLineToTextFile( filename, ss_header );
	}

	// Append line with energy and other information
	stringstream ss;
	const streamsize prec = ss.precision();

	ss << numLoops << " ";
	ss << ctx->getResidual(0) << " ";
	ss.precision(10);
	ss << freeEnergyObj->getFreeEnergy_unitArea() << " ";
	ss.precision(prec);
	ss << int(fluxQuanta) << " ";
	ss << ctx->getGeometry()->area() << " ";
	ss << ctx->getGeometry()->area(1);
	appendLineToTextFile( filename, ss );

	return 0;
}


// Save params to text file as METADATA
int saveParamsMetaData( const char* filename )
{
	Parameters<T>* param = ctx->getParameters();

	// Add header
	stringstream ss_header;
	ss_header << "# R T pairing pairBreaking outerRadius reducedLength reducedEnergy ";
	ss_header << "solenoid geomShape areaCompute internalLatticePoints gauge ";
	ss_header << "B_external fluxQuanta Nx Na energyCutoff initDelta";
	appendLineToTextFile( filename, ss_header, ios_base::trunc );

	// Append line with energy and other information
	stringstream ss;
	ss << coherenceLengths << " ";
	ss << param->getTemperature() << " ";
	ss << "s" << " ";
	ss << "1" << " ";
	ss << outerRadius << " ";
	ss << "1" << " ";
	ss << "-1" << " ";
	ss << "0" << " ";
	ss << "square" << " ";
	ss << ctx->getGeometry()->area() << " ";
	ss << ctx->getGeometry()->area(1) << " ";
	//ss << latticeOrientation << " ";
	ss << "radial" << " ";
	ss << B_external << " ";
	ss << fluxQuanta << " ";
	ss << param->getGridResolutionBase() << " ";
	ss << param->getAngularResolution() << " ";
	ss << param->getCutOff() << " ";
	ss << "phaseWinding";
	appendLineToTextFile( filename, ss );

	return 0;
}


// Save observable(s) data to file
void saveDataFile( const int numIteration )
{
	std::cout << "Preparing to save data to file...\n";

	// Fetch run parameters from context
	Parameters<T>* param = ctx->getParameters();

	// Construct filenames
	char delta_file[512];
	char current_file[512];
	char delta_files[512];
	char current_files[512];

	sprintf(delta_file, "%s/delta.grd", runPath);
	sprintf(current_file, "%s/current.grd", runPath);
	sprintf(delta_files, "%s/delta_%d.grd", runPath, numIteration);
	sprintf(current_files, "%s/current_%d.grd", runPath, numIteration);

	// Save Order Parameter
	Grid<T>::GPU* delta_save = ctx->getComputeOrderParameter()->getResult();
	ctx->getGeometry()->zeroValuesOutsideDomain( delta_save );

	if ( delta_save->write(delta_file,param) )
	{
		cout << "Could not save Order Parameter to file." << endl;
		exit(0);
	}

	// Save Current
	Grid<T>::GPU* current_save = ctx->getComputeCurrent()->getResult();
	ctx->getGeometry()->zeroValuesOutsideDomain( current_save );

	if ( current_save->write(current_file,param) )
	{
		cout << "Could not save Current to file." << endl;
		exit(0);
	}

	// Save delta and current to file
	if ( numIteration > 0 )
	{
		if ( delta_save->write(delta_files,param) )
		{
			cout << "Could not save Order Parameter to file." << endl;
			exit(0);
		}
		if ( current_save->write(current_files,param) )
		{
			cout << "Could not save Current to file." << endl;
			exit(0);
		}
	}

	// Save metadata file
	char paramFile[512];
	sprintf(paramFile, "%s/params.dat", runPath);
	saveParamsMetaData(paramFile);
}

// Plot data with OpenGL
void plotData()
{
	// Extract computed data
	Grid<T>::CPU delta_cpu = *(ctx->getComputeOrderParameter()->getResult());
	Grid<T>::CPU j = *(ctx->getComputeCurrent()->getResult());

	// Plot stuff
	vis->clearImage();

	// Plot order parameter
	vis->plotImage( delta_cpu, 0, Type::real, 0, 1 );
	vis->plotImage( delta_cpu, 0, Type::imag, 1, 1 );
	vis->plotImage( delta_cpu.abs(), 0, Type::real, 2, 1 );

	vis->plotImage( j, 0, Type::real, 0, 0 );
	vis->plotImage( j, 1, Type::real, 1, 0 );
	vis->plotImage( j.fieldMagnitude(), 0, Type::real, 2, 0 );

	// Draw results
	vis->draw();
}

// Run a single kernel iteration.
void runIteration()
{
	numLoops++;

	// Update run parameters from file
	updateBfromFile();

	// Run kernel, solving equations for all compute objects
	ctx->compute();

	// Output status to terminal
	printIterationStatus();

	// Plot data in OpenGL
	if ( runVisual )
		plotData();
}


// Cleanup of context and visualization objects.
// Automatically called by OpenGL upon its completion.
void cleanup()
{
	std::cout << "Cleaning up...\n";
	
	delete vis;
	std::cout << "Cleaned vis object.\n";
	delete ctx;
	std::cout << "Cleaned ctx object.\n";

	std::cout << "Done cleaning up...\n";
}


// Wrapper for running program.
// Return true if program is done (converged or reach maximum number of iterations).
bool runWrapper()
{
	// Run single kernel iteration
	runIteration();

	// Check for convergence or maximum number of loops (can add user termination through OpenGL hook - check myKeyboardFunc)
	if (	(acc->isConverged() || numLoops > MAXLOOPS)
			&& numLoops > MINLOOPS )
	{
		Parameters<T>* param = ctx->getParameters();

		freeEnergyObj->setCompute(1);
		runIteration();
		freeEnergyObj->setCompute(0);

		// Save data to file
		if ( saveData )
		{
			saveDataFile(0);
			saveInducedB();

			char tempFilename[512];
			sprintf(tempFilename, "%s/freeEnergy.dat", runPath);
			saveFreeEnergy( tempFilename );
		}

		return true;
	} else {
		// Check if user has requested to save data
		if ( ( g_forceSave == 1 ) || 
			 ( saveData && ((!(numLoops % saveDataInterval)) && (numLoops > 0)) )
		   )
		{
			freeEnergyObj->setCompute(1);
			runIteration();
			freeEnergyObj->setCompute(0);

			// Save induced magnetic field to file
			if ( g_forceSave )
				saveInducedB();

			// Save data to file
			saveDataFile(0);

			char tempFilename[512];
			sprintf(tempFilename, "%s/freeEnergy.dat", runPath);
			saveFreeEnergy( tempFilename );

			g_forceSave = 0;
		}
	}

	return false;
}


// OpenGL display loop
void display()
{
	// Run one iteration (two if convergence)
	bool allConverged = runWrapper();

	if ( allConverged )
		exit(0);
}


// Terminal mode wrapper
void runInTerminal()
{
	bool allConverged = false;
	
	while ( !allConverged )
	{
		// Run one iteration (two if convergence)
		allConverged = runWrapper();
	}

	// Clean up context and visualization
	cleanup();
}


// Detect keypresses in OpenGL instance - useful for changing parameters live, saving data or quitting the program
void myKeyboardFunc( unsigned char key, int x, int y )
{
	cout << "key pressed: " << key << ", at (" << x << ", " << y << ")" << endl;
	if ( key == 's' )
		g_forceSave = 1;
}


// Read command line arguments
int readCommandLineArguments( int ___argc, char *const *___argv )
{
	int ___verbose = 0;

	if ( ___verbose )
		cout << "readCommandLineArguments(): " << ___argc << " number of arguments found." << endl;
	int numUpdates = 0;

	for ( int i = 0; i < ___argc; i++ )
	{
		if (i + 1 != ___argc) // Check that we haven't finished parsing already
		{
			string thisArg = ___argv[i];
			string nextArg = ___argv[i+1];

			if( thisArg == "--w" )
			{
				numUpdates += 1;
				if ( ___verbose )
					cout << "Updated fluxQuanta from command line arguments, from " << fluxQuanta << " to " << nextArg << endl;
				fluxQuanta = T(stod(nextArg));
			} else if( thisArg == "--T" ) {
				numUpdates += 1;
				if ( ___verbose )
					cout << "Updated T_Tc from command line arguments, from " << T_Tc << " to " << nextArg << endl;
				T_Tc = T(stod(nextArg));
			}
		}
	}

	if ( ___verbose )
		cout << "readCommandLineArguments(): " << numUpdates << " arguments updated." << endl;

	return 0;
}


// Save induced magnetic field and vector potential to file
void saveInducedB()
{
	Parameters<T>* param = ctx->getParameters();

	// Calculate induced B
	Grid<T>::GPU* b = new Grid<T>::GPU;
	char current_file[512];
	sprintf(current_file, "%s/current.grd", runPath);
	b->read(current_file, param);

	int N = b->dimX();

	GeometryGroup<T>* geom = ctx->getGeometry();

	const T twopi = 2.0*M_PI;
	const T h = param->getGridElementSize() * twopi;

	const T lambda = 100.0;
	const T c = -1.0 / (lambda*lambda);

	Grid<char>::GPU* dom = geom->createDomain(N);

	Grid<T>::GPU* A = new Grid<T>::GPU(N,N,2,Type::real);

	// Compute induced vector potential due to currents (Maxwell)
	computeVectorPotential(A, b, dom, h, c);

	// Apply arbitrary scaling factor
	Grid<T>::GPU B = A->curl(h) * 1.0E5;

	char A_file[512];
	char B_file[512];
	//char B_zero_file[512];
	//char divA_file[512];
	//char divJ_file[512];
	sprintf(A_file, "%s/A.grd", runPath);
	sprintf(B_file, "%s/B.grd", runPath);
	//sprintf(B_zero_file, "%s/B_zero.grd", runPath);
	//sprintf(divA_file, "%s/divA.grd", runPath);
	//sprintf(divJ_file, "%s/divJ.grd", runPath);

	A->write(A_file, param);
	B.write(B_file, param);

	//ctx->getGeometry()->zeroValuesOutsideDomain( & B );
	//B.write(B_zero_file, param);
	//A->div().write(divA_file, param);
	//b->div().write(divJ_file, param);

	delete b;
	delete A;
}


// Update parameters LIVE from data file - reads data file between each self-consistent iteration
void updateBfromFile( )
{
	string paramFile = "demo_hexagon_swave_VL.dat";
	if ( boost::filesystem::exists( paramFile ) )
	{
		Parameters<T>* param = ctx->getParameters();

		ifstream	fq_param_file( paramFile.c_str() );
		string		fq_line;
		while( getline(fq_param_file, fq_line) )
		{
			stringstream	fq_linestream(fq_line);
			string			fq_first;
			string			fq_second;
			T				fq_fluxQuanta;
			T				fq_T_Tc;
			int changes = 0;
			int newTemp = 0;

			fq_linestream >> fq_first >> fq_second;
			if( fq_first == "#" )
			{
				//cout << "  Run header:\t\"" << fef_line << "\"" << endl;
			} else {

				if( fq_first != "same" )
				{
					fq_fluxQuanta = stod(fq_first);
					if( abs(fq_fluxQuanta - fluxQuanta) > MYEPSILON )
					{
						cout << "Updated fluxQuanta from " << fluxQuanta << " to " << fq_fluxQuanta << endl;
						changes++;
						fluxQuanta = fq_fluxQuanta;
						B_external = fluxQuanta / area;
						param->setMagneticFieldStrength( B_external );
					}
				}

				if( fq_second != "same" )
				{
					fq_T_Tc = stod(fq_second);
					if( abs(fq_T_Tc - T_Tc) > MYEPSILON )
					{
						cout << "Updated T_Tc from " << T_Tc << " to " << fq_T_Tc << endl;
						changes++;
						T_Tc = fq_T_Tc;
						param->setTemperature( T_Tc );
						newTemp = 1;
					}
				}

				if( changes > 0 )
				{
					if( newTemp )
						ctx->initialize();
					else
						ctx->getIntegrationIterator()->setBoundaryInitialCondition();
				}
			}
		}
	} else {
		cout << "Tried updating params from file, ";
		cout << "but could not find demo_hexagon_swave_VL.dat" << endl;
	}
}
