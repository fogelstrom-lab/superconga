#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "PlotRegistration.h"

#include <Grid.cpp>
#include <Parameters.cpp>
//#include <Parameters_old.cu>
#include <Visualization.cu>

#ifdef __CUDACC__
#include <GeometryGroup.cu>
#include <GeometryComponent.cu>
#endif

typedef double T;
//typedef float T;

// Typedefs to simplify things
typedef typename Grid<T>::GPU GridGPU;
typedef typename Grid<T>::CPU GridCPU;

// Main grid to plot is stored here
GridCPU* g_grid = new GridCPU;

// Complex magnitude and field magnitude data (if applicable) is stored here
GridCPU* g_gridMod = 0;

// Complex magnitude and field magnitude data (if applicable) is stored here
GridCPU* g_gridPhase = 0;

// Auxiliary file to print to terminal. Array of length same as # of fields in g_grid
GridCPU* g_gridAux = 0;

// Generate spectral energy data (from parameters file) and store here. Prints to terminal if applicable.
GridCPU* g_energies = 0;

// Keep track of which grid, and what part, is plotted where.
PlotRegistration<GridCPU>* plots = 0;

Visualization<T>* vis = 0;
Parameters<T>* param = new Parameters<T>;

// Various global variables for the interactive controls and input arguments
int g_fieldComponents = 0, g_normalize = 1;
int g_plotType = -1;
int g_plotMarker = 0;
int g_plotPhase = 0;
int g_writePlotDataCount = 0;

// File sequence info
std::string filename = "";
int g_filenumMin = -1, g_filenumMax = -1;
int g_oneFile = 0;


// Functions for interactive control (definitions further down)
void myMouseFunc( int button, int state, int x, int y );
void myMotionFunc( int x, int y );
void myKeyboardFunc( unsigned char key, int x, int y );
void arrowKeyPress( int key, int x, int y );

std::string stripExtension( std::string filename )
{
	using namespace std;
	using namespace boost::filesystem;

	string filename_base;

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
	return filename_base;
}

// Main function
int main( int argc, char *argv[] )
{
	using namespace std;
	using namespace boost::filesystem;
	
	// Parse arguments
	//
	if ( argc == 1 )
	{
		cout << "Usage:\nviewgrid -f \033[1mfilename\033[00m [-min \033[1m#\033[00m] [-max \033[1m#\033[00m] [-c \033[1m#\033[00m] [-z \033[1m#\033[00m] [-aux \033[1mfilename\033[00m] [-plot]\n";
		cout << "\n--> Filename is specified without numbering and without .grd, e.g.\n";
		cout << "viewgrid -f file -min 0 -max 7\n will read file.0.grd through file.7.grd, and\n";
		cout << "viewgrid -f file.1\n will read file.1.grd\n\n";
		cout << "-c Field components. Specify -c 2 for currents, for example\n";
		cout << "-n Normalize modulus fields, 0 or 1\n";
		cout << "-z Zoom level, integer\n";
		cout << "-aux Auxiliary data file, print content to terminal\n";
		cout << "-phase Show complex phase instead of Re,Im\n";
		cout << "-plot Activate plotting function, no arg\n\n";
		cout << "In-view controls:\n";
		cout << "Left/Right arrows - Change fields\n";
		cout << "Up/Down arrows - Change scale of plot (when invoked by -plot)\n";
		cout << "LMB+drag - Changes displayed field\n";
		cout << "MMB+drag - Changes image intensity\n";
		cout << "SHIFT+LMB - Plots cross-section of field values at current coordinate (requires -plot flag)\n";
		cout << "CTRL+LMB - Plots sum of field values, for each field (requires -plot flag)\n";
		cout << "a - Toggles horizontal lock for cross-section plot\n";
		cout << "s - Toggles vertical lock for cross-section plot\n\n";
	}
	else if ( argc == 2 )
	{
		filename = argv[1];

/*		string file = filename;
		path myPath(filename);

		if ( myPath.has_extension() && myPath.extension() == ".grd" )
		{
			string stem = myPath.stem().string();
			string ppath = myPath.parent_path().string();

			if ( ppath == "" )
				filename = stem;
			else
				filename = ppath + "/" + stem;
		}
*/
	}
	
	string auxfile = "";
	int zoom = 1;

	for ( int i=1; i<argc; i+=2 )
	{
		string arg = argv[i];

		if ( arg == "-f" )
		{
			//cout << "f = " << argv[i+1] << endl;
			filename = argv[i+1];
		}
		else if ( arg == "-aux" )
		{
			auxfile = argv[i+1];
		}
		else if ( arg == "-min" )
		{
			//cout << "min = " << argv[i+1] << endl;
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> g_filenumMin;
		}
		else if ( arg == "-max" )
		{
			//cout << "max = " << argv[i+1] << endl;
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> g_filenumMax;		
		}
		else if ( arg == "-c" )
		{
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> g_fieldComponents;
		}
		else if ( arg == "-n" )
		{
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> g_normalize;
		}
		else if ( arg == "-z" )
		{
			stringstream sstream;
			sstream << argv[i+1];
			sstream >> zoom;				
		}
		else if ( arg == "-phase" )
		{
			g_plotPhase = 1;
			i--;
		}
		else if ( arg == "-plot" )
		{
			g_plotType = 0;
			g_plotMarker = 1;
			i--;
		}
	}
	
	if ( filename == "" )
	{
		cout << "No filename provided.\n";
		return 0;
	}

	filename = stripExtension( filename );
	
	// Read files into grid
	//
	int numFields;
	
	// One file
	if ( g_filenumMin == -1 )
	{
		string file = filename + ".grd";

		if ( g_grid->read( file.c_str(), param ) )
			exit(0);

		numFields = g_grid->numFields();

		if ( g_fieldComponents == 0 )
		{
			if ( numFields <= 3 && !(g_grid->isComplex()) )
				g_fieldComponents = numFields;
			else
				g_fieldComponents = 1;
		}

		// Recreate energies for LDOS and spectral current density
		if ( numFields > g_fieldComponents )
		{
			const T e_min = param->getEnergyMin();
			const T e_max = param->getEnergyMax();
			
			//cout << "E_min = " << e_min << ", E_max = " << e_max << ", E_num = " << param->getNumEnergies() << "\n";
			if ( e_min != e_max )
			{
				if ( param->getNumEnergies() == 0 ) param->setNumEnergies(numFields/g_fieldComponents);

				int e_num = param->getNumEnergies();
				
				g_energies = new GridCPU(e_num,1,1,Type::real);
				T* e_ptr = g_energies->getDataPointer();

				for ( int i=0; i<e_num; ++i )
					e_ptr[i] = e_min + (e_max-e_min)*(T(i)/T(e_num-1));
			}
		}
		g_oneFile = 1;
	}
	else // Multiple files in a sequence
	{
		GridCPU tmp;

		// Figure out how many fields to allocate memory for
		if ( tmp.read( filename.c_str(), g_filenumMin, param ) )
			exit(0);

		// How many files?
		if ( g_filenumMax == -1 )
		{
			int f = g_filenumMin;
			while (1)
			{
				path p_tst( filename + "." + boost::lexical_cast<std::string>(f) + ".grd" );
				if ( !exists(p_tst) )
				{
					f--;
					break;
				}
				f++;
			}
			g_filenumMax = f;
		}

		// Allocate new grid to hold all files
		numFields = tmp.numFields() * (g_filenumMax-g_filenumMin+1);

		if ( tmp.numFields() <= 3 )
			g_fieldComponents = tmp.numFields();

		// Allocate grid
		delete g_grid;
		g_grid = new GridCPU( tmp, numFields );

		int field = 0;
	
		for ( int i=g_filenumMin; i<=g_filenumMax; ++i )
		{
			if ( tmp.read( filename.c_str(), i ) )
				exit(0);

			for ( int c=0; c<tmp.numFields(); ++c )
			{
				tmp.copyFieldTo( g_grid, c, field );
				field++;
			}
		}
	}
	

	// Read auxiliary data, if specified
	//
	if ( auxfile != "" )
	{
		g_gridAux = new GridCPU;

		if ( auxfile.find(".grd") < auxfile.length() )
			g_gridAux->read( auxfile.c_str() );
		else
			g_gridAux->readNumpyArray( auxfile.c_str(), -1 );

		if ( g_gridAux->dimX() != g_grid->numFields() )
			cout << "Warning, entries in auxiliary grid does not match input grid # field components\n";
	}


	// Create additional data
	if ( g_fieldComponents > 1 )
	{
		int fields = g_grid->numFields() / g_fieldComponents;
		g_gridMod = new GridCPU( *g_grid, fields );

		for ( int i=0; i<fields; ++i )
		{
			GridCPU tmp( *g_grid, 1 );
			tmp.setZero();
			
			for ( int c=0; c<g_fieldComponents; ++c )
			{
				//cout << g_fieldComponents*i + c << endl;
				GridCPU tmp2 = (*g_grid)[g_fieldComponents*i + c];
				tmp += tmp2*tmp2;
			}
			
			tmp.sqrt();
			tmp.copyFieldTo( g_gridMod, 0, i );
		}		
	}
	else if ( g_grid->isComplex() )
	{
		g_gridMod = new GridCPU( g_grid->dimX(), g_grid->dimY(), numFields, Type::real );

		g_grid->complexMagnitudeSq( g_gridMod );
		g_gridMod->sqrt();

		g_gridPhase = new GridCPU( g_grid->dimX(), g_grid->dimY(), numFields, Type::real );
		g_grid->complexPhase( g_gridPhase );
	}
	
	// Init visualization to griddata
	//
	int nImg = (g_grid->isComplex()+1) * g_fieldComponents;
	if ( nImg > 1 && !g_plotPhase)
		nImg++;
	
	vis = new Visualization<T>( g_grid, nImg+int(g_plotType>-1), 1 );
	vis->setZoom( zoom );

	if ( param->isValid() )	
		vis->setPrintParameters( param );
	else
		cout << "No parameters provided with file\n";

	// Create object to keep track of plots
	int numImages = vis->getNumImagesX() * vis->getNumImagesY();
	plots = new PlotRegistration<GridCPU>( numImages );

	// Print some info
	if ( param && param->isValid() )
		param->print();

	g_grid->printDimensions();

	// OpenGL display loop
	//
	initGL( argc, argv, vis->getNumPixelsWindowX(), vis->getNumPixelsWindowY() );

	glutMouseFunc( myMouseFunc );
	glutMotionFunc( myMotionFunc );
	glutKeyboardFunc( myKeyboardFunc );
	glutSpecialFunc( arrowKeyPress );

	glutMainLoop();
	
	return 0;
}


// Global variables for interactive management
int g_buttonPressed;//, g_requestField = 0;
char g_keyPressed = '\0';

float g_imgMult = 1.0;
float g_imgMult_press;

float g_plotMult = 1.0;

int g_currentField = -1;
int g_requestField = 0;

int g_slice_dir = 0;

int g_fieldHasChanged = 1, g_displayMultHasChanged = 0;
int g_plotHasChanged = 0, g_plotDataHasChanged = 0, g_plotDataMultHasChanged = 0;

int g_currentField_press, g_coord_press;
int g_button_state, g_active_modifier;

int g_coord_x, g_coord_y;
int g_coord_down_x, g_coord_down_y, g_coord_up_x, g_coord_up_y;
int g_coordPlot_x, g_coordPlot_y;

GridCPU* g_x_data = 0;
GridCPU* g_y_data = 0;

void printMessage()
{
	using namespace std;
	
	cout << "\rDisplaying field: " << g_requestField << " ";

	// Load parameters for each grid, if we are displaying a sequence of grids
	if ( g_oneFile == 0 )
	{
		GridCPU::readParameters( filename.c_str(), g_requestField, param );
	}

	if ( g_oneFile && g_requestField < param->getNumEnergies() )
	{
		if ( g_energies )
			cout << "(E = " << (g_energies->getDataPointer())[g_requestField]  << ") ";
	}

	if ( g_gridAux != 0 )
	{
		int Nf = g_gridAux->numFields();
		T* ptr;
		
		cout << "; Aux:";
		for ( int c=0; c<Nf; ++c ) 
		{
			ptr = g_gridAux->getDataPointer(c,Type::real);
			cout << " [" << c << "]: " << ptr[g_requestField];
			if ( c < Nf-1 )
				cout << ", ";
		}
	}

	cout << "     " << flush;
}


void genDataSum( GridCPU* plotGrid, const int field, const int x, const int y )
{
	const int fieldComponent = field % g_fieldComponents;
	const int N = plotGrid->numFields() / g_fieldComponents;

	T *x_ptr = g_x_data->getDataPointer();
	T *y_ptr = g_y_data->getDataPointer();

	int c;
	T re, im;

	for ( int i=0; i<N; ++i )
	{
		c = i*g_fieldComponents + fieldComponent;
		plotGrid->sumField(re,im,c);

		x_ptr[i] = i;
		y_ptr[i] = re;
	}

	if ( param->getAngularResolution() )
	{
		int Nx = param->getGridResolutionBase();
		*g_y_data *= 1.0 / double(Nx*Nx);
	}
}


void genDataCrossSection( GridCPU* plotGrid, const int field, const int x, const int y )
{
	const int Nx_input = plotGrid->dimX();
	const int idx = y*Nx_input + x;
	const int fieldComponent = field % g_fieldComponents;
	const int N = plotGrid->numFields() / g_fieldComponents;

	int c;
	T *x_ptr = g_x_data->getDataPointer();
	T *y_ptr = g_y_data->getDataPointer();

	for ( int i=0; i<N; ++i )
	{
		c = i*g_fieldComponents + fieldComponent;

		x_ptr[i] = i;
		y_ptr[i] = (plotGrid->getDataPointer(c,Type::real))[idx];
	}
}


void plot( void(*genDataFunc)(GridCPU*, const int, const int, const int) )
{
	if ( g_plotHasChanged )
	{
		int img_x, img_y, x, y;
		vis->getImageCoordinatesFromWindow( g_coordPlot_x, g_coordPlot_y, img_x, img_y, x, y );

		if ( img_x < vis->getNumImagesX()-1 && img_y < vis->getNumImagesY() )
		{
			GridCPU* plotGrid;
			Type::type representation;
			int field;

			const int img_idx = img_y * vis->getNumImagesX() + img_x;
			plots->getPlotInfo( plotGrid, representation, field, img_idx );

			const int plot_img_x = vis->getNumImagesX() - 1;
			const int plot_img_y = 0;
			const int N = plotGrid->numFields() / g_fieldComponents;

			if ( GridCPU::initializeGrid( g_x_data, N,1,1,Type::real ) )
				g_x_data->setZero();
			if ( GridCPU::initializeGrid( g_y_data, N,1,1,Type::real ) )
				g_y_data->setZero();

			if ( g_plotDataHasChanged )
			{
				genDataFunc(plotGrid,field,x,y);
				g_plotDataHasChanged = 0;
			}

			vis->clearImage( plot_img_x, plot_img_y );

			if ( g_energies )
			{
				GridCPU fermiFunc(*g_x_data);
				T* f_ptr = fermiFunc.getDataPointer();
				T* x_ptr = g_energies->getDataPointer();

				const T T_Tc = param->getTemperature();

				for ( int i=0; i<N; ++i )
					f_ptr[i] = (T)0.5 / (exp(x_ptr[i]/T_Tc) + (T)1.0);

				vis->plotGraph( g_energies, g_y_data, 0.4, 0.4, 0.4, plot_img_x, plot_img_y, g_plotMult );
				vis->plotGraph( g_energies, &fermiFunc, 0.0, 0.15, 0.15, plot_img_x, plot_img_y );
				vis->plotVerticalMarker( g_energies, 0.0, 1.0, 0.0, 0.0, plot_img_x, plot_img_y );
				vis->plotVerticalMarker( g_energies, -2.0, 1.0, 1.0, 0.0, plot_img_x, plot_img_y );
				vis->plotVerticalMarker( g_energies, 2.0, 1.0, 1.0, 0.0, plot_img_x, plot_img_y );
				
				// Update field indicator
				vis->plotVerticalMarker( g_energies, x_ptr[g_currentField], 0.3, 0.3, 0.3, plot_img_x, plot_img_y );
			}
			else
				vis->plotGraph( g_x_data, g_y_data, 0.5, 0.5, 0.5, plot_img_x, plot_img_y, g_plotMult );

			g_plotHasChanged = 0;
		}
	}
}


void plotCrossSection_slice()
{
	if ( g_plotHasChanged )
	{
		int img_x, img_y, x, y;
		vis->getImageCoordinatesFromWindow( g_coordPlot_x, g_coordPlot_y, img_x, img_y, x, y );

		if ( img_x < vis->getNumImagesX()-1 && img_y < vis->getNumImagesY() )
		{
			GridCPU* plotGrid = 0;
			Type::type representation;
			int field;

			int img_idx = img_y * vis->getNumImagesX() + img_x;
			plots->getPlotInfo( plotGrid, representation, field, img_idx );

			int plot_img_x = vis->getNumImagesX() - 1;
			int plot_img_y = 0;
			
			int Ne = plotGrid->numFields() / g_fieldComponents;
			int Nx = g_grid->dimX();

			if ( GridCPU::initializeGrid( g_x_data, Ne,1,1,Type::real ) )
				g_x_data->setZero();
			if ( GridCPU::initializeGrid( g_y_data, Ne,Nx,1,Type::real ) )
				g_y_data->setZero();

			T *x_ptr = g_x_data->getDataPointer();
			T *y_ptr = g_y_data->getDataPointer();

			if ( g_plotDataHasChanged )
			{
				const int Nx_input = plotGrid->dimX();
				const int fieldComponent = field % g_fieldComponents;

				int c, idx, idx_plot;

				for ( int i=0; i<Ne; ++i )
				{
					c = i*g_fieldComponents + fieldComponent;
					
					for ( int j=0; j<Nx; ++j )
					{
						if ( g_slice_dir == 0 )
							idx = y*Nx_input + j;
						else
							idx = j*Nx_input + x;

						idx_plot = j*Ne + i;

						x_ptr[i] = i;
						y_ptr[idx_plot] = (plotGrid->getDataPointer(c,Type::real))[idx];
					}
				}

				g_plotDataHasChanged = 0;
			}

			vis->clearImage( plot_img_x, plot_img_y );

			if ( g_energies )
			{
				GridCPU tst( *g_y_data );
				tst.resample( g_grid->dimX(), g_grid->dimY() );

				Color::RedBlueWhite rb;
				Color::Base* old = vis->getColorScheme();

				if ( g_fieldComponents > 1 )
					vis->setColorScheme( &rb );

				vis->drawField( &tst, 0, plot_img_x, plot_img_y, Type::real, g_normalize );
				
				if ( g_fieldComponents > 1 )
					vis->setColorScheme( old );
			
				// Update field indicator
				T* e_ptr = g_energies->getDataPointer(0,Type::real);
				vis->plotVerticalMarker( g_energies, e_ptr[g_currentField], 0.3, 0.3, 0.3, plot_img_x, plot_img_y );
			}
			//else
			//	vis->plot( g_x_data, g_y_data, 0.5, 0.5, 0.5, plot_img_x, plot_img_y, g_plotMult );

			g_plotHasChanged = 0;
		}
	}
}


void plotMarker( int x, int y, int size )
{
	int win_x = vis->getNumPixelsWindowX();
	int win_y = vis->getNumPixelsWindowY();

	float xf = 2.0f*float(x)/(win_x-1) - 1.0f;
	float yf = 1.0f - 2.0f*(float(y)/(win_y-1));

	glPointSize(4.0f);
	
	glBegin(GL_POINTS);
	glColor3f(0.0f, 0.0f, 0.0f);
	glVertex2f(xf, yf);
	glEnd();

	glPointSize(2.0f);
	
	glBegin(GL_POINTS);
	glColor3f(1.0f, 1.0f, 1.0f);
	glVertex2f(xf, yf);
	glEnd();
}


void plotSlice( int x, int y, int dir )
{
	int win_x = vis->getNumPixelsWindowX();
	int win_y = vis->getNumPixelsWindowY();

	int img_x, img_y, grid_x, grid_y, num_img_x, num_img_y;
	vis->getImageCoordinatesFromWindow( x, y, img_x, img_y, grid_x, grid_y );
	num_img_x = vis->getNumImagesX();
	num_img_y = vis->getNumImagesY();

	float x_min = float(img_x) / float(num_img_x) * 2.0f - 1.0f;
	float x_max = x_min + 2.0f / float(num_img_x);

	float y_min = float(img_y) / float(num_img_y) * 2.0f - 1.0f;
	float y_max = y_min + 2.0f / float(num_img_y);

	float xf = 2.0f*float(x)/(win_x-1) - 1.0f;
	float yf = 1.0f - 2.0f*(float(y)/(win_y-1));

	//glPointSize(4.0f);
	
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.5f, 0.5f);
	if ( dir == 0 )
	{
		glVertex2f(x_min, yf);
		glVertex2f(x_max, yf);
	}
	else
	{
		glVertex2f(xf, y_min);
		glVertex2f(xf, y_max);
	}
	glEnd();
}


void drawFields()
{
	if ( g_fieldHasChanged || g_displayMultHasChanged )
	{
		if ( !g_grid->isComplex() )
		{
			int field = g_requestField * g_fieldComponents;
	
			if ( g_fieldComponents == 1 )
			{
				vis->drawField( g_grid, g_requestField, 0, 0, Type::real, g_normalize );
				plots->registerPlot( g_grid, Type::real, g_requestField, 0 );
			}
			else
			{
				for ( int i=0; i<g_fieldComponents; ++i )
				{
					vis->drawField( g_grid, field+i, i, 0, Type::real );
					plots->registerPlot( g_grid, Type::real, field+i, i );
				}
				if ( g_gridMod )
				{
					vis->drawField( g_gridMod, g_requestField, g_fieldComponents, 0, Type::real, g_normalize );
					plots->registerPlot( g_gridMod, Type::real, g_requestField, g_fieldComponents );
				}
			}
		}
		else
		{
			if ( g_plotPhase )
			{
				vis->drawField( g_gridPhase, g_requestField, 0, 0, Type::real, g_normalize );
				plots->registerPlot( g_gridPhase, Type::real, g_requestField, 0 );
			}
			else
			{
				vis->drawField( g_grid, g_requestField, 0, 0, Type::real, g_normalize );
				vis->drawField( g_grid, g_requestField, 1, 0, Type::imag, g_normalize );

				plots->registerPlot( g_grid, Type::real, g_requestField, 0 );
				plots->registerPlot( g_grid, Type::imag, g_requestField, 1 );
			}
			if ( g_gridMod )
			{
				plots->registerPlot( g_gridMod, Type::real, g_requestField, 2-g_plotPhase );
				vis->drawField( g_gridMod, g_requestField, 2-g_plotPhase, 0, Type::real, g_normalize );
			}
		}

		g_currentField = g_requestField;
		g_fieldHasChanged = 0;
		g_displayMultHasChanged = 0;
	}
}


void generatePixelBuffer()
{
	drawFields();

	if ( g_plotType == 1 )
		plot( genDataCrossSection ); //plotCrossSection();
	if ( g_plotType == 2 )
		plot( genDataSum );  //plotSum();
	if ( g_plotType == 3 )
		plotCrossSection_slice();
}


void display()
{
	generatePixelBuffer();
	
	// Display image
	vis->drawPixels();
	
	if ( g_plotMarker == 1 )
	{
		if ( g_plotType == 3 )
			plotSlice( g_coordPlot_x, g_coordPlot_y, g_slice_dir );
		else
			plotMarker( g_coordPlot_x, g_coordPlot_y, 1 );
	}

	vis->drawInfo();
	glutSwapBuffers();
	printMessage();
}


// OpenGL stuff for interactive visualization
//
void setViewerVariables( int x, int y )
{
	if ( g_active_modifier == GLUT_ACTIVE_SHIFT || g_active_modifier == GLUT_ACTIVE_CTRL )
	{
		if ( g_plotType > -1 )
		{
			// Plotting window varibles
			g_coordPlot_x = x;
			g_coordPlot_y = y;

			if ( g_button_state == 0 )
			{
				g_plotHasChanged = 1;
				g_plotDataHasChanged = 1;
			}

			if ( g_active_modifier == GLUT_ACTIVE_SHIFT )
			{
				if ( g_buttonPressed == 0 )
					g_plotType = 1;
				else if ( g_buttonPressed == 1 )
					g_plotType = 3;

				if ( g_keyPressed == 'A' || g_keyPressed == 'a' )
					g_coordPlot_y = g_coord_down_y;

				if ( g_keyPressed == 'S' || g_keyPressed == 's' )
					g_coordPlot_x = g_coord_down_x;

			}
			if ( g_active_modifier == GLUT_ACTIVE_CTRL )
				g_plotType = 2;
		}
	}
	else if ( g_button_state == 0 )
	{
		// Field interaction variables
		if ( g_buttonPressed == 0 )
		{
			//int numFields = vis->getGrid()->getNumFieldComponents() / g_fieldComponents;
			int numFields = g_grid->numFields() / g_fieldComponents;
	
			g_requestField = g_currentField_press + 0.2*(x - g_coord_press);
	
			if ( g_requestField < 0 )
				g_requestField = 0;

			if ( g_requestField >= numFields )
				g_requestField = numFields - 1;
		
			g_fieldHasChanged = 1;	// Signal to update field image
			g_plotHasChanged = 1;	// Signal to update plot field marker
		}
		else
		{
			g_imgMult = g_imgMult_press + 0.01*(x - g_coord_press);
		
			if ( g_imgMult < 0.0f ) g_imgMult = 0.0f;
		
			vis->setDisplayMult( g_imgMult );
			g_displayMultHasChanged = 1;
		}
	}
	
	glutPostRedisplay();
}


void myMouseFunc( int button, int state, int x, int y )
{
	using namespace std;
	
	g_buttonPressed = button;
	g_button_state = state;
	g_active_modifier = glutGetModifiers();

	g_coord_x = x;
	g_coord_y = y;
	
	if ( state == 0 )
	{
		g_coord_down_x = x;
		g_coord_down_y = y;
	}
	else
	{
		g_coord_up_x = x;
		g_coord_up_y = y;		
	}

	// Record mouse down
	if ( state == 0 )
	{
		g_coord_press = x;
		g_currentField_press = g_currentField;
		g_imgMult_press = g_imgMult;
	}

	setViewerVariables(x,y);
}


void myMotionFunc( int x, int y )
{
	g_coord_x = x;
	g_coord_y = y;
	
	setViewerVariables( x, y );
}


void myKeyboardFunc( unsigned char key, int x, int y )
{
	using namespace std;

	if ( key == 'A' || key == 'a'  )
	{
		if ( g_keyPressed == 'A' || g_keyPressed == 'a' )
			g_keyPressed = '\0';
		else
		{
			g_keyPressed = key;
			g_coord_down_y = y;
		}
	}
	
	if ( key == 'S' || key == 's'  )
	{
		if ( g_keyPressed == 'S' || g_keyPressed == 's' )
			g_keyPressed = '\0';
		else
		{
			g_keyPressed = key;
			g_coord_down_x = x;
		}
	}

	if ( key == 't' )
	{
		g_slice_dir = int(!g_slice_dir);
		g_plotHasChanged = 1;
		g_plotDataHasChanged = 1;
		glutPostRedisplay();
		//cout << "g_slice_dir = " << g_slice_dir << "\n";
	}

	if ( key == 'W' )
	{
		// Create dir for saving data in
		using namespace boost::filesystem;

		path myPath(filename);
		// path myDir = myPath.parent_path();
		path myFile = myPath.filename();
		path fullPath;

		if ( g_filenumMin != -1 || g_filenumMax != -1 )
		{
			stringstream s;
			string num;
			s << g_filenumMin;
			s >> num;
			fullPath = canonical(filename+"."+num+".grd");
		}
		else
			fullPath = canonical(filename+".grd");


		//string plotdir = myDir.string() + "/extracted_data";
		string plotdir = fullPath.parent_path().string() + "/extracted_data";

		// cout << myPath << endl;
		// cout << myDir << endl;
		// cout << myFile << endl;
		// cout << current_path() << endl;
		// cout << canonical(filename+".grd") << endl;
		// cout << fullPath.parent_path() << endl;
		// cout << plotdir << endl;

		boost::filesystem::create_directories(plotdir);

		// Create filename
		int img_x, img_y, coord_x, coord_y;
		vis->getImageCoordinatesFromWindow( g_coordPlot_x, g_coordPlot_y, img_x, img_y, coord_x, coord_y );
	
		//GridData<T>* x_data = param->getEnergies();
		//T energyLevel = (x_data->getDataPointerCPU())[g_currentField];

		stringstream sstream;
		string dumpfile;

		sstream << plotdir + "/" + myFile.string() << "_field_" << g_currentField << ".txt";
		sstream >> dumpfile;	

//		cout << dumpfile << "\n";

		GridCPU dumpGrid( *g_grid, g_fieldComponents );

		for ( int i=0; i<g_fieldComponents; ++i )
		{
			int copyField = g_fieldComponents*g_currentField;
			g_grid->copyFieldTo( &dumpGrid, copyField+i, i );
		}

		dumpGrid.writeNumpyArray( dumpfile.c_str() );

	}


	// Save contents of plot window
	if ( key == 'P' && g_plotType != -1 )
	{
		// Create dir for saving data in
		using namespace boost::filesystem;

		path myPath(filename);
		path myFile = myPath.filename();
		path fullPath = canonical(filename+".grd");

		string plotdir = fullPath.parent_path().string() + "/extracted_data";

		boost::filesystem::create_directories(plotdir);

		// Create filename
		int img_x, img_y, coord_x, coord_y;
		vis->getImageCoordinatesFromWindow( g_coordPlot_x, g_coordPlot_y, img_x, img_y, coord_x, coord_y );

		stringstream sstream;
		string dumpfile;

		if ( g_plotType == 1 )
			sstream << plotdir + "/" + myFile.string() << "_plotdata_" << coord_x << "_" << coord_y << ".txt";
		else if ( g_plotType == 2 )
			sstream << plotdir + "/" + myFile.string() << "_plotdata_sum.txt";
		else if ( g_plotType == 3 )
		{
			string slicedir;
			int coord;
			if ( g_slice_dir == 0 )
			{
				slicedir = "x";
				coord = coord_y;
			}
			else
			{
				slicedir = "y";
				coord = coord_x;
			}
			sstream << plotdir + "/" + myFile.string() << "_slicedata_c" << img_x << "_" << slicedir << "_" << coord;
		}

//		cout << dumpfile << "\n";

//		GridData<T>* x_data = param->getEnergies();
		sstream >> dumpfile;

		// Local value per energy and sum per energy
		if ( g_plotType == 1 || g_plotType == 2 )
		{
			GridCPU xy_data( *g_y_data, 2 );
			g_energies->copyFieldTo( &xy_data, 0, 0 );
			g_y_data->copyFieldTo( &xy_data, 0, 1 );
			xy_data.writeNumpyArray( dumpfile.c_str() );
		}
		// Slice cross section
		else if ( g_plotType == 3 )
		{
			string energyfile, datafile;
			energyfile = dumpfile + "_energies.txt";
			datafile = dumpfile + "_data.txt";
			g_energies->writeNumpyArray( energyfile.c_str(), Type::real );
			g_y_data->writeNumpyArray( datafile.c_str() );
		}
	}
}


void arrowKeyPress( int key, int x, int y )
{
	using namespace std;
	
	//int numFields = vis->getGrid()->getNumFieldComponents() / g_fieldComponents;
	int numFields = g_grid->numFields() / g_fieldComponents;

	if ( key == GLUT_KEY_LEFT )
	{
		g_requestField = max(0,g_currentField-1);
		g_fieldHasChanged = 1;
	}
	
	if ( key == GLUT_KEY_RIGHT )
	{
		g_requestField = min(numFields-1,g_currentField+1);
		g_fieldHasChanged = 1;
	}

	if ( key == GLUT_KEY_UP )
	{
		g_plotMult *= 0.9;
		g_plotHasChanged = 1;
		//cout << "key up\n";
		//g_plotDataMultHasChanged = 1;
	}

	if ( key == GLUT_KEY_DOWN )
	{
		g_plotMult *= 1.1;
		g_plotHasChanged = 1;
		//cout << "key down\n";
		//g_plotDataMultHasChanged = 1;
	}

	glutPostRedisplay();
}


void cleanup()
{
	std::cout << "\nGL cleanup()\n";
	
	if ( g_grid ) delete g_grid;
	if ( g_gridMod ) delete g_gridMod;
	if ( g_gridPhase ) delete g_gridPhase;
	if ( g_gridAux ) delete g_gridAux;
	if ( g_energies ) delete g_energies;

	if ( g_x_data ) delete g_x_data;
	if ( g_y_data ) delete g_y_data;

	if ( plots ) delete plots;

	if ( vis ) delete vis;
}

