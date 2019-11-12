#ifndef _VISUALIZATION_H
#define _VISUALIZATION_H

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

#include <Parameters.h>
#include <Grid.h>
#include <Colormap.h>
#include "IncludeGL.h"
#ifdef __CUDACC__
#include <GeometryGroup.h>
#endif


template <typename T>
class Visualization
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

	// template typename GRID can be either GridCPU or GridGPU.

public:
			Visualization();
			~Visualization();

			template <typename GRID>
			Visualization( const GRID* grid, const int numImageX = 1, const int numImageY = 1 );
			Visualization( const int Nx, const int Ny, const int numImageX = 1, const int numImageY = 1 );
	
	void 	constructor();

	void	setZoom( int zoom );
	void	setDisplayMult( const float displayMult ) { _displayMult = displayMult; }
	void	setPrintParameters( Parameters<T>* param ) { _param = param; }

	Color::Base *	getColorScheme() { return &_colorScheme; }
	void			setColorScheme( Color::Base* colorScheme );
	void			setColorScheme( const Color::Base& colorScheme );

	// Initialize pixel buffer. num_img = 3 means allocate memory for 3 images -
	// i.e. one each for real, imag, and abs parts.
	void	allocMemImageGrid( const int num_image_X, const int num_image_Y );

	template <typename GRID>
	void	drawField(	const GRID* grid, int field, int image_idx_x, int image_idx_y, Type::type representation,
						int normalize = 1, T minVal = 0.0, T maxVal = 1.0 );

	template <typename GRID>
	void 	plotImage( const GRID& grid, int field, Type::type representation, int img_x, int img_y,
							int normalize = 1, T minVal = 0.0, T maxVal = 1.0 );

	void 	label( std::string info, int img_x, int img_y );

	void 	plotGraph( GridCPU* xdata, GridCPU* ydata, float r, float g, float b, int image_idx_x, int image_idx_y, float y_mult = 1.0 );
	void 	plotVerticalMarker( GridCPU* xdata, T x_coord, float r, float g, float b, int image_idx_x, int image_idx_y );
	
	void	eraseValuesOutsideDomain( const int image_idx_x, const int image_idx_y = 0 );


	// Drawing calls
	void	drawPixels();
	void	drawInfo();
	void	draw();
	void	clearImage( int img_x = -1, int img_y = -1 );

	// Getters
	int		getNumImagesX() const { return _num_image_X; }
	int		getNumImagesY() const { return _num_image_Y; }

	int		getBorderX() const { return _border_x; }
	int		getBorderY() const { return _border_y; }
	int		getBorderTop() const { return _border_top_y; }

	// Resolution
	int		getNumPixelsImageX() const { return _Nx_img; }
	int		getNumPixelsImageY() const { return _Ny_img; }

	int		getNumPixelsImagesX() const { return (_Nx_img*_displayZoom + 2*_border_x) * _num_image_X; }
	int		getNumPixelsImagesY() const { return (_Ny_img*_displayZoom + 2*_border_y) * _num_image_Y; }

	int		getNumPixelsWindowX() const { return (_Nx_img*_displayZoom + 2*_border_x) * _num_image_X; }
	int		getNumPixelsWindowY() const { return (_Ny_img*_displayZoom + 2*_border_y) * _num_image_Y + _border_top_y; }

	// Image interaction
	void	getImageCoordinatesFromWindow( int x, int y, int& img_x, int& img_y, int& grid_x, int& grid_y );

	// Helpers
	int		getIndexToRow( const int row, const int image = 0 ) const;
	int		getImageIndexOffset( int image_idx_x, int image_idx_y ) const;
	int		getRowIndexOffset() const;

	void	printRange( int img_row, int img_col, T minVal, T maxVal );
	void	printRangeToImage( int img_row, int img_col, T minVal, T maxVal );
	void	printParametersToImage( const Parameters<T> * param );
	void	printParametersToImage();

#ifdef __CUDACC__
	void	setGeometry( GeometryGroup<T>* geom );
#endif

private:
	// Size of inpute grid
	int 	_Nx_grid;
	int 	_Ny_grid;

	// Pixels per subimage
	int		_Nx_img;
	int		_Ny_img;

	// Entire displayed image
	int		_Nx_window;
	int		_Ny_window;

	// Border pixel size
	int		_border_top_y;
	int		_border_x;
	int		_border_y;

	// Subimages that make up whole image
	int		_num_image;
	int		_num_image_X;
	int		_num_image_Y;
	
	int		_numPixelsAllocated;
	float*	_image;

	int 	_currentDrawnField;

	int		_displayZoom;
	float	_displayMult;

	// Print parameters and range
	Parameters<T>*	_param;
	int				_valImgUsed[64];
	float			_minValImg[64];
	float			_maxValImg[64];

	std::string		_textImg[64];

	// Other
	Color::Base		_colorScheme;

#ifdef __CUDACC__	
	GeometryGroup<T>*	_geom;
	Grid<char>::CPU*	_geom_mask;
#endif
};

#endif

