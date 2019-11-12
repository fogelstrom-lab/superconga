#include "Visualization.h"

template <typename T>
Visualization<T>::Visualization()
	:	_image(0), _colorScheme(0), _displayZoom(1),
		_num_image(0), _border_x(0), _border_y(0), _border_top_y(0), _currentDrawnField(-1), _displayMult(1.0)
{
	for ( int i=0; i<64; ++i )
	{
		_valImgUsed[i] = 0;
		_textImg[i] = "";
	}

#ifdef __CUDACC__
	_geom = 0;
	_geom_mask = 0;
#endif
}


template <typename T>
Visualization<T>::~Visualization()
{
	if ( _image ) delete[] _image;
#ifdef __CUDACC__
	if ( _geom_mask ) delete _geom_mask;
#endif
}


template <typename T>
template <typename GRID>
Visualization<T>::Visualization( const GRID* grid, const int numImageX, const int numImageY )
	:	_image(0), _numPixelsAllocated(0), _param(0), _Nx_grid(grid->dimX()), _Ny_grid(grid->dimY()),
		_num_image_X(numImageX), _num_image_Y(numImageY), _num_image(numImageX*numImageY),
		_border_top_y(32), _border_x(16), _border_y(16), _displayZoom(1), _displayMult(1.0),
		_currentDrawnField(-1)
{
	constructor();
}


template <typename T>
Visualization<T>::Visualization( const int Nx, const int Ny, const int numImageX, const int numImageY )
	:	_image(0), _numPixelsAllocated(0), _param(0), _Nx_grid(Nx), _Ny_grid(Ny),
		_num_image_X(numImageX), _num_image_Y(numImageY), _num_image(numImageX*numImageY),
		_border_top_y(32), _border_x(16), _border_y(16), _displayZoom(1), _displayMult(1.0),
		_currentDrawnField(-1)
{
	constructor();
}


template <typename T>
void Visualization<T>::constructor()
{
	allocMemImageGrid( _num_image_X, _num_image_Y );
	
	_colorScheme = Color::Jet();
	
	for ( int i=0; i<64; ++i )
	{
		_valImgUsed[i] = 0;
		_textImg[i] = "";
	}

#ifdef __CUDACC__
	_geom = 0;
	_geom_mask = 0;
#endif
}


template <typename T>
void Visualization<T>::allocMemImageGrid( const int num_image_X, const int num_image_Y )
{
	int Nx = _Nx_grid;
	int Ny = _Ny_grid;
	
	if ( (Nx%4) != 0 )
		Nx += 4 - (Nx%4);	// glDrawPixels want sizes to be evenly divisible by 4
	
	if ( (Ny%4) != 0 )
		Ny += 4 - (Ny%4);

	_Nx_img = Nx;
	_Ny_img = Ny;
	
	_num_image_X = num_image_X;
	_num_image_Y = num_image_Y;
	_num_image = num_image_X * num_image_Y;

	_Nx_window = getNumPixelsWindowX();
	_Ny_window = getNumPixelsWindowY();

	int numPixels = _Nx_window * _Ny_window;
	
	if ( numPixels != _numPixelsAllocated )
	{
		if (_image) delete[] _image;
	
		_image = new float[numPixels*3];
		_numPixelsAllocated = numPixels;
	}	
}


template <typename T>
void Visualization<T>::getImageCoordinatesFromWindow( int x, int y, int& img_x, int& img_y, int& grid_x, int& grid_y )
{
	// ADD ZOOM!

	int bx = getBorderX();
	int by = getBorderY();
	int bt = getBorderTop();

	img_x = x / (2*bx + _Nx_img);

	if ( img_x > _num_image_X - 1 )
		img_x = _num_image_X - 1;

	grid_x = x - img_x*(2*bx + _Nx_img) - bx;
	
	if ( grid_x < 0 )
		grid_x = 0;
	else if ( grid_x > _Nx_img - 1)
		grid_x = _Nx_img - 1;

	y -= bt;									// Remove top border
	y = (_Ny_img + 2*by)*_num_image_Y - y;		// Flip coordinate system in y

	img_y = y / (2*by + _Ny_img);

	if ( img_y > _num_image_Y - 1 )
		img_y = _num_image_Y - 1;

	grid_y = y - img_y*(2*by + _Ny_img) - by;
	
	if ( grid_y < 0 )
		grid_y = 0;
	else if ( grid_y > _Ny_img - 1)
		grid_y = _Ny_img - 1;
}


template <typename T>
template <typename GRID>
void Visualization<T>::plotImage( const GRID& grid, int field, Type::type representation,
	int img_x, int img_y, int normalize, T minVal, T maxVal )
{
	if ( grid.dimX() > _Nx_img || grid.dimY() > _Ny_img )
	{
		std::cout << "Visualization<T>::plotImage() ERROR: dimensions of input grid does not match allocated image.\n";
		std::cout << "--> _Nx_img = " << _Nx_img << ", gridDimX = " << grid.dimX() << ", _Ny_img = " << _Ny_img << ", gridDimY = " << grid.dimY() << "\n";
		return;
	}

	const int numFields = grid.numFields();
		
	if ( field < numFields && img_x < _num_image_X && img_y < _num_image_Y )
	{
		if ( representation == Type::real || representation == Type::imag )
		{
			drawField( &grid, field, img_x, img_y, representation, normalize, minVal, maxVal );
			eraseValuesOutsideDomain( img_x, img_y );
		}
		else if ( representation == Type::complex  &&  img_x+1 < _num_image_X )
		{
			drawField( &grid, field, img_x, img_y, Type::real, normalize, minVal, maxVal );
			drawField( &grid, field, img_x+1, img_y, Type::imag, normalize, minVal, maxVal );

			eraseValuesOutsideDomain( img_x, img_y );
			eraseValuesOutsideDomain( img_x+1, img_y );
		}
	}
	else
	{
		std::cout << "Visualization<T>::plotImage() ERROR: field index or subimage index out of range.\n";
	}
}


template <typename T>
void Visualization<T>::label( std::string info, int img_x, int img_y )
{
	int imgIdx = _num_image_X*img_y + img_x;
	_textImg[imgIdx] = info;
}


template <typename T>
template <typename GRID>
void Visualization<T>::drawField(
	const GRID* grid, const int field, const int image_idx_x, const int image_idx_y,
	Type::type representation, int normalize, T minVal, T maxVal )
{
	const int Nx = grid->dimX();
	const int Ny = grid->dimY();

	GridCPU grid_cpu = (*grid)[field];

	T* data = grid_cpu.getDataPointer( 0, representation );

	// Get normalization parameters
	int img_idx = image_idx_y*_num_image_X + image_idx_x;

	T minVal_img, maxVal_img;
	GridCPU::minmax( grid_cpu, minVal_img, maxVal_img, 0, representation );

	_minValImg[img_idx] = minVal_img;
	_maxValImg[img_idx] = maxVal_img;
	_valImgUsed[img_idx] = 1;
	
	if ( normalize )
	{
		minVal = minVal_img;
		maxVal = maxVal_img;
	}
	
	float pixelValue, *pixelPtr, diff, norm;

	diff = maxVal - minVal;
	
	if ( abs(diff) < 0.0000001f )
		diff = 1.0f;
		
	norm = _displayMult/diff;
	
	// Image pointer and indexing
	int imgIndexOffset = getImageIndexOffset( image_idx_x, image_idx_y );
	int rowIndexOffset = getRowIndexOffset();
	int i, j, ii, jj, idx, row, col;
		
	// Render image into buffer
	for ( j=0; j<Ny; j++ )
	{
		for ( i=0; i<Nx; i++ )
		{
			idx = j*Nx + i;
			
			pixelValue = ((float)data[idx] - minVal) * norm;
						
			for ( jj=0; jj<_displayZoom; ++jj )
			{
				for ( ii=0; ii<_displayZoom; ++ii )
				{
					row = _displayZoom*j + jj;
					col = _displayZoom*i + ii;
					
					// Find pointer to current pixel (3 values per pixel)
					pixelPtr = _image + 3*(imgIndexOffset + row*rowIndexOffset + col);
										
					_colorScheme.getColor( pixelValue, pixelPtr[0], pixelPtr[1], pixelPtr[2] );
				}	
			}			
		}
	}	
}



template <typename T>
void Visualization<T>::plotGraph( GridCPU* xdata, GridCPU* ydata, float r, float g, float b, int image_idx_x, int image_idx_y, float y_mult )
{
	int N = xdata->dimX();
	
	// Draw plot into buffer
	int imgIndexOffset = getImageIndexOffset( image_idx_x, image_idx_y );
	int rowIndexOffset = getRowIndexOffset();
	int i, xi, yi;
	
	float *pixelPtr, pixelVal;
	
	T x, y, xf, yf, xd, yd, *xp,*yp;

	xp = xdata->getDataPointer(0);
	yp = ydata->getDataPointer(0);

	const T x_min = xp[0], x_max = xp[N-1];
	const T y_min = -1.0 * y_mult, y_max = 1.0 * y_mult;

	int img_idx = image_idx_y*_num_image_X + image_idx_x;
	_minValImg[img_idx] = x_min;
	_maxValImg[img_idx] = x_max;
	_valImgUsed[img_idx] = 1;

	float cropMult;

	// Render plot into buffer
	for ( i=0; i<N; ++i )
	{
		x = xp[i];
		y = yp[i];

		xf = (x - x_min)/(x_max - x_min) * T(_Nx_img-1);
		yf = (y - y_min)/(y_max - y_min) * T(_Ny_img-1);
		
		xi = xf;
		yi = yf;
		
		xd = xf - int(xf);
		yd = yf - int(yf);
		
		int dir = 1;
		if ( yf > _Ny_img/2 )
			dir = -1;
		
		cropMult = 1.0f; 

		if ( yf > _Ny_img-2 )
		{
			yi = _Ny_img-2;
			cropMult = 0.5f;
		}
		else if ( yf < 0)
		{
			yi = 0;
			cropMult = 0.5f;
		}

		if ( xi > -1 && xi < _Nx_img-1 && yi > -1 && yi < _Ny_img-1 )
		{
			while ( yi != _Ny_img/2 )
			{
				// Find pointer to current pixel (3 values per pixel)
				for ( int sy=0; sy<2; ++sy )
				{
					for ( int sx=0; sx<2; ++sx )
					{
						pixelPtr = _image + 3*(imgIndexOffset + (yi+sy)*rowIndexOffset + (xi+sx));
						pixelVal = (1.0f-xd + (sx*(2.0f*xd-1.0f))) * (1.0f-yd + (sy*(2.0f*yd-1.0f)));
					
						pixelPtr[0] += r * pixelVal;
						pixelPtr[1] += g * pixelVal * cropMult;
						pixelPtr[2] += b * pixelVal * cropMult;
					}
				}
				
				yi += dir;
			}
		}
	}
}


template <typename T>
void Visualization<T>::plotVerticalMarker( GridCPU* xdata, T x_coord, float r, float g, float b, int image_idx_x, int image_idx_y )
{
	int N = xdata->dimX();
	
	// Draw plot into buffer
	int imgIndexOffset = getImageIndexOffset( image_idx_x, image_idx_y );
	int rowIndexOffset = getRowIndexOffset();
	int i, xi;
	
	float *pixelPtr;
	
	T x, xf, *xp;//, xd;

	xp = xdata->getDataPointer(0);

	const T x_min = xp[0], x_max = xp[N-1], x_delta = (x_max-x_min)/(N-1);

	// Render plot into buffer
	for ( i=0; i<N; ++i )
	{
		x = xp[i];

		xf = (x - x_min)/(x_max - x_min) * T(_Nx_img-1);
		xi = xf;
				
		if ( xi > -1 && xi < _Nx_img-1 )
		{		
			// Zero energy
			if ( x < x_coord+x_delta*0.5f && x > x_coord-x_delta*0.5f  )
			{
				for ( int yi=0; yi<_Ny_img; ++yi )
				{
					pixelPtr = _image + 3*(imgIndexOffset + yi*rowIndexOffset + xi);
					
					pixelPtr[0] += r;
					pixelPtr[1] += g;
					pixelPtr[2] += b;
				}
			}
		}
	}
}


template <typename T>
void Visualization<T>::eraseValuesOutsideDomain( const int image_idx_x, const int image_idx_y )
{
#ifdef __CUDACC__
	if ( _geom_mask )
	{
		char* data_dom = _geom_mask->getDataPointer( 0, Type::real );
		
		const int Nx_dom = _geom_mask->dimX();
		const int Ny_dom = _geom_mask->dimY();
		
		const int Nx_img = _Nx_grid;
		const int Ny_img = _Ny_grid;
	
		const int Nx_diff = (Nx_dom - Nx_img)/2;
		const int Ny_diff = (Ny_dom - Ny_img)/2;
		
		int i, j, ii, jj, row, col, idx_dom;
		
		float *pixelPtr;
		
		int imgIndexOffset = getImageIndexOffset( image_idx_x, image_idx_y );
		int rowIndexOffset = getRowIndexOffset();

		// Render image into buffer
		for ( j=0; j<Ny_img; ++j )
		{
			for ( i=0; i<Nx_img; ++i )
			{
				idx_dom = (j+Ny_diff)*Nx_dom + (i+Nx_diff);
							
				if ( data_dom[idx_dom] == 0 )
				{
					for ( jj=0; jj<_displayZoom; ++jj )
					{
						for ( ii=0; ii<_displayZoom; ++ii )
						{
							row = _displayZoom*j + jj;
							col = _displayZoom*i + ii;
							
							pixelPtr = _image + 3*(imgIndexOffset + row*rowIndexOffset + col);
							pixelPtr[0] = 0.2f;
							pixelPtr[1] = 0.2f;
							pixelPtr[2] = 0.2f;							
						}	
					}
				}
			}		
		}
	}
#endif
}



template <typename T>
int Visualization<T>::getImageIndexOffset( int image_idx_x, int image_idx_y ) const
{
	if ( image_idx_x >= 0 && image_idx_x < _num_image_X && image_idx_y >=0 && image_idx_y < _num_image_Y )
	{
		int imgWidth = _Nx_img*_displayZoom + 2*_border_x;
		int imgHeight = _Ny_img*_displayZoom + 2*_border_y;
	
		int topBorderOffset = imgWidth * _num_image_X * _border_y;
		int imgRowOffset = (imgWidth*imgHeight*_num_image_X) * image_idx_y;
	
		return topBorderOffset + imgRowOffset + image_idx_x * imgWidth + _border_x;
	}
	else
	{
		std::cout << "Visualization<T>::getImageIndexOffset() ERROR: image indices out of bounds\n";
		return 0;
	}
}


template <typename T>
int Visualization<T>::getRowIndexOffset() const
{
	return (_Nx_img*_displayZoom+2*_border_x) * _num_image_X;
}


template <typename T>
int Visualization<T>::getIndexToRow( const int row, const int image ) const
{
	const int pixelsPerRow = (_Nx_img*_displayZoom + 2*_border_x) * _num_image_X;
	return (_border_y + row) * pixelsPerRow + (_Nx_img*_displayZoom + 2*_border_x)*image + _border_x;
}


template <typename T>
void Visualization<T>::setZoom( const int zoom )
{
	if ( _displayZoom != zoom )
	{
		_displayZoom = zoom;
		allocMemImageGrid( _num_image_X, _num_image_Y );
	}
}


#ifdef __CUDACC__
template <typename T>
void Visualization<T>::setGeometry( GeometryGroup<T>* geom )
{
	_geom = geom;
}
#endif


template <typename T>
void Visualization<T>::setColorScheme( Color::Base* colorScheme )
{
	_colorScheme = *colorScheme;
}


template <typename T>
void Visualization<T>::setColorScheme( const Color::Base& colorScheme )
{
	_colorScheme = colorScheme;
}


template <typename T>
void Visualization<T>::clearImage( int img_x, int img_y )
{
	if ( img_x == -1 && img_y == -1 )
	{
		for ( int i=0; i<_numPixelsAllocated*3; ++i )
			_image[i] = 0.0f;
	}
	else if ( img_x > -1 && img_y > -1 )
	{
		int imgIndexOffset = getImageIndexOffset( img_x, img_y );
		int rowIndexOffset = getRowIndexOffset();
	
		float *pixelPtr;

		for ( int j=0; j<_Ny_img; ++j )
		{
			for ( int i=0; i<_Nx_img; ++i )
			{
				pixelPtr = _image + 3*(imgIndexOffset + j*rowIndexOffset + i);
				pixelPtr[0] = 0.0f; pixelPtr[1] = 0.0f; pixelPtr[2] = 0.0f;
			}		
		}
	}

#ifdef __CUDACC__
	if ( _geom_mask == 0 && _geom )
	{
		_geom->create(0.0);
		_geom_mask = new Grid<char>::CPU( *_geom->getDomainLabelsGrid() );
	}
#endif
}


template <typename T>
void Visualization<T>::printRange( int img_row, int img_col, T minVal, T maxVal )
{
	std::cout << "Img(row,col) = (" << img_row << "," << img_col <<
		"):\tValue range = (\t" << minVal << ",\t" << maxVal << "\t)\n";
}


template <typename T>
void Visualization<T>::printRangeToImage( int img_row, int img_col, T minVal, T maxVal )
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
 
 	float posX = (float)img_col/(float)_num_image_X * 2.0f - 1.0f;
	float posY = (float)img_row/(float)_num_image_Y * 2.0f - 1.0f;
 
 	posX += (float)_border_x/(float)getNumPixelsImagesX() * 2.0f;
 	posY += (float)_border_y/(float)getNumPixelsImagesY() * 0.5f;
 
 	posX = (posX+1.0f)*0.5f * (float)getNumPixelsImagesX() / (float)glutGet(GLUT_WINDOW_WIDTH);
 	posX = posX*2.0f - 1.0f;
 
 	// Shift coordinates according to top border and window size
	float b = (float)getNumPixelsImagesY() / (float)getNumPixelsWindowY() * 2.0f - 1.0f;

	posY = (posY+1.0f)*0.5f * (b+1.0f) - 1.0f;
 
	glColor3f( 1.0f, 1.0f, 1.0f );
	glRasterPos2f( posX, posY );
	int i;

	char txt[256];
	int imgIdx = _num_image_X*img_row + img_col;

	sprintf( txt, "%s (%f, %f)", _textImg[imgIdx].c_str(), minVal, maxVal );

	for ( i=0; txt[i]; i++ )
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, txt[i]);
 
	glRasterPos2f(-1.0f,-1.0f);	
}


template <typename T>
void Visualization<T>::printParametersToImage( const Parameters<T> * param )
{
	if ( param !=0 && param->isValid() )
	{
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
	 
	 	float posX = (float)_border_x/(float)getNumPixelsImagesX() * 2.0f - 1.0f;
	 	float posY = 1.0f - (float)_border_top_y / (float)getNumPixelsWindowY();
	 
		glColor3f( 1.0f, 1.0f, 1.0f );
		glRasterPos2f( posX, posY );
		int i;

		int numE = param->getNumEnergies();
		if ( numE == 0 )
			numE = param->getNumFrequencies();

		char txt[256];
		sprintf( txt, "Discrete width = %d,   Discrete angles = %d,   Discrete Energies = %d",
			param->getGridResolutionBase(), param->getAngularResolution(), numE );

		for ( i=0; txt[i]; i++ )
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, txt[i]);
	  
		glRasterPos2f( posX, posY - 36.0f/getNumPixelsWindowY() );
		sprintf( txt, "Physical width = %f c.l.,  \nB = %f,   T/T_c = %f",
			(param->getGridResolutionBase()-1)*param->getGridElementSize(),	param->getMagneticFieldStrength(), param->getTemperature() );

	 	for ( i=0; txt[i]; i++ )
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, txt[i]);
	 
		glRasterPos2f(-1.0f,-1.0f);
	}
}


template <typename T>
void Visualization<T>::printParametersToImage()
{
	printParametersToImage(_param);
}


template <typename T>
void Visualization<T>::drawPixels()
{
	// GL calls to draw image
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
	glDrawPixels( getNumPixelsWindowX(), getNumPixelsWindowY(), GL_RGB, GL_FLOAT, _image );
}


template <typename T>
void Visualization<T>::drawInfo()
{
	for ( int j=0; j<_num_image_Y; ++j )
	{
		for ( int i=0; i<_num_image_X; ++i )
		{
			int img_idx = j*_num_image_X + i;
			if ( _valImgUsed[img_idx] )
			{
				printRangeToImage(j,i,_minValImg[img_idx],_maxValImg[img_idx]);
			}
		}
	}

	if ( _param )
		printParametersToImage();	
}


template <typename T>
void Visualization<T>::draw()
{
	drawPixels();
	drawInfo();
	glutSwapBuffers();
}

