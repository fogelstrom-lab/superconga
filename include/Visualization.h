//===------------- Visualization.h - OpenGL live visualization. -----------===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of OpenGL live visualization.
///
//===----------------------------------------------------------------------===//
#ifndef VISUALIZATION_H_
#define VISUALIZATION_H_

#ifdef __linux__
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

#include "Colormap.h"
#include "Grid.h"
#include "IncludeGL.h"
#include "Parameters.h"
#include "defines.h"
#include "utils.h"
#include <cmath>

#ifdef __CUDACC__
#include "GeometryGroup.h"
#endif

namespace scg {
template <typename T> class Visualization {
public:
  Visualization();
  ~Visualization();

  template <typename GRID>
  Visualization(const GRID *grid, const int numImageX = 1,
                const int numImageY = 1);
  Visualization(const int Nx, const int Ny, const int numImageX = 1,
                const int numImageY = 1);

  void setZoom(int zoom);
  void setDisplayMult(const float displayMult) { _displayMult = displayMult; }
  void setPrintParameters(Parameters<T> *param) { _param = param; }

  // Initialize pixel buffer. num_img = 3 means allocate memory for 3 images -
  // i.e. one each for real, imag, and abs parts.
  void allocMemImageGrid(const int num_image_X, const int num_image_Y);

  template <typename GRID>
  void drawField(const GRID &grid, int field, int image_idx_x, int image_idx_y,
                 Type::type representation, const bool inIsPositive);

  template <typename GRID>
  void plotImage(const GRID &grid, const int inColumnIdx,
                 const bool erase = true);

  void label(std::string info, int img_x, int img_y);

  void plotGraph(GridCPU<T> *xdata, GridCPU<T> *ydata, float r, float g,
                 float b, int image_idx_x, int image_idx_y, float y_mult = 1.0);
  void plotVerticalMarker(GridCPU<T> *xdata, T x_coord, float r, float g,
                          float b, int image_idx_x, int image_idx_y);

  void eraseValuesOutsideDomain(const int image_idx_x,
                                const int image_idx_y = 0);

  // Drawing calls
  void drawPixels();
  void drawInfo();
  void draw();
  void clearImage(int img_x = -1, int img_y = -1);

  // Getters
  int getNumImagesX() const { return _num_image_X; }
  int getNumImagesY() const { return _num_image_Y; }

  int getBorderX() const { return _border_x; }
  int getBorderY() const { return _border_y; }
  int getBorderTop() const { return _border_top_y; }

  // Resolution
  int getNumPixelsImageX() const { return _Nx_img; }
  int getNumPixelsImageY() const { return _Ny_img; }

  int getNumPixelsImagesX() const {
    return (_Nx_img * _displayZoom + 2 * _border_x) * _num_image_X;
  }
  int getNumPixelsImagesY() const {
    return (_Ny_img * _displayZoom + 2 * _border_y) * _num_image_Y;
  }

  int getNumPixelsWindowX() const {
    return (_Nx_img * _displayZoom + 2 * _border_x) * _num_image_X;
  }
  int getNumPixelsWindowY() const {
    return (_Ny_img * _displayZoom + 2 * _border_y) * _num_image_Y +
           _border_top_y;
  }

  // Image interaction
  void getImageCoordinatesFromWindow(int x, int y, int &img_x, int &img_y,
                                     int &grid_x, int &grid_y);

  // Helpers
  int getIndexToRow(const int row, const int image = 0) const;
  int getImageIndexOffset(int image_idx_x, int image_idx_y) const;
  int getRowIndexOffset() const;

  void printRange(int img_row, int img_col, T minVal, T maxVal);
  void printRangeToImage(int img_row, int img_col, T minVal, T maxVal);
  void printParametersToImage(const Parameters<T> *param);
  void printParametersToImage();

#ifdef __CUDACC__
  void setGeometry(GeometryGroup<T> *geom);
#endif

private:
  void constructor();

  // Size of inpute grid
  int _Nx_grid;
  int _Ny_grid;

  // Pixels per subimage
  int _Nx_img;
  int _Ny_img;

  // Entire displayed image
  int _Nx_window;
  int _Ny_window;

  // Border pixel size
  int _border_top_y;
  int _border_x;
  int _border_y;

  // Subimages that make up whole image
  int _num_image;
  int _num_image_X;
  int _num_image_Y;

  int _numPixelsAllocated;
  float *_image;

  int _currentDrawnField;

  int _displayZoom;
  float _displayMult;

  // Print parameters and range
  Parameters<T> *_param;
  int _valImgUsed[64];
  float _minValImg[64];
  float _maxValImg[64];

  std::string _textImg[64];

  // Other
  Colormap m_colormap;

#ifdef __CUDACC__
  GeometryGroup<T> *_geom;
  GridCPU<char> *_geom_mask;
#endif
};

template <typename T>
Visualization<T>::Visualization()
    : _image(0), _displayZoom(1), _num_image(0), _border_x(0), _border_y(0),
      _border_top_y(0), _currentDrawnField(-1), _displayMult(1.0) {
  for (int i = 0; i < 64; ++i) {
    _valImgUsed[i] = 0;
    _textImg[i] = "";
  }

#ifdef __CUDACC__
  _geom = 0;
  _geom_mask = 0;
#endif
}

template <typename T> Visualization<T>::~Visualization() {
  if (_image)
    delete[] _image;
#ifdef __CUDACC__
  if (_geom_mask)
    delete _geom_mask;
#endif
}

template <typename T>
template <typename GRID>
Visualization<T>::Visualization(const GRID *grid, const int numImageX,
                                const int numImageY)
    : _image(0), _numPixelsAllocated(0), _param(0), _Nx_grid(grid->dimX()),
      _Ny_grid(grid->dimY()), _num_image_X(numImageX), _num_image_Y(numImageY),
      _num_image(numImageX * numImageY), _border_top_y(32), _border_x(16),
      _border_y(16), _displayZoom(1), _displayMult(1.0),
      _currentDrawnField(-1) {
  constructor();
}

template <typename T>
Visualization<T>::Visualization(const int Nx, const int Ny, const int numImageX,
                                const int numImageY)
    : _image(0), _numPixelsAllocated(0), _param(0), _Nx_grid(Nx), _Ny_grid(Ny),
      _num_image_X(numImageX), _num_image_Y(numImageY),
      _num_image(numImageX * numImageY), _border_top_y(32), _border_x(16),
      _border_y(16), _displayZoom(1), _displayMult(1.0),
      _currentDrawnField(-1) {
  constructor();
}

template <typename T> void Visualization<T>::constructor() {
  allocMemImageGrid(_num_image_X, _num_image_Y);

  for (int i = 0; i < 64; ++i) {
    _valImgUsed[i] = 0;
    _textImg[i] = "";
  }

#ifdef __CUDACC__
  _geom = 0;
  _geom_mask = 0;
#endif
}

template <typename T>
void Visualization<T>::allocMemImageGrid(const int num_image_X,
                                         const int num_image_Y) {
  int Nx = _Nx_grid;
  int Ny = _Ny_grid;

  if ((Nx % 4) != 0)
    Nx += 4 - (Nx % 4); // glDrawPixels want sizes to be evenly divisible by 4

  if ((Ny % 4) != 0)
    Ny += 4 - (Ny % 4);

  _Nx_img = Nx;
  _Ny_img = Ny;

  _num_image_X = num_image_X;
  _num_image_Y = num_image_Y;
  _num_image = num_image_X * num_image_Y;

  _Nx_window = getNumPixelsWindowX();
  _Ny_window = getNumPixelsWindowY();

  int numPixels = _Nx_window * _Ny_window;

  if (numPixels != _numPixelsAllocated) {
    if (_image)
      delete[] _image;

    _image = new float[numPixels * 3];
    _numPixelsAllocated = numPixels;
  }
}

template <typename T>
void Visualization<T>::getImageCoordinatesFromWindow(int x, int y, int &img_x,
                                                     int &img_y, int &grid_x,
                                                     int &grid_y) {
  // ADD ZOOM!

  int bx = getBorderX();
  int by = getBorderY();
  int bt = getBorderTop();

  img_x = x / (2 * bx + _Nx_img);

  if (img_x > _num_image_X - 1)
    img_x = _num_image_X - 1;

  grid_x = x - img_x * (2 * bx + _Nx_img) - bx;

  if (grid_x < 0)
    grid_x = 0;
  else if (grid_x > _Nx_img - 1)
    grid_x = _Nx_img - 1;

  y -= bt;                                   // Remove top border
  y = (_Ny_img + 2 * by) * _num_image_Y - y; // Flip coordinate system in y

  img_y = y / (2 * by + _Ny_img);

  if (img_y > _num_image_Y - 1)
    img_y = _num_image_Y - 1;

  grid_y = y - img_y * (2 * by + _Ny_img) - by;

  if (grid_y < 0)
    grid_y = 0;
  else if (grid_y > _Ny_img - 1)
    grid_y = _Ny_img - 1;
}

template <typename T>
template <typename GRID>
void Visualization<T>::plotImage(const GRID &grid, const int inColumnIdx,
                                 const bool erase) {
  if (grid.dimX() > _Nx_img || grid.dimY() > _Ny_img) {
#ifndef NDEBUG
    std::cout << "-- ERROR! Visualization<T>::plotImage(): dimensions of input "
                 "grid does not match allocated image.\n";
    std::cout << "--> _Nx_img = " << _Nx_img << ", gridDimX = " << grid.dimX()
              << ", _Ny_img = " << _Ny_img << ", gridDimY = " << grid.dimY()
              << "\n";
#endif
    return;
  }

  // Get the number of fields.
  const int numFields = grid.numFields();

  // Early exit if there is nothing to do.
  if (numFields == 0) {
    return;
  }

  // Check if this is a complex field.
  const bool isComplex = grid.isComplex();

  // Check if valid index.
  if (inColumnIdx < _num_image_X) {
    if (isComplex) {
      // Only one field for complex grids are supported.
      const int field = 0;

      // Set the row index.
      int rowIdx = _num_image_Y - 1;

      // Check if valid index.
      if ((0 <= rowIdx) && (rowIdx < _num_image_Y)) {
        // Plot the magnitude of the grid.
        const bool isPositive = true;
        drawField(grid.abs(), field, inColumnIdx, rowIdx, Type::real,
                  isPositive);

        // Erase values outside of the grain.
        if (erase) {
          eraseValuesOutsideDomain(inColumnIdx, rowIdx);
        }
      }

      // Decrease the row index, i.e. jump down one row.
      rowIdx--;

      // Check if valid index.
      if ((0 <= rowIdx) && (rowIdx < _num_image_Y)) {
        // Plot the phase of the grid.
        const bool isPositive = false;
        drawField(grid.complexPhase(), field, inColumnIdx, rowIdx, Type::real,
                  isPositive);

        // Erase values outside of the grain.
        if (erase) {
          eraseValuesOutsideDomain(inColumnIdx, rowIdx);
        }
      }
    } else {
      // Set the row index.
      int rowIdx = _num_image_Y - 1;

      if (numFields == 1) {
        // Check if valid index.
        if ((0 <= rowIdx) && (rowIdx < _num_image_Y)) {
          // If there is only one (real valued) field, just plot it.
          const int field = 0;
          const bool isPositive = false;
          drawField(grid, field, inColumnIdx, rowIdx, Type::real, isPositive);

          // Erase values outside of the grain.
          if (erase) {
            eraseValuesOutsideDomain(inColumnIdx, rowIdx);
          }
        }
      } else if (numFields > 1) {
        // Check if valid index.
        if ((0 <= rowIdx) && (rowIdx < _num_image_Y)) {
          // Plot the magnitude of the grid.
          const int field = 0;
          const bool isPositive = true;
          drawField(grid.fieldMagnitude(), field, inColumnIdx, rowIdx,
                    Type::real, isPositive);

          // Erase values outside of the grain.
          if (erase) {
            eraseValuesOutsideDomain(inColumnIdx, rowIdx);
          }
        }

        // Plot all fields, untouched.
        const bool isPositive = false;
        for (int field = 0; field < 2; ++field) {
          // Decrease the row index, i.e. jump down one row.
          rowIdx--;

          // Check if valid index.
          if ((0 <= rowIdx) && (rowIdx < _num_image_Y)) {
            // Plot this field.
            drawField(grid, field, inColumnIdx, rowIdx, Type::real, isPositive);

            // Erase values outside of the grain.
            if (erase) {
              eraseValuesOutsideDomain(inColumnIdx, rowIdx);
            }
          }
        }
      }
    }
  } else {
#ifndef NDEBUG
    std::cout
        << "-- ERROR! Visualization<T>::plotImage(): field index or subimage "
           "index out of range.\n";
#endif
  }
}

template <typename T>
void Visualization<T>::label(std::string info, int img_x, int img_y) {
  int imgIdx = _num_image_X * img_y + img_x;
  _textImg[imgIdx] = info;
}

template <typename T>
template <typename GRID>
void Visualization<T>::drawField(const GRID &grid, const int field,
                                 const int image_idx_x, const int image_idx_y,
                                 const Type::type representation,
                                 const bool inIsPositive) {
  const int Nx = grid.dimX();
  const int Ny = grid.dimY();

  GridCPU<T> grid_cpu = grid[field];

  T *data = grid_cpu.getDataPointer(0, representation);

  // Get normalization parameters
  int img_idx = image_idx_y * _num_image_X + image_idx_x;

  T minVal, maxVal;
  grid_cpu.minmax(minVal, maxVal, 0, representation);

  // Save the min/max values so we can print them in the window.
  _minValImg[img_idx] = static_cast<float>(minVal);
  _maxValImg[img_idx] = static_cast<float>(maxVal);
  _valImgUsed[img_idx] = 1;

  // Set the limit, for normalizaing the pixel values. This is done so that the
  // full range of colors is not used when the grid is essentially just zeros.
  // Epsilon is the value where we stop using the full color range.
  const T maxAbsVal = std::max(std::abs(minVal), std::abs(maxVal));
  const T epsilon = static_cast<T>(1e-6);
  const T limit = std::max(maxAbsVal, epsilon);

  if (inIsPositive) {
    minVal = static_cast<T>(0);
    maxVal = limit;
  } else {
    minVal = -limit;
    maxVal = limit;
  }

  // Set the colormap type.
  // TODO(niclas): Use a cyclic colormap for the phase.
  const Colormap::Type colormapType =
      inIsPositive ? Colormap::Type::HOT : Colormap::Type::RDBU;

  // The normalization factor. Note that the denominator cannot be zero due to
  // 'epsilon' above.
  const T normalization = _displayMult / (maxVal - minVal);

  // Image pointer and indexing
  const int imgIndexOffset = getImageIndexOffset(image_idx_x, image_idx_y);
  const int rowIndexOffset = getRowIndexOffset();

  float *pixelPtr;

  // Render image into buffer
  for (int j = 0; j < Ny; j++) {
    for (int i = 0; i < Nx; i++) {
      const int idx = j * Nx + i;

      // Normalized pixel value.
      const float pixelValue =
          static_cast<float>((data[idx] - minVal) * normalization);

      for (int jj = 0; jj < _displayZoom; ++jj) {
        for (int ii = 0; ii < _displayZoom; ++ii) {
          const int row = _displayZoom * j + jj;
          const int col = _displayZoom * i + ii;

          // Find pointer to current pixel (3 values per pixel)
          pixelPtr = _image + 3 * (imgIndexOffset + row * rowIndexOffset + col);

          // Set the pixel.
          m_colormap.getColor(pixelValue, colormapType, pixelPtr[0],
                              pixelPtr[1], pixelPtr[2]);
        }
      }
    }
  }
}

template <typename T>
void Visualization<T>::plotGraph(GridCPU<T> *xdata, GridCPU<T> *ydata, float r,
                                 float g, float b, int image_idx_x,
                                 int image_idx_y, float y_mult) {
  int N = xdata->dimX();

  // Draw plot into buffer
  int imgIndexOffset = getImageIndexOffset(image_idx_x, image_idx_y);
  int rowIndexOffset = getRowIndexOffset();
  int i, xi, yi;

  float *pixelPtr, pixelVal;

  T x, y, xf, yf, xd, yd, *xp, *yp;

  xp = xdata->getDataPointer(0);
  yp = ydata->getDataPointer(0);

  const T x_min = xp[0], x_max = xp[N - 1];
  const T y_min = -1.0 * y_mult, y_max = 1.0 * y_mult;

  int img_idx = image_idx_y * _num_image_X + image_idx_x;
  _minValImg[img_idx] = x_min;
  _maxValImg[img_idx] = x_max;
  _valImgUsed[img_idx] = 1;

  float cropMult;

  // Render plot into buffer
  for (i = 0; i < N; ++i) {
    x = xp[i];
    y = yp[i];

    xf = (x - x_min) / (x_max - x_min) * T(_Nx_img - 1);
    yf = (y - y_min) / (y_max - y_min) * T(_Ny_img - 1);

    xi = xf;
    yi = yf;

    xd = xf - int(xf);
    yd = yf - int(yf);

    int dir = 1;
    if (yf > _Ny_img / 2)
      dir = -1;

    cropMult = 1.0f;

    if (yf > _Ny_img - 2) {
      yi = _Ny_img - 2;
      cropMult = 0.5f;
    } else if (yf < 0) {
      yi = 0;
      cropMult = 0.5f;
    }

    if (xi > -1 && xi < _Nx_img - 1 && yi > -1 && yi < _Ny_img - 1) {
      while (yi != _Ny_img / 2) {
        // Find pointer to current pixel (3 values per pixel)
        for (int sy = 0; sy < 2; ++sy) {
          for (int sx = 0; sx < 2; ++sx) {
            pixelPtr = _image + 3 * (imgIndexOffset +
                                     (yi + sy) * rowIndexOffset + (xi + sx));
            pixelVal = (1.0f - xd + (sx * (2.0f * xd - 1.0f))) *
                       (1.0f - yd + (sy * (2.0f * yd - 1.0f)));

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
void Visualization<T>::plotVerticalMarker(GridCPU<T> *xdata, T x_coord, float r,
                                          float g, float b, int image_idx_x,
                                          int image_idx_y) {
  int N = xdata->dimX();

  // Draw plot into buffer
  int imgIndexOffset = getImageIndexOffset(image_idx_x, image_idx_y);
  int rowIndexOffset = getRowIndexOffset();
  int i, xi;

  float *pixelPtr;

  T x, xf, *xp; //, xd;

  xp = xdata->getDataPointer(0);

  const T x_min = xp[0], x_max = xp[N - 1], x_delta = (x_max - x_min) / (N - 1);

  // Render plot into buffer
  for (i = 0; i < N; ++i) {
    x = xp[i];

    xf = (x - x_min) / (x_max - x_min) * T(_Nx_img - 1);
    xi = xf;

    if (xi > -1 && xi < _Nx_img - 1) {
      // Zero energy
      if (x < x_coord + x_delta * 0.5f && x > x_coord - x_delta * 0.5f) {
        for (int yi = 0; yi < _Ny_img; ++yi) {
          pixelPtr = _image + 3 * (imgIndexOffset + yi * rowIndexOffset + xi);

          pixelPtr[0] += r;
          pixelPtr[1] += g;
          pixelPtr[2] += b;
        }
      }
    }
  }
}

template <typename T>
void Visualization<T>::eraseValuesOutsideDomain(const int image_idx_x,
                                                const int image_idx_y) {
#ifdef __CUDACC__
  if (_geom_mask) {
    char *data_dom = _geom_mask->getDataPointer(0, Type::real);

    const int Nx_dom = _geom_mask->dimX();
    const int Ny_dom = _geom_mask->dimY();

    const int Nx_img = _Nx_grid;
    const int Ny_img = _Ny_grid;

    const int Nx_diff = (Nx_dom - Nx_img) / 2;
    const int Ny_diff = (Ny_dom - Ny_img) / 2;

    int i, j, ii, jj, row, col, idx_dom;

    float *pixelPtr;

    int imgIndexOffset = getImageIndexOffset(image_idx_x, image_idx_y);
    int rowIndexOffset = getRowIndexOffset();

    // Render image into buffer
    for (j = 0; j < Ny_img; ++j) {
      for (i = 0; i < Nx_img; ++i) {
        idx_dom = (j + Ny_diff) * Nx_dom + (i + Nx_diff);

        if (data_dom[idx_dom] == 0) {
          for (jj = 0; jj < _displayZoom; ++jj) {
            for (ii = 0; ii < _displayZoom; ++ii) {
              row = _displayZoom * j + jj;
              col = _displayZoom * i + ii;

              pixelPtr =
                  _image + 3 * (imgIndexOffset + row * rowIndexOffset + col);
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
int Visualization<T>::getImageIndexOffset(int image_idx_x,
                                          int image_idx_y) const {
  if (image_idx_x >= 0 && image_idx_x < _num_image_X && image_idx_y >= 0 &&
      image_idx_y < _num_image_Y) {
    int imgWidth = _Nx_img * _displayZoom + 2 * _border_x;
    int imgHeight = _Ny_img * _displayZoom + 2 * _border_y;

    int topBorderOffset = imgWidth * _num_image_X * _border_y;
    int imgRowOffset = (imgWidth * imgHeight * _num_image_X) * image_idx_y;

    return topBorderOffset + imgRowOffset + image_idx_x * imgWidth + _border_x;
  } else {
#ifndef NDEBUG
    std::cout
        << "-- ERROR! Visualization<T>::getImageIndexOffset(): image indices "
           "out of bounds\n";
#endif
    return 0;
  }
}

template <typename T> int Visualization<T>::getRowIndexOffset() const {
  return (_Nx_img * _displayZoom + 2 * _border_x) * _num_image_X;
}

template <typename T>
int Visualization<T>::getIndexToRow(const int row, const int image) const {
  const int pixelsPerRow =
      (_Nx_img * _displayZoom + 2 * _border_x) * _num_image_X;
  return (_border_y + row) * pixelsPerRow +
         (_Nx_img * _displayZoom + 2 * _border_x) * image + _border_x;
}

template <typename T> void Visualization<T>::setZoom(const int zoom) {
  if (_displayZoom != zoom) {
    _displayZoom = zoom;
    allocMemImageGrid(_num_image_X, _num_image_Y);
  }
}

#ifdef __CUDACC__
template <typename T>
void Visualization<T>::setGeometry(GeometryGroup<T> *geom) {
  _geom = geom;
}
#endif

template <typename T> void Visualization<T>::clearImage(int img_x, int img_y) {
  if (img_x == -1 && img_y == -1) {
    for (int i = 0; i < _numPixelsAllocated * 3; ++i)
      _image[i] = 0.0f;
  } else if (img_x > -1 && img_y > -1) {
    int imgIndexOffset = getImageIndexOffset(img_x, img_y);
    int rowIndexOffset = getRowIndexOffset();

    float *pixelPtr;

    for (int j = 0; j < _Ny_img; ++j) {
      for (int i = 0; i < _Nx_img; ++i) {
        pixelPtr = _image + 3 * (imgIndexOffset + j * rowIndexOffset + i);
        pixelPtr[0] = 0.0f;
        pixelPtr[1] = 0.0f;
        pixelPtr[2] = 0.0f;
      }
    }
  }

#ifdef __CUDACC__
  if (_geom_mask == 0 && _geom) {
    _geom->create(0.0);
    _geom_mask = new GridCPU<char>(*_geom->getDomainLabelsGrid());
  }
#endif
}

template <typename T>
void Visualization<T>::printRange(int img_row, int img_col, T minVal,
                                  T maxVal) {
  std::cout << "Img(row,col) = (" << img_row << "," << img_col
            << "):\tValue range = (\t" << minVal << ",\t" << maxVal << "\t)\n";
}

template <typename T>
void Visualization<T>::printRangeToImage(int img_row, int img_col, T minVal,
                                         T maxVal) {
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  float posX = (float)img_col / (float)_num_image_X * 2.0f - 1.0f;
  float posY = (float)img_row / (float)_num_image_Y * 2.0f - 1.0f;

  posX += (float)_border_x / (float)getNumPixelsImagesX() * 2.0f;
  posY += (float)_border_y / (float)getNumPixelsImagesY() * 0.5f;

  posX = (posX + 1.0f) * 0.5f * (float)getNumPixelsImagesX() /
         (float)glutGet(GLUT_WINDOW_WIDTH);
  posX = posX * 2.0f - 1.0f;

  // Shift coordinates according to top border and window size
  float b =
      (float)getNumPixelsImagesY() / (float)getNumPixelsWindowY() * 2.0f - 1.0f;

  posY = (posY + 1.0f) * 0.5f * (b + 1.0f) - 1.0f;

  glColor3f(1.0f, 1.0f, 1.0f);
  glRasterPos2f(posX, posY);
  int i;

  char txt[256];
  int imgIdx = _num_image_X * img_row + img_col;

  sprintf(txt, "%s : [%.4f, %.4f]", _textImg[imgIdx].c_str(), minVal, maxVal);

  for (i = 0; txt[i]; i++) {
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, txt[i]);
  }

  glRasterPos2f(-1.0f, -1.0f);
}

template <typename T>
void Visualization<T>::printParametersToImage(const Parameters<T> *param) {
  if (param != 0 && param->isValid()) {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();

    float posX = (float)_border_x / (float)getNumPixelsImagesX() * 2.0f - 1.0f;
    float posY = 1.0f - (float)_border_top_y / (float)getNumPixelsWindowY();

    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(posX, posY);
    int i;

    char txt[256];
    sprintf(txt, "Discrete width = %d,   #Momenta = %d,   #Ozaki Poles = %d",
            param->getGridResolutionBase() - 2 * constants::GRAIN_SIZE_MARGIN,
            param->getAngularResolution(), param->getNumMatsubaraEnergies());

    for (i = 0; txt[i]; i++) {
      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, txt[i]);
    }

    glRasterPos2f(posX, posY - 36.0f / getNumPixelsWindowY());
    sprintf(txt,
            "Physical width = %.3f c.l.,  \n#Ext. flux quanta = %.3f,   T/T_c "
            "= %.3f",
            utils::toInternalInvLength(param->getGrainWidth()),
            param->getNumFluxQuanta(), param->getTemperature());

    for (i = 0; txt[i]; i++) {
      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, txt[i]);
    }

    glRasterPos2f(-1.0f, -1.0f);
  }
}

template <typename T> void Visualization<T>::printParametersToImage() {
  printParametersToImage(_param);
}

template <typename T> void Visualization<T>::drawPixels() {
  // GL calls to draw image
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glDrawPixels(getNumPixelsWindowX(), getNumPixelsWindowY(), GL_RGB, GL_FLOAT,
               _image);
}

template <typename T> void Visualization<T>::drawInfo() {
  for (int j = 0; j < _num_image_Y; ++j) {
    for (int i = 0; i < _num_image_X; ++i) {
      int img_idx = j * _num_image_X + i;
      if (_valImgUsed[img_idx]) {
        printRangeToImage(j, i, _minValImg[img_idx], _maxValImg[img_idx]);
      }
    }
  }

  if (_param)
    printParametersToImage();
}

template <typename T> void Visualization<T>::draw() {
  drawPixels();
  drawInfo();
  glutSwapBuffers();
}
} // namespace scg

#endif // VISUALIZATION_H_
