#ifndef _POLYGON_H
#define _POLYGON_H

#include <Grid.h>
#include <iostream>

// This class provides a basic shape implementation for a polygon.
// Polygons must be defined in a counterclockwise fashion.

template <typename T> class Polygon {
  typedef typename Grid<T>::GPU GridGPU;

public:
  Polygon();
  ~Polygon();

  // Copy constructor
  Polygon(const Polygon &other);

  // Initialize the polygon to an approximation of a disc
  // numVertices = 4 will produce a square. cx, cy is the center coordinate
  Polygon(const int numVertices, T radius, T angle = 0.0, T cx = 0.0,
          T cy = 0.0);

  // Initialize the polygon to a predefined vertex set
  Polygon(const int numVertices, T *vertX, T *vertY);

  Polygon &operator=(Polygon other);

  int getNumVertices() const { return _numVertices; }
  GridGPU *getVertices() const { return _vertices; }

  // Copy vertices from arrays vertX and vertY
  void setVertices(const T *vertX, const T *vertY);

  // Approximate a disc (this gets called from corresponding constructor)
  void setVerticesRadial(const int N, T radius, T angle, T cx, T cy);

  // Polygon vertex coordinates needs to be in normalized coordinates (-0.5 to
  // 0.5). This helper function transforms coordinates in physical space to
  // normalized space, should you want to enter coordinates in physical space.
  static void transformPhysicalToNormalized(const int numVertices, T *px, T *py,
                                            const T physical_width);

private:
  friend void swap(Polygon &A, Polygon &B) {
    std::swap(A._numVertices, B._numVertices);
    std::swap(A._vertices, B._vertices);
  }

  int _numVertices;
  GridGPU *_vertices;
};

template <typename T> Polygon<T>::Polygon() : _vertices(0), _numVertices(0) {}

template <typename T>
Polygon<T>::Polygon(const Polygon<T> &other)
    : _vertices(0), _numVertices(other._numVertices) {
  if (other._vertices)
    _vertices = new GridGPU(*other._vertices);
}

template <typename T>
Polygon<T>::Polygon(const int numVertices, T radius, T angle, T cx, T cy)
    : _numVertices(numVertices) {
  int Nx = numVertices;
  int Ny = 1;
  int dim = 2;

  _vertices = new GridGPU(Nx, Ny, dim, Type::real);

  setVerticesRadial(numVertices, radius, angle, cx, cy);
}

template <typename T>
Polygon<T>::Polygon(const int numVertices, T *vertX, T *vertY)
    : _numVertices(numVertices) {
  int dim = 2;

  _vertices = new GridGPU(numVertices, 1, dim, Type::real);

  setVertices(vertX, vertY);
}

template <typename T> Polygon<T>::~Polygon() {
  if (_vertices)
    delete _vertices;
}

template <typename T> Polygon<T> &Polygon<T>::operator=(Polygon<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void Polygon<T>::setVertices(const T *px, const T *py) {
  typename Vector<T>::GPU &vert = *(_vertices->getVector());
  const int offset = _vertices->getFieldComponentOffset(1);

  for (int i = 0; i < _numVertices; ++i) {
    vert[i] = px[i];
    vert[i + offset] = py[i];
  }
}

template <typename T>
void Polygon<T>::setVerticesRadial(const int numVertices, T radius, T angle,
                                   T cx, T cy) {
  if (numVertices == _numVertices) {
    T px[numVertices], py[numVertices];

    for (int p = 0; p < numVertices; ++p) {
      T phi = M_PI / (T)numVertices * (1.0 + 2.0 * p) + angle;

      px[p] = radius * cos(phi) + cx;
      py[p] = radius * sin(phi) + cy;
    }

    setVertices(px, py);
  }
}

template <typename T>
void Polygon<T>::transformPhysicalToNormalized(const int numVertices, T *px,
                                               T *py, const T w) {
  const T wn = 0.35;

  for (int i = 0; i < numVertices; ++i) {
    T x = px[i];
    T y = py[i];

    x = (x / w - 0.5) * 2.0 * wn;
    y = (y / w - 0.5) * 2.0 * wn;

    px[i] = x;
    py[i] = y;
  }
}
#endif
