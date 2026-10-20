//===--------------------- Polygon.h - Polygon class. ---------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Polygon class, providing a basic shape implementation for a polygon.
/// Polygons must be defined in a counterclockwise fashion.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_POLYGON_H_
#define CONGA_POLYGON_H_

#include "../grid.h"

#include <cmath>

namespace conga {
template <typename T> class Polygon {
public:
  Polygon();
  ~Polygon();

  // Copy constructor
  Polygon(const Polygon &inOther);

  // Initialize the polygon to an approximation of a disc
  // numVertices = 4 will produce a square. inCenterX, inCenterY is the center
  // coordinate.
  Polygon(const int inNumVertices, const T inRadius,
          const T inAngle = static_cast<T>(0.0),
          const T inCenterX = static_cast<T>(0.0),
          const T inCenterY = static_cast<T>(0.0));

  // Initialize the polygon to a predefined vertex set.
  Polygon(const int inNumVertices, const T *const inVerticesX,
          const T *const inVerticesY);

  Polygon &operator=(Polygon inOther);

  int getNumVertices() const { return m_numVertices; }
  GridGPU<T> *getVertices() const { return m_vertices; }

  // Copy vertices from arrays inVerticesX and inVerticesY.
  void setVertices(const T *const inVerticesX, const T *const inVerticesY);

  // Approximate a disc (this gets called from corresponding constructor).
  void setVerticesRadial(const int inNumVertices, const T inRadius,
                         const T inAngle, const T inCenterX, const T inCenterY);

  // Polygon vertex coordinates needs to be in normalized coordinates (-0.5 to
  // 0.5). This helper function transforms coordinates in physical space to
  // normalized space, should you want to enter coordinates in physical space.
  static void transformPhysicalToNormalized(const int inNumVertices,
                                            const T inPhysicalWidth,
                                            T *const inOutVerticesX,
                                            T *const inOutVerticesY);

private:
  friend void swap(Polygon &A, Polygon &B) {
    std::swap(A.m_numVertices, B.m_numVertices);
    std::swap(A.m_vertices, B.m_vertices);
  }

  // We are dealing with 2D polygons.
  const int m_dimension = 2;

  int m_numVertices;
  GridGPU<T> *m_vertices;
};

template <typename T> Polygon<T>::Polygon() : m_vertices(0), m_numVertices(0) {}

template <typename T>
Polygon<T>::Polygon(const Polygon<T> &inOther)
    : m_vertices(0), m_numVertices(inOther.m_numVertices) {
  if (inOther.m_vertices)
    m_vertices = new GridGPU<T>(*inOther.m_vertices);
}

template <typename T>
Polygon<T>::Polygon(const int inNumVertices, const T inRadius, const T inAngle,
                    const T inCenterX, const T inCenterY)
    : m_numVertices(inNumVertices) {
  // Essentially, for every dimension create an array of length inNumVertices.
  const int numX = inNumVertices;
  const int numY = 1;

  m_vertices = new GridGPU<T>(numX, numY, m_dimension, Type::real);

  setVerticesRadial(inNumVertices, inRadius, inAngle, inCenterX, inCenterY);
}

template <typename T>
Polygon<T>::Polygon(const int inNumVertices, const T *const inVerticesX,
                    const T *const inVerticesY)
    : m_numVertices(inNumVertices) {
  // Essentially, for every dimension create an array of length inNumVertices.
  const int numX = inNumVertices;
  const int numY = 1;

  m_vertices = new GridGPU<T>(numX, numY, m_dimension, Type::real);

  setVertices(inVerticesX, inVerticesY);
}

template <typename T> Polygon<T>::~Polygon() {
  if (m_vertices)
    delete m_vertices;
}

template <typename T> Polygon<T> &Polygon<T>::operator=(Polygon<T> inOther) {
  swap(*this, inOther);
  return *this;
}

template <typename T>
void Polygon<T>::setVertices(const T *const inVerticesX,
                             const T *const inVerticesY) {
  VectorGPU<T> &vertices = m_vertices->getVector();
  const int offset = m_vertices->getFieldComponentOffset(1);

  for (int i = 0; i < m_numVertices; ++i) {
    vertices[i] = inVerticesX[i];
    vertices[i + offset] = inVerticesY[i];
  }
}

template <typename T>
void Polygon<T>::setVerticesRadial(const int inNumVertices, const T inRadius,
                                   const T inAngle, const T inCenterX,
                                   const T inCenterY) {
  if (inNumVertices == m_numVertices) {
    T verticesX[inNumVertices];
    T verticesY[inNumVertices];

    // The angle between vertices.
    const T angleStep =
        static_cast<T>(2.0 * M_PI) / static_cast<T>(inNumVertices);

    // A matter of taste if the first vertex should be on the x-axis or not.
    // Here it is not.
    const T angleOffset = angleStep * static_cast<T>(0.5) + inAngle;

    for (int i = 0; i < inNumVertices; ++i) {
      const T angle = angleStep * static_cast<T>(i) + angleOffset;

      verticesX[i] = inRadius * std::cos(angle) + inCenterX;
      verticesY[i] = inRadius * std::sin(angle) + inCenterY;
    }

    setVertices(verticesX, verticesY);
  }
}

template <typename T>
void Polygon<T>::transformPhysicalToNormalized(const int inNumVertices,
                                               const T inPhysicalWidth,
                                               T *const inOutVerticesX,
                                               T *const inOutVerticesY) {
  // TODO(niclas): I think this comes from 1 / (2 sqrt(2)) = 0.35355339059327373
  const T wn = static_cast<T>(0.35);

  // For convenience.
  const T scale = static_cast<T>(2.0) * wn / inPhysicalWidth;
  const T offset = -wn;

  for (int i = 0; i < inNumVertices; ++i) {
    const T xPhysical = inOutVerticesX[i];
    const T yPhysical = inOutVerticesY[i];

    // Previously it was written e.g.:
    // xNormalized = (xPhysical / inPhysicalWidth - 0.5) * 2.0 * wn;
    inOutVerticesX[i] = xPhysical * scale + offset;
    inOutVerticesY[i] = yPhysical * scale + offset;
  }
}
} // namespace conga

#endif // CONGA_POLYGON_H_
