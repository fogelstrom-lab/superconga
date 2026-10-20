//===------------------------ Disc.h - Disc class. ------------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Disc class, providing a basic shape implementation for a disc.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_DISC_H_
#define CONGA_DISC_H_

namespace conga {
template <typename T> class Disc {
public:
  Disc();
  ~Disc(){};

  // Copy constructor
  Disc(const Disc &other);

  Disc(const T inRadius, const T inCenterX = static_cast<T>(0.0),
       const T inCenterY = static_cast<T>(0.0));

  Disc &operator=(Disc other);

  void setRadius(const T inRadius);
  void setPosX(const T inCenterX);
  void setPosY(const T inCenterY);

  T getRadius() const { return m_radius; }
  T getPosX() const { return m_centerX; }
  T getPosY() const { return m_centerY; }

private:
  friend void swap(Disc &A, Disc &B) {
    std::swap(A.m_radius, B.m_radius);
    std::swap(A.m_centerX, B.m_centerX);
    std::swap(A.m_centerY, B.m_centerY);
  }

  // Radius of disc
  T m_radius;

  // Center of disc, stored in centered coordinates -0.5 < x,y < 0.5
  T m_centerX, m_centerY;
};

template <typename T>
Disc<T>::Disc()
    : m_radius(static_cast<T>(0.5)), m_centerX(static_cast<T>(0.0)),
      m_centerY(static_cast<T>(0.0)) {}

template <typename T>
Disc<T>::Disc(const T inRadius, const T inCenterX, const T inCenterY)
    : m_radius(inRadius), m_centerX(inCenterX), m_centerY(inCenterY) {}

template <typename T>
Disc<T>::Disc(const Disc<T> &other)
    : m_radius(other.m_radius), m_centerX(other.m_centerX),
      m_centerY(other.m_centerY) {}

template <typename T> Disc<T> &Disc<T>::operator=(Disc<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void Disc<T>::setRadius(const T inRadius) {
  m_radius = inRadius;
}

template <typename T> void Disc<T>::setPosX(const T inCenterX) {
  m_centerX = inCenterX;
}

template <typename T> void Disc<T>::setPosY(const T inCenterY) {
  m_centerY = inCenterY;
}
} // namespace conga

#endif // CONGA_DISC_H_
