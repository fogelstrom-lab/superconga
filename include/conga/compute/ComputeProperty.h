//===---------- ComputeProperty.h - ComputeProperty base class. -----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// ComputeProperty and its derived classes work in tandem with the
/// IntegrationIterator class. Functions in this base class are mostly 'hooks'
/// which gets called by IntegrationIterator as it integrates over energies and
/// angles.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_COMPUTE_PROPERTY_H_
#define CONGA_COMPUTE_PROPERTY_H_

#include "../ContextModule.h"
#include "../Parameters.h"
#include "../geometry/GeometryGroup.h"
#include "../green/GreensFunctionStatic.h"
#include "../grid.h"

namespace conga {
template <typename> class IntegrationIterator;

template <typename T> class ComputeProperty : public ContextModule<T> {
public:
  // Constructors and Destructors
  ComputeProperty();
  virtual ~ComputeProperty();

  ComputeProperty(Context<T> *ctx);

  GridGPU<T> *getGreensCurrentAngle() { return _greensCurrentAngle; }
  void initGreensAngle(const int N, const int numFields = 1);

  T getResidual() const { return m_residual; }
  void setResidual(const T inResidual) { m_residual = inResidual; }

  bool isConverged() const { return m_isConverged; }
  void setConverged(const bool inConverged = true) {
    m_isConverged = inConverged;
  }

  bool isPartiallyConverged() const { return m_isPartiallyConverged; }
  void setPartiallyConverged(const bool isPartiallyConverged = true) {
    m_isPartiallyConverged = isPartiallyConverged;
  }

  // Static Functions
  static GridGPU<T> computeSFMomentum(Parameters<T> *param,
                                      OrderParameter<T> *delta);

  // Virtual functions

  // Should return pointer to the computed field (order parameter, current,
  // etc.)
  virtual GridGPU<T> *getResult() = 0;

  // Gets called (by IntegrationIterator) once for every iteration, before any
  // integration takes place.
  virtual void initializeIteration(int numFields = 1) = 0;

  // Gets called once for each angle, before computing greens function.
  virtual void initializeGreensAngle() { _greensCurrentAngle->setZero(); }

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies.
  virtual void computeGreensAngle(const GridGPU<T> *const inGamma,
                                  const GridGPU<T> *const inGammaTilde) = 0;

  // Gets called once for each angle, after all calls (per angle) to
  // computeGreensAngle() are finished. Adds the result from
  // computeGreensAngle() to total.
  virtual void integrateGreensAngle(GeometryGroup<T> *geom,
                                    const int angleIdx) = 0;

  // Gets called once per iteration, after all angle integration of greens
  // functions are finished. Computes physical property from greens function.
  virtual void computePropertyFromGreens() = 0;

  // We don't always have to compute for every iteration. For example, current
  // and free energy only needs to be computed once the order parameter has
  // converged.
  void setCompute(bool doCompute) { m_doCompute = doCompute; }
  bool doCompute() const { return m_doCompute; }

private:
  // If the compute object is completely converged.
  bool m_isConverged;

  // If the compute object is partially converged. Only relevant for Keldysh
  // objects where we converge a few energies separately.
  bool m_isPartiallyConverged;

  bool m_doCompute;

  T m_residual;

  GridGPU<T> *_greensCurrentAngle;
};

template <typename T>
ComputeProperty<T>::ComputeProperty()
    : ContextModule<T>(), _greensCurrentAngle(0), m_isConverged(false),
      m_isPartiallyConverged(false), m_doCompute(true) {}

template <typename T>
ComputeProperty<T>::ComputeProperty(Context<T> *ctx)
    : ContextModule<T>(ctx), _greensCurrentAngle(0), m_isConverged(false),
      m_isPartiallyConverged(false), m_doCompute(true) {}

template <typename T> ComputeProperty<T>::~ComputeProperty() {
#ifndef NDEBUG
  std::cout << "~ComputeProperty()\n";
#endif
  if (_greensCurrentAngle) {
    delete _greensCurrentAngle;
  }
}

template <typename T>
void ComputeProperty<T>::initGreensAngle(const int N, const int numFields) {
  GridGPU<T>::initializeGrid(_greensCurrentAngle, N, N, numFields);
}
} // namespace conga

#endif // CONGA_COMPUTE_PROPERTY_H_
