//===----------- OrderParameter.h - Order parameter container. ------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Order parameter container, for storing different order parameter components.
///
//===----------------------------------------------------------------------===//
#ifndef ORDER_PARAMETER_H_
#define ORDER_PARAMETER_H_

#include "Context.h"
#include "ContextModule.h"
#include "Grid.h"
#include "OrderParameterComponent.h"
#include "Parameters.h"
#include "utils.h"
#include <string>

namespace conga {
template <typename T> class OrderParameter : public ContextModule<T> {
public:
  /// \brief Constructor.
  OrderParameter();

  /// \brief Constructor.
  ///
  /// \param inOther Another order parameter.
  OrderParameter(const OrderParameter<T> &inOther);

  /// \brief Constructor.
  ///
  /// \param inOutContext Context pointer.
  OrderParameter(Context<T> *inOutContext);

  /// \brief Destructor.
  ~OrderParameter();

  OrderParameter<T> &operator=(OrderParameter<T> inOther);

  /// \brief Set the rotation of the crystal ab-axes.
  ///
  /// \param inCrystalAxesRotation The rotation angle in radians.
  ///
  /// \return Void.
  void setCrystalAxesRotation(const T inCrystalAxesRotation) {
    m_crystalAxesRotation = inCrystalAxesRotation;
  }

  /// \brief Get the rotation of the crystal ab-axes.
  ///
  /// \return The rotation angle in radians.
  T getCrystalAxesRotation() const { return m_crystalAxesRotation; }

  /// \brief Initialize the order parameter.
  ///
  /// \return Whether the initialization was a success.
  // TODO(niclas): Return bool or status enum.
  int initialize() override;

  /// \brief Add component to the order parameter.
  ///
  /// \param inComponent The component to add.
  ///
  /// \return Void.
  void add(OrderParameterComponent<T> *inComponent);

  /// \brief Get the number of components.
  ///
  /// \return The number of components.
  int getNumComponents() const;

  /// \brief Get a particular component.
  ///
  /// \param inIndex The index of the component.
  ///
  /// \return A pointer the the component.
  OrderParameterComponent<T> *getComponent(int inIndex) const;

  /// \brief Get a particular component's grid.
  ///
  /// \param inIndex The index of the component.
  ///
  /// \return A pointer the the grid.
  GridGPU<T> *getComponentGrid(int inIndex) const;

  /// \brief Get the rotated grid of sum of all components with momentum
  /// dependence for a particular angle index.
  ///
  /// \param inAngleIdx The angle index, which determines the momentum and the
  /// rotation angle.
  ///
  /// \return The rotated grid of the total order-parameter for this index.
  const GridGPU<T> *getRotatedGrid(const int inAngleIdx);

  /// \brief Get the rotated grid of sum of all components with momentum
  /// dependence for the latest provided angle index.
  ///
  /// \return The rotated grid of the total order-parameter for the latest
  /// index.
  const GridGPU<T> *getRotatedGrid() const { return m_rotated; }

  /// \brief Get the types of all components.
  ///
  /// \return The types, e.g. {"s", "dxy"}.
  std::vector<std::string> getTypes() const;

private:
  /// \brief Swap to order parameters.
  ///
  /// \param inA The first order parameter.
  /// \param inB The second order parameter.
  ///
  /// \return Void.
  template <typename S>
  friend void swap(OrderParameter<S> &inA, OrderParameter<S> &inB);

  // The rotation of the crystal ab-axes.
  T m_crystalAxesRotation;

  // A list of all components.
  std::vector<OrderParameterComponent<T> *> m_components;

  // The total rotated order-parameter with momentum dependence.
  GridGPU<T> *m_rotated;
};

template <typename T>
OrderParameter<T>::OrderParameter()
    : ContextModule<T>(), m_rotated(nullptr),
      m_crystalAxesRotation(static_cast<T>(0)) {}

template <typename T>
OrderParameter<T>::OrderParameter(const OrderParameter<T> &inOther)
    : ContextModule<T>(inOther) {
  m_crystalAxesRotation = inOther.getCrystalAxesRotation();

  const int numComponents = inOther.getNumComponents();

  for (int i = 0; i < numComponents; ++i)
    m_components.push_back(inOther.m_components[i]->clone());

  if (inOther.m_rotated)
    m_rotated = new GridGPU<T>(*inOther.m_rotated);
}

template <typename T>
OrderParameter<T>::OrderParameter(Context<T> *inOutContext)
    : ContextModule<T>(inOutContext), m_rotated(nullptr),
      m_crystalAxesRotation(static_cast<T>(0)) {
  inOutContext->set(this);
}

template <typename T> OrderParameter<T>::~OrderParameter() {
  for (int i = 0; i < getNumComponents(); ++i) {
    delete m_components[i];
  }

  if (m_rotated) {
    delete m_rotated;
  }
}

template <typename T>
OrderParameter<T> &OrderParameter<T>::operator=(OrderParameter<T> inOther) {
  swap(*this, inOther);
  return *this;
}

template <typename T>
void swap(OrderParameter<T> &inA, OrderParameter<T> &inB) {
  std::swap(inA.m_components, inB.m_components);
  std::swap(inA.m_rotated, inB.m_rotated);
}

template <typename T> int OrderParameter<T>::initialize() {
  const int N1 =
      ContextModule<T>::getParameters()->getGridResolutionCoherence();

  GridGPU<T>::initializeGrid(m_rotated, N1, N1);

  const FermiSurface<T> *const fermiSurface =
      ContextModule<T>::getFermiSurface();

  for (int i = 0; i < getNumComponents(); ++i) {
    // Initialize the component.
    m_components[i]->initialize(fermiSurface);

    // Zero the values outside of the grain, so that we only compute residuals
    // of values that actually matter.
    GridGPU<T> *const component = getComponentGrid(i);
    ContextModule<T>::getGeometry()->zeroValuesOutsideDomainRef(component);
  }
  return 0;
}

template <typename T>
void OrderParameter<T>::add(OrderParameterComponent<T> *inComponent) {
  if (!(inComponent->getParameters())) {
    inComponent->setParameters(ContextModule<T>::getParameters());
  }

  m_components.push_back(inComponent);
}

template <typename T> int OrderParameter<T>::getNumComponents() const {
  return m_components.size();
}

template <typename T>
OrderParameterComponent<T> *OrderParameter<T>::getComponent(int inIndex) const {
  const int numComponents = getNumComponents();

  // TODO(niclas): Is this behaviour really what a user expects? Shouldn't we
  // throw an exception?
  if (inIndex >= numComponents) {
    inIndex = numComponents - 1;
  }

  return m_components[inIndex];
}

template <typename T>
GridGPU<T> *OrderParameter<T>::getComponentGrid(int inIndex) const {
  return getComponent(inIndex)->getGrid();
}

template <typename T>
const GridGPU<T> *OrderParameter<T>::getRotatedGrid(const int inAngleIdx) {
  // Get global model parameters
  const int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();

  FermiSurface<T> *const fermiSurface = ContextModule<T>::getFermiSurface();

  // Rotating the order parameter field PHI degrees is equivalent to solving
  // the trajectories along -PHI, hence the minus sign in the function call
  // below.
  T momentumX;
  T momentumY;
  fermiSurface->getFermiMomentum(-inAngleIdx, momentumX, momentumY);

  // We need to sum all components with their basis function function
  // multiplied.
  GridGPU<T> opSummed(N0, N0);
  opSummed.setZero();
  for (int i = 0; i < getNumComponents(); ++i) {
    OrderParameterComponent<T> *component = getComponent(i);
    GridGPU<T> grid = *(component->getGrid());
    component->computeMomentumDependence(&grid, momentumX, momentumY,
                                         m_crystalAxesRotation);
    opSummed += grid;
  }

  // Extend the values at the boundary. Note that this is done using the
  // non-rotated geometry.
  GeometryGroup<T> *const geometry = ContextModule<T>::getGeometry();
  geometry->extendValuesAtBoundaryRef(&opSummed);

  // Rotate the order parameter and insert it in the larger grid
  // m_rotated.
  const T systemAngle = fermiSurface->getSystemAngle(inAngleIdx);
  m_rotated->setZero();
  m_rotated->insertRotatedGridAdditive(&opSummed, systemAngle, 0);
  return m_rotated;
}

template <typename T>
std::vector<std::string> OrderParameter<T>::getTypes() const {
  const int numOrderParameterComponents = getNumComponents();
  std::vector<std::string> types;
  types.reserve(numOrderParameterComponents);

  for (int i = 0; i < numOrderParameterComponents; ++i) {
    const std::string componentType = getComponent(i)->getType();
    types.emplace_back(componentType);
  }
  return types;
}
} // namespace conga

#endif // ORDER_PARAMETER_H_
