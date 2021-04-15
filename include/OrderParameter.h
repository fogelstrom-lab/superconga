#ifndef ORDER_PARAMETER_H_
#define ORDER_PARAMETER_H_

#include "Context.h"
#include "ContextModule.h"
#include "Grid.h"
#include "OrderParameterComponent.h"
#include "Parameters.h"
#include "utils.h"
#include <string>

namespace scg {
template <typename T> class OrderParameter : public ContextModule<T> {
public:
  // Constructors and Destructors
  OrderParameter();
  virtual ~OrderParameter();

  OrderParameter(const OrderParameter<T> &other);
  OrderParameter(Context<T> *ctx);

  OrderParameter<T> &operator=(OrderParameter<T> right);

  // Functions
  virtual int initialize();

  void add(OrderParameterComponent<T> *delta);

  // Getters
  int getNumOrderParameterComponents() const;
  OrderParameterComponent<T> *getOrderParameterComponent(int delta_idx) const;
  GridGPU<T> *getOrderParameterComponentField(int delta_idx) const;

  // Perform rotation and return pointer
  GridGPU<T> *getOrderParameterFieldRotated(const int angleIdx);
  GridGPU<T> *getOrderParameterFieldRotated() const {
    return _orderParameterCompoundRotated;
  } // Return pointer to current rotational frame

  std::vector<std::string> getTypes() const;

private:
  template <typename S>
  friend void swap(OrderParameter<S> &A, OrderParameter<S> &B);

  std::vector<OrderParameterComponent<T> *> _orderParameterComponentList;

  GridGPU<T> *_orderParameterCompound;
  GridGPU<T> *_orderParameterCompoundRotated;
};

template <typename T>
OrderParameter<T>::OrderParameter()
    : ContextModule<T>(), _orderParameterCompound(0),
      _orderParameterCompoundRotated(0) {}

template <typename T>
OrderParameter<T>::OrderParameter(const OrderParameter<T> &other)
    : ContextModule<T>(other), _orderParameterCompound(0),
      _orderParameterCompoundRotated(0) {
  const int Nc = other.getNumOrderParameterComponents();

  for (int i = 0; i < Nc; ++i)
    _orderParameterComponentList.push_back(
        other._orderParameterComponentList[i]->clone());

  if (other._orderParameterCompound)
    _orderParameterCompound = new GridGPU<T>(*other._orderParameterCompound);

  if (other._orderParameterCompoundRotated)
    _orderParameterCompoundRotated =
        new GridGPU<T>(*other._orderParameterCompoundRotated);
}

template <typename T>
OrderParameter<T>::OrderParameter(Context<T> *ctx)
    : ContextModule<T>(ctx), _orderParameterCompound(0),
      _orderParameterCompoundRotated(0) {
  ctx->set(this);
}

template <typename T> OrderParameter<T>::~OrderParameter() {
  for (int i = 0; i < getNumOrderParameterComponents(); ++i) {
    delete _orderParameterComponentList[i];
  }

  if (_orderParameterCompound)
    delete _orderParameterCompound;
  if (_orderParameterCompoundRotated)
    delete _orderParameterCompoundRotated;
}

template <typename T>
OrderParameter<T> &OrderParameter<T>::operator=(OrderParameter<T> other) {
  swap(*this, other);
  return *this;
}

template <typename T> void swap(OrderParameter<T> &A, OrderParameter<T> &B) {
  std::swap(A._orderParameterComponentList, B._orderParameterComponentList);
  std::swap(A._orderParameterCompound, B._orderParameterCompound);
  std::swap(A._orderParameterCompoundRotated, B._orderParameterCompoundRotated);
}

template <typename T> int OrderParameter<T>::initialize() {
  const int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();
  const int N1 =
      ContextModule<T>::getParameters()->getGridResolutionCoherence();

  GridGPU<T>::initializeGrid(_orderParameterCompound, N0, N0);
  GridGPU<T>::initializeGrid(_orderParameterCompoundRotated, N1, N1);

  const FermiSurface<T> *const fermiSurface =
      ContextModule<T>::getFermiSurface();

  for (int i = 0; i < getNumOrderParameterComponents(); ++i) {
    // Initialize the component.
    _orderParameterComponentList[i]->initialize(fermiSurface);

    // Zero the values outside of the grain, so that we only compute residuals
    // of values that actually matter.
    GridGPU<T> *const component = getOrderParameterComponentField(i);
    ContextModule<T>::getGeometry()->zeroValuesOutsideDomainRef(component);
  }
  return 0;
}

template <typename T>
void OrderParameter<T>::add(OrderParameterComponent<T> *delta) {
  if (!(delta->getParameters()))
    delta->setParameters(ContextModule<T>::getParameters());

  _orderParameterComponentList.push_back(delta);
}

template <typename T>
int OrderParameter<T>::getNumOrderParameterComponents() const {
  return _orderParameterComponentList.size();
}

template <typename T>
OrderParameterComponent<T> *
OrderParameter<T>::getOrderParameterComponent(int delta_idx) const {
  const int Nc = getNumOrderParameterComponents();

  // TODO(niclas): Is this behaviour really what a user expects? Shouldn't we
  // throw an exception?
  if (delta_idx >= Nc)
    delta_idx = Nc - 1;

  return _orderParameterComponentList[delta_idx];
}

template <typename T>
GridGPU<T> *
OrderParameter<T>::getOrderParameterComponentField(int delta_idx) const {
  return getOrderParameterComponent(delta_idx)->getOrderParameterField();
}

template <typename T>
GridGPU<T> *
OrderParameter<T>::getOrderParameterFieldRotated(const int angleIdx) {
  // Get global model parameters
  const int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();
  const int N1 =
      ContextModule<T>::getParameters()->getGridResolutionCoherence();

  FermiSurface<T> *const fermiSurface = ContextModule<T>::getFermiSurface();

  // Rotating the order parameter field PHI degrees is equivalent to solving
  // the trajectories along -PHI, hence the minus sign in the function call
  // below.
  T momentumX;
  T momentumY;
  fermiSurface->getFermiMomentum(-angleIdx, momentumX, momentumY);

  // We need to sum all components with their basis function function
  // multiplied.
  GridGPU<T> opSummed(N0, N0);
  opSummed.setZero();
  for (int i = 0; i < getNumOrderParameterComponents(); ++i) {
    OrderParameterComponent<T> *component = getOrderParameterComponent(i);
    GridGPU<T> grid = *(component->getOrderParameterField());
    component->computeMomentumDependence(&grid, momentumX, momentumY);
    opSummed += grid;
  }

  // Extend the values at the boundary. Note that this is done using the
  // non-rotated geometry.
  GeometryGroup<T> *const geometry = ContextModule<T>::getGeometry();
  geometry->extendValuesAtBoundaryRef(&opSummed);

  // Rotate the order parameter and insert it in the larger grid
  // _orderParameterCompoundRotated.
  const T systemAngle = fermiSurface->getSystemAngle(angleIdx);
  _orderParameterCompoundRotated->setZero();
  _orderParameterCompoundRotated->insertRotatedGridAdditive(&opSummed,
                                                            systemAngle, 0);
  return _orderParameterCompoundRotated;
}

template <typename T>
std::vector<std::string> OrderParameter<T>::getTypes() const {
  const int numOrderParameterComponents = getNumOrderParameterComponents();
  std::vector<std::string> types;
  types.reserve(numOrderParameterComponents);

  for (int i = 0; i < numOrderParameterComponents; ++i) {
    const std::string componentType = getOrderParameterComponent(i)->getType();
    types.emplace_back(componentType);
  }
  return types;
}
} // namespace scg

#endif // ORDER_PARAMETER_H_
