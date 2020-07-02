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
  typedef typename Grid<T>::GPU GridGPU;

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
  GridGPU *getOrderParameterComponentField(int delta_idx) const;

  GridGPU *getOrderParameterField();
  // Perform rotation and return pointer
  GridGPU *getOrderParameterFieldRotated(const int angleIdx);
  GridGPU *getOrderParameterFieldRotated() const {
    return _orderParameterCompoundRotated;
  } // Return pointer to current rotational frame

  std::string getTypes() const;

private:
  template <typename S>
  friend void swap(OrderParameter<S> &A, OrderParameter<S> &B);

  std::vector<OrderParameterComponent<T> *> _orderParameterComponentList;

  GridGPU *_orderParameterCompound;
  GridGPU *_orderParameterCompoundRotated;
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
    _orderParameterCompound = new GridGPU(*other._orderParameterCompound);

  if (other._orderParameterCompoundRotated)
    _orderParameterCompoundRotated =
        new GridGPU(*other._orderParameterCompoundRotated);
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
  int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();
  int N1 = ContextModule<T>::getParameters()->getGridResolutionCoherence();

  GridGPU::initializeGrid(_orderParameterCompound, N0, N0);
  GridGPU::initializeGrid(_orderParameterCompoundRotated, N1, N1);

  FermiSurface<T> *fermiSurface = ContextModule<T>::getFermiSurface();

  for (int i = 0; i < getNumOrderParameterComponents(); ++i) {
    _orderParameterComponentList[i]->initialize(fermiSurface);
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
typename Grid<T>::GPU *
OrderParameter<T>::getOrderParameterComponentField(int delta_idx) const {
  return getOrderParameterComponent(delta_idx)->getOrderParameterField();
}

template <typename T>
typename Grid<T>::GPU *OrderParameter<T>::getOrderParameterField() {
  // Add delta components together
  _orderParameterCompound->setZero();

  for (int i = 0; i < getNumOrderParameterComponents(); ++i)
    *_orderParameterCompound +=
        (*(getOrderParameterComponent(i)->getOrderParameterField()));

  return _orderParameterCompound;
}

template <typename T>
typename Grid<T>::GPU *
OrderParameter<T>::getOrderParameterFieldRotated(const int angleIdx) {
  // Get global model parameters
  int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();
  int N1 = ContextModule<T>::getParameters()->getGridResolutionCoherence();

  FermiSurface<T> *const fermiSurface = ContextModule<T>::getFermiSurface();
  const T systemAngle = fermiSurface->getSystemAngle(angleIdx);

  // Rotating the order parameter field PHI degrees is equivalent to solving
  // the trajectories along -PHI, hence the minus sign in the function call
  // below.
  T momentumX;
  T momentumY;
  fermiSurface->getFermiMomentum(-angleIdx, momentumX, momentumY);

  // Add delta components together in rotated frame
  OrderParameterComponent<T> *delta_tmp;

  _orderParameterCompoundRotated->setZero();

  GridGPU delta_comp_tmp(N0, N0);

  for (int i = 0; i < getNumOrderParameterComponents(); ++i) {
    delta_tmp = getOrderParameterComponent(i);
    delta_comp_tmp = *(delta_tmp->getOrderParameterField());

    delta_tmp->computeMomentumDependence(&delta_comp_tmp, momentumX, momentumY);

    _orderParameterCompoundRotated->insertRotatedGridAdditive(&delta_comp_tmp,
                                                              systemAngle, 0);
  }

  return _orderParameterCompoundRotated;
}

template <typename T> std::string OrderParameter<T>::getTypes() const {
  std::string types;

  const int numOrderParameterComponents = getNumOrderParameterComponents();
  for (int i = 0; i < numOrderParameterComponents; ++i) {
    const std::string componentType = getOrderParameterComponent(i)->getType();
    types.append(componentType);
  }

  return types;
}
} // namespace scg

#endif // ORDER_PARAMETER_H_
