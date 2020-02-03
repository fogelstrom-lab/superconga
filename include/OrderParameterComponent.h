#ifndef ORDER_PARAMETER_COMPONENT_H_
#define ORDER_PARAMETER_COMPONENT_H_

#include "FermiSurface.h"
#include "Grid.h"
#include "Parameters.h"
#include <cmath>
#include <math.h>
#include <stdexcept>

template <typename T> class OrderParameterComponent {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  OrderParameterComponent();
  virtual ~OrderParameterComponent();

  OrderParameterComponent(const OrderParameterComponent<T> &other);

  OrderParameterComponent(Parameters<T> *param, T Tcthis_Tc = 1.0);

  virtual OrderParameterComponent<T> *clone() const = 0;

  // Functions
  virtual int initialize();

  Parameters<T> *getParameters() const { return _param; }
  void setParameters(Parameters<T> *param) { _param = param; }

  GridGPU *getOrderParameterField() const { return _deltaComp; }
  T getRelativeTc() const { return _Tcthis_Tc; }
  void setRelativeTc(const T Tc) { _Tcthis_Tc = Tc; }

  virtual void computeMomentumDependence(GridGPU *target, const T pX,
                                         const T pY,
                                         bool conjugateBasis = false) = 0;

  virtual void initBasisNormalization(FermiSurface<T> *fermiSurface) = 0;

  void initCouplingConstant(const T *energies, const T *residues);

  void applyCouplingConstant();
  void applyCouplingConstant(GridGPU *target);

  virtual double deltaBulk() const;

  char *getType() const { return _type; }
  void setType(char *type) { _type = type; }

private:
  template <typename S>
  friend void swap(OrderParameterComponent<S> &A,
                   OrderParameterComponent<S> &B);

  // Global model parameters
  Parameters<T> *_param;

  // Relative critical temperature  Tc_thiscomponent / Tc
  T _Tcthis_Tc;

  // Store coupling constant, calculate only once
  // CAUTION!!! Upon changing T_Tc/energies/similar, need to re-calculate
  T m_couplingConstant;
  // True if coupling constant has been initialized, then it never needs to be
  // re-calculated
  bool m_isCouplingConstantInitialized;

  // Order parameter component
  GridGPU *_deltaComp;

  char *_type;
};

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent()
    : _param(0), _deltaComp(0), _Tcthis_Tc(1.0), m_couplingConstant(0),
      m_isCouplingConstantInitialized(false) {}

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(
    const OrderParameterComponent<T> &other)
    : _param(other._param), _Tcthis_Tc(other._Tcthis_Tc),
      m_couplingConstant(other.m_couplingConstant),
      m_isCouplingConstantInitialized(other.m_isCouplingConstantInitialized) {
  _type = other._type;
  if (other._deltaComp)
    _deltaComp = new GridGPU(*other._deltaComp);
}

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(Parameters<T> *param,
                                                    T Tcthis_Tc)
    : _param(param), _deltaComp(0), _Tcthis_Tc(Tcthis_Tc),
      m_couplingConstant(0), m_isCouplingConstantInitialized(false) {
  // Do an initialization on creation so we can set initial value
  initialize();
}

template <typename T> OrderParameterComponent<T>::~OrderParameterComponent() {
  if (_deltaComp)
    delete _deltaComp;
}

template <typename T>
void swap(OrderParameterComponent<T> &A, OrderParameterComponent<T> &B) {
  std::swap(A._param, B._param);
  std::swap(A._Tcthis_Tc, B._Tcthis_Tc);
  std::swap(A._deltaComp, B._deltaComp);
  std::swap(A._type, B._type);
  std::swap(A.m_couplingConstant, B.m_couplingConstant);
  std::swap(A.m_isCouplingConstantInitialized,
            B.m_isCouplingConstantInitialized);
}

template <typename T> int OrderParameterComponent<T>::initialize() {
  int N = _param->getGridResolutionBase();

  if (_deltaComp && _deltaComp->dimX() != N)
    _deltaComp->resample(N, N);
  else
    GridGPU::initializeGrid(_deltaComp, N, N);

  return 0;
}

template <typename T>
void OrderParameterComponent<T>::initCouplingConstant(const T *energies,
                                                      const T *residues) {
  // Get some info.
  const T T_Tc = _param->getTemperature();
  const int numEnergies = _param->getNumMatsubaraEnergies();

  // For convenience.
  const T piT = static_cast<T>(M_PI) * T_Tc;

  // Compute (inverted) energy sum.
  T energySum = static_cast<T>(0.0);
  for (int i = 0; i < numEnergies; ++i) {
    energySum += residues[i] / energies[i];
  }
  // A factor of two because we are excluding negative energies.
  energySum *= static_cast<T>(2.0) * piT;

  // Note, that here we implicitly assume that the basis functions are
  // normalized. I.e. <|eta|^2> = 1. Otherwise the energySum would be scaled by
  // <|eta|^2>.
  m_couplingConstant = (piT / (std::log(T_Tc / _Tcthis_Tc) + energySum));
  m_isCouplingConstantInitialized = true;

  // Print the value.
  std::cout << "m_couplingConstant: " << m_couplingConstant << "\n";
}

template <typename T> void OrderParameterComponent<T>::applyCouplingConstant() {
  applyCouplingConstant(_deltaComp);
}

template <typename T>
void OrderParameterComponent<T>::applyCouplingConstant(GridGPU *target) {
  if (m_isCouplingConstantInitialized) {
    (*target) *= m_couplingConstant;
  } else {
    throw std::logic_error("Coupling constant not initialized.");
  }
}

template <typename T> double OrderParameterComponent<T>::deltaBulk() const {
  T T_Tc = _param->getTemperature();

  const double deltaConvergence = 1E-9;

  const double gMix = -0.1;
  const int numMatsubara = 1000.0 / T_Tc;

  // Delta, initial guess
  double delta = 1.76 * sqrt(1.0 - T_Tc);

  int isDeltaConverged = 0;
  double matsubaraSum, eps, g, deltaDiff;

  while (!isDeltaConverged) {
    matsubaraSum = 0.0;

    for (int i = 0; i <= numMatsubara; ++i) {
      eps = M_PI * T_Tc * (2.0 * i + 1.0);
      matsubaraSum += 1.0 / sqrt(eps * eps + delta * delta) - 1.0 / eps;
    }
    matsubaraSum *= delta;

    g = delta * log(T_Tc) - 2.0 * M_PI * T_Tc * matsubaraSum;
    deltaDiff = g * gMix;
    delta += deltaDiff;

    if (abs(deltaDiff / delta) < deltaConvergence)
      isDeltaConverged = 1;
  }

  return delta;
}

#endif // ORDER_PARAMETER_COMPONENT_H_
