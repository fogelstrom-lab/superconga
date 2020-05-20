#ifndef ORDER_PARAMETER_COMPONENT_H_
#define ORDER_PARAMETER_COMPONENT_H_

#include "FermiSurface.h"
#include "Grid.h"
#include "OrderParameterBulk.h"
#include "Parameters.h"
#include <cmath>
#include <stdexcept>
#include <string>

template <typename T> class OrderParameterComponent {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and destructor.
  OrderParameterComponent();
  OrderParameterComponent(const OrderParameterComponent<T> &other);
  OrderParameterComponent(Parameters<T> *param, T Tcthis_Tc = 1.0);
  virtual ~OrderParameterComponent();

  virtual OrderParameterComponent<T> *clone() const = 0;

  // Functions
  virtual void initialize();

  Parameters<T> *getParameters() const { return _param; }
  void setParameters(Parameters<T> *param) { _param = param; }

  GridGPU *getOrderParameterField() const { return _deltaComp; }
  T getRelativeTc() const { return _Tcthis_Tc; }
  void setRelativeTc(const T Tc) { _Tcthis_Tc = Tc; }

  void setCrystalAxesRotation(const T inCrystalAxesRotation) {
    m_crystalAxesRotation = inCrystalAxesRotation;
  }
  T getCrystalAxesRotation() const { return m_crystalAxesRotation; }

  virtual void
  computeMomentumDependence(GridGPU *target, const T pX, const T pY,
                            bool conjugateBasisFunction = false) const = 0;

  virtual void
  initBasisFunctionNormalization(FermiSurface<T> *fermiSurface) = 0;

  void initCouplingConstant(const T *energies, const T *residues);

  void applyCouplingConstant();
  void applyCouplingConstant(GridGPU *target) const;

  virtual T deltaBulk() const;

  void setBasisFunctionNormalization(const T inBasisFunctionNormalization) {
    m_basisFunctionNormalization = inBasisFunctionNormalization;
  }
  T getBasisFunctionNormalization() const {
    return m_basisFunctionNormalization;
  }

  virtual std::string getType() const = 0;

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

  // Specified in radians, as a rotational offset from the default.
  // A value of 0 (default) aligns the crystal lattice along the x- and y- axes.
  T m_crystalAxesRotation;

  // Factor to make <|eta|^2> = 1.
  T m_basisFunctionNormalization;

  // Order parameter component
  GridGPU *_deltaComp;
};

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent()
    : _param(0), _deltaComp(0), _Tcthis_Tc(1.0),
      m_couplingConstant(static_cast<T>(0.0)),
      m_isCouplingConstantInitialized(false),
      m_crystalAxesRotation(static_cast<T>(0.0)),
      m_basisFunctionNormalization(static_cast<T>(1.0)) {}

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(
    const OrderParameterComponent<T> &other)
    : _param(other._param), _Tcthis_Tc(other._Tcthis_Tc),
      m_couplingConstant(other.m_couplingConstant),
      m_isCouplingConstantInitialized(other.m_isCouplingConstantInitialized),
      m_crystalAxesRotation(other.m_crystalAxesRotation),
      m_basisFunctionNormalization(other.m_basisFunctionNormalization) {
  if (other._deltaComp) {
    _deltaComp = new GridGPU(*other._deltaComp);
  }
}

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(Parameters<T> *param,
                                                    T Tcthis_Tc)
    : _param(param), _deltaComp(0), _Tcthis_Tc(Tcthis_Tc),
      m_couplingConstant(static_cast<T>(0.0)),
      m_isCouplingConstantInitialized(false),
      m_crystalAxesRotation(static_cast<T>(0.0)),
      m_basisFunctionNormalization(static_cast<T>(1.0)) {
  // Do an initialization on creation so we can set initial value
  initialize();
}

template <typename T> OrderParameterComponent<T>::~OrderParameterComponent() {
  if (_deltaComp) {
    delete _deltaComp;
  }
}

template <typename T>
void swap(OrderParameterComponent<T> &A, OrderParameterComponent<T> &B) {
  std::swap(A._param, B._param);
  std::swap(A._Tcthis_Tc, B._Tcthis_Tc);
  std::swap(A._deltaComp, B._deltaComp);
  std::swap(A.m_couplingConstant, B.m_couplingConstant);
  std::swap(A.m_isCouplingConstantInitialized,
            B.m_isCouplingConstantInitialized);
  std::swap(A.m_crystalAxesRotation, B.m_crystalAxesRotation);
  std::swap(A.m_basisFunctionNormalization, B.m_basisFunctionNormalization);
}

template <typename T> void OrderParameterComponent<T>::initialize() {
  int N = _param->getGridResolutionBase();

  if (_deltaComp && _deltaComp->dimX() != N) {
    _deltaComp->resample(N, N);
  } else {
    GridGPU::initializeGrid(_deltaComp, N, N);
  }
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
void OrderParameterComponent<T>::applyCouplingConstant(GridGPU *target) const {
  if (m_isCouplingConstantInitialized) {
    (*target) *= m_couplingConstant;
  } else {
    throw std::logic_error("Coupling constant not initialized.");
  }
}

template <typename T> T OrderParameterComponent<T>::deltaBulk() const {
  // TODO(Niclas): Is it really necessary to wrap it here?
  const T temperature = _param->getTemperature();
  return computeOrderParameterBulkSWave(temperature);
}

#endif // ORDER_PARAMETER_COMPONENT_H_
