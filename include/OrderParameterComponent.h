//===-- OrderParameterComponent.h - Component of order parameter group. ---===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage for a single order parameter component, added to the total order
/// parameter.
///
//===----------------------------------------------------------------------===//
#ifndef ORDER_PARAMETER_COMPONENT_H_
#define ORDER_PARAMETER_COMPONENT_H_

#include "FermiSurface.h"
#include "Grid.h"
#include "OrderParameterBasis.h"
#include "OrderParameterBulk.h"
#include "Parameters.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace scg {
template <typename T> class OrderParameterComponent {
public:
  // Constructors and destructor.
  OrderParameterComponent(const BasisFunctionSymmetry inSymmetry);
  OrderParameterComponent(const OrderParameterComponent<T> &other);
  OrderParameterComponent(const BasisFunctionSymmetry inSymmetry,
                          Parameters<T> *param, T Tcthis_Tc = 1.0);
  ~OrderParameterComponent();

  OrderParameterComponent<T> *clone() const;

  // Functions
  void initialize(const FermiSurface<T> *const fermiSurface);

  Parameters<T> *getParameters() const { return _param; }
  void setParameters(Parameters<T> *param) { _param = param; }

  GridGPU<T> *getOrderParameterField() const { return _deltaComp; }
  T getRelativeTc() const { return _Tcthis_Tc; }
  void setRelativeTc(const T Tc) { _Tcthis_Tc = Tc; }

  void computeMomentumDependence(GridGPU<T> *target, const T pX, const T pY,
                                 const T inCrystalAxesRotation,
                                 bool conjugateBasisFunction = false) const;

  void initCouplingConstant(const T *energies, const T *residues);

  void applyCouplingConstant();
  void applyCouplingConstant(GridGPU<T> *target) const;

  T deltaBulk() const;

  std::string getType() const { return m_basisFunction.getSymmetryStr(); };

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

  // The basis function of this order-parameter component.
  BasisFunction<T> m_basisFunction;

  // Order parameter component
  GridGPU<T> *_deltaComp;
};

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(
    const BasisFunctionSymmetry inSymmetry)
    : _param(0), _deltaComp(0), _Tcthis_Tc(1.0),
      m_couplingConstant(static_cast<T>(0.0)),
      m_isCouplingConstantInitialized(false),
       m_basisFunction(inSymmetry) {}

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(
    const OrderParameterComponent<T> &other)
    : _param(other._param), _Tcthis_Tc(other._Tcthis_Tc),
      m_couplingConstant(other.m_couplingConstant),
      m_isCouplingConstantInitialized(other.m_isCouplingConstantInitialized),
      m_basisFunction(other.m_basisFunction) {
  if (other._deltaComp) {
    _deltaComp = new GridGPU<T>(*other._deltaComp);
  }
}

template <typename T>
OrderParameterComponent<T>::OrderParameterComponent(
    const BasisFunctionSymmetry inSymmetry, Parameters<T> *param, T Tcthis_Tc)
    : _param(param), _deltaComp(0), _Tcthis_Tc(Tcthis_Tc),
      m_couplingConstant(static_cast<T>(0.0)),
      m_isCouplingConstantInitialized(false),
      m_basisFunction(inSymmetry) {
  // Do an initialization on creation so we can set initial value.
  // Note, pass a null pointer as the Fermi surface so it won't try to
  // initialize the basis function.
  initialize(nullptr);
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
  std::swap(A.m_basisFunction, B.m_basisFunction);
}

template <typename T>
OrderParameterComponent<T> *OrderParameterComponent<T>::clone() const {
  return new OrderParameterComponent<T>(*this);
}

template <typename T>
void OrderParameterComponent<T>::initialize(
    const FermiSurface<T> *const inFermiSurface) {
  int N = _param->getGridResolutionBase();

  if (_deltaComp && _deltaComp->dimX() != N) {
    _deltaComp->resample(N, N);
  } else {
    GridGPU<T>::initializeGrid(_deltaComp, N, N);
  }

  if (inFermiSurface) {
    m_basisFunction.normalize(inFermiSurface);
  }
}

template <typename T>
void OrderParameterComponent<T>::initCouplingConstant(const T *energies,
                                                      const T *residues) {
  // Get some info.
  const T T_Tc = _param->getTemperature();
  const int numEnergies = _param->getNumOzakiEnergies();

  // Compute (inverted) energy sum.
  T energySum = static_cast<T>(0.0);
  for (int i = 0; i < numEnergies; ++i) {
    energySum += residues[i] / energies[i];
  }
  energySum *= T_Tc;

  // Note, that here we implicitly assume that the basis functions are
  // normalized. I.e. <|eta|^2> = 1. Otherwise the energySum would be scaled by
  // <|eta|^2>.
  m_couplingConstant =
      (static_cast<T>(0.5) * T_Tc / (std::log(T_Tc / _Tcthis_Tc) + energySum));
  m_isCouplingConstantInitialized = true;

#ifndef NDEBUG
  // Print the value.
  std::cout << "m_couplingConstant: " << m_couplingConstant << "\n";
#endif
}

template <typename T> void OrderParameterComponent<T>::applyCouplingConstant() {
  applyCouplingConstant(_deltaComp);
}

template <typename T>
void OrderParameterComponent<T>::applyCouplingConstant(
    GridGPU<T> *target) const {
  if (m_isCouplingConstantInitialized) {
    (*target) *= m_couplingConstant;
  } else {
    throw std::logic_error("Coupling constant not initialized.");
  }
}

template <typename T>
void OrderParameterComponent<T>::computeMomentumDependence(
    GridGPU<T> *target, const T pX, const T pY, const T inCrystalAxesRotation,
    bool conjugateBasisFunction) const {
  T pXRotated;
  T pYRotated;
  utils::rotate(inCrystalAxesRotation, pX, pY, pXRotated, pYRotated);
  const thrust::complex<T> basisFunction =
      m_basisFunction.evaluate(pXRotated, pYRotated);
  (*target) *=
      (conjugateBasisFunction ? thrust::conj(basisFunction) : basisFunction);
}

template <typename T> T OrderParameterComponent<T>::deltaBulk() const {
  // TODO(niclas): Shouldn't the temperature be scaled with the relative Tc?
  const T temperature = _param->getTemperature();

  // Compute the s-wave bulk order parameter for this temperature.
  const T sWaveBulk = computeOrderParameterBulkSWave(temperature);

  // This is an a bit ugly, but it is faster to just solve for s-wave and
  // rescale it by the known d-wave value.
  // TODO(niclas): Perhaps this functionality should lie in
  // 'computeOrderParameterBulkSWave' instead? And rename that function if so.
  const BasisFunctionSymmetry symmetry = m_basisFunction.getSymmetry();
  T scale;
  switch (symmetry) {
  case BasisFunctionSymmetry::DWAVE_XY:
  case BasisFunctionSymmetry::DWAVE_X2_Y2: {
    scale = static_cast<T>(constants::D_WAVE_BULK / constants::S_WAVE_BULK);
    break;
  }
  default: {
    scale = static_cast<T>(1.0);
  }
  }
  return sWaveBulk * scale;
}
} // namespace scg

#endif // ORDER_PARAMETER_COMPONENT_H_
