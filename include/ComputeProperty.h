// ComputeProperty and its derived classes work in tandem with
// the IntegrationIterator class. Functions in this base class are mostly
// 'hooks' which gets called by IntegrationIterator as it integrates over
// energies and angles.

#ifndef COMPUTE_PROPERTY_H_
#define COMPUTE_PROPERTY_H_

#include "ContextModule.h"
#include "GeometryGroup.h"
#include "GreensFunctionStatic.h"
#include "Grid.h"
#include "IntegrationIterator.h"
#include "ParameterIterator.h"
#include "Parameters.h"

namespace scg {
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
  //
  virtual IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1) = 0;

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
  virtual void computeGreensAngle(ParameterIterator<int> *iter,
                                  GridGPU<T> *gamma, GridGPU<T> *gammabar) = 0;

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

namespace compute_property_internal {
template <typename T>
__global__ void addFermiFunction_kernel(T *data, T *e_re, T T_Tc, int Ne,
                                        int N) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int te = blockIdx.z;

  if (tx < N && ty < N && te < Ne) {
    const int idx = te * N * N + ty * N + tx;

    const T E = e_re[te];
    const T fermi = (T)1.0 / (exp(E / T_Tc) + (T)1.0);

    data[idx] *= fermi;
  }
}

template <typename T>
__global__ void addGauge_Kernel(T *data_x, T *data_y, const T h, const T phi,
                                const int N) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;

  if (tx < N) {
    const int idx = ty * N + tx;
    const T coord_x = (-(T)0.5 * (N - 1) + tx) * h;
    const T coord_y = (-(T)0.5 * (N - 1) + ty) * h;

    T r2 = coord_x * coord_x + coord_y * coord_y;

    if (r2 < (T)1E-07)
      r2 = (T)1E-07;

    const T mult = phi / ((T)2.0 * M_PI * r2);

    data_x[idx] += mult * (-coord_y);
    data_y[idx] += mult * coord_x;
  }
}
} // namespace compute_property_internal

template <typename T>
ComputeProperty<T>::ComputeProperty()
    : ContextModule<T>(), _greensCurrentAngle(0), m_isConverged(false),
      m_isPartiallyConverged(false), m_doCompute(true) {}

template <typename T>
ComputeProperty<T>::ComputeProperty(Context<T> *ctx)
    : ContextModule<T>(ctx), _greensCurrentAngle(0), m_isConverged(false),
      m_isPartiallyConverged(false), m_doCompute(true) {}

template <typename T> ComputeProperty<T>::~ComputeProperty() {
  std::cout << "~ComputeProperty()\n";
  if (_greensCurrentAngle) {
    delete _greensCurrentAngle;
  }
}

template <typename T>
void ComputeProperty<T>::initGreensAngle(const int N, const int numFields) {
  GridGPU<T>::initializeGrid(_greensCurrentAngle, N, N, numFields);
}

template <typename T>
GridGPU<T> ComputeProperty<T>::computeSFMomentum(Parameters<T> *param,
                                                 OrderParameter<T> *delta) {
  const int N = param->getGridResolutionBase();
  const T h = param->getGridElementSize();
  const T Phi = param->getMagneticFieldStrength();

  GridGPU<T> *delta_grid = delta->getOrderParameterField();

  GridGPU<T> phaseGradient = delta_grid->phaseGradient(h);
  phaseGradient *= 0.5;

  // Add gauge
  dim3 bpg, tpb;
  phaseGradient.getKernelDimensions(bpg, tpb);

  compute_property_internal::addGauge_Kernel<<<bpg, tpb>>>(
      phaseGradient.getDataPointer(0), phaseGradient.getDataPointer(1), h, Phi,
      N);

  return phaseGradient;
}
} // namespace scg

#endif // COMPUTE_PROPERTY_H_
