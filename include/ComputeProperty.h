// ComputeProperty and its derived classes work in tandem with
// the IntegrationIterator class. Functions in this base class are mostly
// 'hooks' which gets called by IntegrationIterator as it integrates over
// energies and angles.

#ifndef COMPUTEPROPERTY_H
#define COMPUTEPROPERTY_H

#include <ContextModule.h>
#include <GeometryGroup.h>
#include <GreensFunctionStatic.h>
#include <Grid.h>
#include <IntegrationIterator.h>
#include <ParameterIterator.h>
#include <Parameters.h>

template <typename> class IntegrationIterator;

template <typename T> class ComputeProperty : public ContextModule<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors
  ComputeProperty();
  virtual ~ComputeProperty();

  ComputeProperty(Context<T> *ctx);

  GridGPU *getGreensCurrentAngle() { return _greensCurrentAngle; }
  void initGreensAngle(const int N, const int numFields = 1);

  T getResidual() const { return _residual; }
  void setResidual(T res) { _residual = res; }

  bool isConverged() const { return _isConverged; }
  void setConverged(bool converged = true) { _isConverged = converged; }

  // Static Functions
  static GridGPU computeSFMomentum(Parameters<T> *param,
                                   OrderParameter<T> *delta);

  // Virtual functions
  //
  virtual IntegrationIterator<T> *
  createIntegrationIterator(int numEnergiesPerBlock = -1) = 0;

  // Should return pointer to the computed field (order parameter, current,
  // etc.)
  virtual GridGPU *getResult() = 0;

  // Gets called (by IntegrationIterator) once for every iteration, before any
  // integration takes place.
  virtual void initializeIteration(int numFields = 1) = 0;

  // Gets called once for each angle, before computing greens function.
  virtual void initializeGreensAngle() { _greensCurrentAngle->setZero(); }

  // Computes greens function for current angle. This may be called multiple
  // times as we integrate over many energies.
  virtual void computeGreensAngle(ParameterIterator<int> *iter, GridGPU *gamma,
                                  GridGPU *gammabar,
                                  GeometryGroup<T> *geom) = 0;

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
  void setCompute(bool doCompute) { _doCompute = doCompute; }
  bool doCompute() const { return _doCompute; }

private:
  bool _isConverged;
  bool _doCompute;

  T _residual;

  GridGPU *_greensCurrentAngle;
};

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
ComputeProperty<T>::ComputeProperty()
    : ContextModule<T>(), _greensCurrentAngle(0), _isConverged(false),
      _doCompute(true) {}

template <typename T>
ComputeProperty<T>::ComputeProperty(Context<T> *ctx)
    : ContextModule<T>(ctx), _greensCurrentAngle(0), _isConverged(false),
      _doCompute(true) {}

template <typename T> ComputeProperty<T>::~ComputeProperty() {
  std::cout << "~ComputeProperty()\n";
  if (_greensCurrentAngle)
    delete _greensCurrentAngle;
}

template <typename T>
void ComputeProperty<T>::initGreensAngle(const int N, const int numFields) {
  GridGPU::initializeGrid(_greensCurrentAngle, N, N, numFields);
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

template <typename T>
typename Grid<T>::GPU
ComputeProperty<T>::computeSFMomentum(Parameters<T> *param,
                                      OrderParameter<T> *delta) {
  const int N = param->getGridResolutionBase();
  const T h = param->getGridElementSize();
  const T Phi = param->getMagneticFieldStrength();

  GridGPU *delta_grid = delta->getOrderParameterField();

  GridGPU phaseGradient = delta_grid->phaseGradient(h);
  phaseGradient *= 0.5;

  // Add gauge
  dim3 bpg, tpb;
  phaseGradient.getKernelDimensions(bpg, tpb);

  addGauge_Kernel<<<bpg, tpb>>>(phaseGradient.getDataPointer(0),
                                phaseGradient.getDataPointer(1), h, Phi, N);

  return phaseGradient;
}

#endif // COMPUTEPROPERTY_H
