#ifndef _RICCATISOLVERBULK_H
#define _RICCATISOLVERBULK_H

#include <RiccatiSolver.h>

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
class RiccatiSolverBulk : public RiccatiSolver<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  // Constructors and Destructors

  RiccatiSolverBulk();
  virtual ~RiccatiSolverBulk();

  // RiccatiSolverBulk( Parameters<T> & param, GeometryGroup<T> * geom );
  RiccatiSolverBulk(Context<T> *ctx);

  virtual int initialize();

  void initializeBoundaryStorage() {}

  // Functions
  virtual void
  computeInitialCondition(const ParameterIterator<int> &energy_iter,
                          const GridGPU *energies, const GridGPU *delta,
                          const int angle_idx) {}

  virtual void computeBoundaryCondition() {}

  virtual void
  computeCoherenceFunctions(const ParameterIterator<int> &energy_iter,
                            const GridGPU *energies, const GridGPU *delta,
                            const int angle_idx);

private:
};

namespace {
#include <cplx.h>

template <typename T, typename GAMMA, template <typename> class POTENTIAL,
          int ENTRY, int EXIT, int DIR>
__global__ void computeGammaFunctionBulk_kernel(
    const T h, const int N, const int Ne, const int idx_x_offset,
    const int angle_idx, const T system_angle, const T momentum_angle,
    const T B, T *energies_re, T *energies_im, T *gamma_real, T *gamma_imag,
    T *delta_real, T *delta_imag, char *domainLabel, T *domainNormals_x,
    T *domainNormals_y, T *boundaryDistance) {
  const int x =
      blockDim.x * blockIdx.x + threadIdx.x + idx_x_offset; // x-coordinate
  const int e_idx = blockDim.y * blockIdx.y + threadIdx.y;  // energy coordinate

  cplx<T> delta, g((T)0.0, (T)0.0), g0((T)0.0, (T)0.0), energy;
  T potential;

  int idx, gamma_idx, y;
  char nodeType;

  if (x < N && e_idx < Ne) {
    energy.x = energies_re[e_idx];
    energy.y = energies_im[e_idx];

    const T energy_zeroPotential = energy.x;

    const int yStart = (N - 1) * (int)(DIR == -1);

    // Compute gamma along trajectory
    for (int i = 0; i < N; ++i) {
      y = yStart + DIR * i;

      // Add energy contribution from vector potential
      potential = POTENTIAL<T>::computeVectorPotential_local(
          (T)x, (T)y, energy_zeroPotential, B, system_angle, momentum_angle, N,
          h);

      energy.x = energy_zeroPotential + potential;

      // Index to 2d-grids (Delta)
      idx = y * N + x;

      // Index to gamma grid (for all energies)
      gamma_idx = e_idx * 2 * N * N + y * N + x;

      // What type of node are we on?
      nodeType = domainLabel[idx];

      // Read order parameter
      delta.x = delta_real[idx];
      delta.y = delta_imag[idx];

      if (nodeType == 1) // Inside domain:
      {
        GAMMA::computeGamma_local(delta, g0, energy, h,
                                  i * (int)(nodeType != 1), g);
      }

      gamma_real[gamma_idx] = g.x;
      gamma_imag[gamma_idx] = g.y;

      g0 = g;
    }
  }
}
} // namespace

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
RiccatiSolverBulk<T, GAMMA, POTENTIAL>::RiccatiSolverBulk()
    : RiccatiSolver<T>() {}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
RiccatiSolverBulk<T, GAMMA, POTENTIAL>::~RiccatiSolverBulk() {
  std::cout << "~RiccatiSolverBulk()\n";
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
RiccatiSolverBulk<T, GAMMA, POTENTIAL>::RiccatiSolverBulk(Context<T> *ctx)
    : RiccatiSolver<T>(ctx) {
  std::cout << "RiccatiSolverBulk::RiccatiSolverBulk( Context<T>* )\n";
  ctx->set(this);
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
int RiccatiSolverBulk<T, GAMMA, POTENTIAL>::initialize() {
  std::cout << "RiccatiSolverBulk::initialize()\n";
  RiccatiSolver<T>::initialize();

  return 0;
}

template <typename T, template <typename> class GAMMA,
          template <typename> class POTENTIAL>
void RiccatiSolverBulk<T, GAMMA, POTENTIAL>::computeCoherenceFunctions(
    const ParameterIterator<int> &energy_iter, const GridGPU *energies,
    const GridGPU *deltaRot, const int current_angle_idx) {
  typedef RiccatiSolver<T> R;
  typedef ContextModule<T> C;

  // Get model parameters
  Parameters<T> *param = C::getParameters();

  // Determine some scalars
  const T h = param->getGridElementSize();
  const T B = param->getMagneticFieldStrength();
  const int N = param->getGridResolutionCoherence();
  const int numAngles = param->getAngularResolution();

  // Get pointers to grids
  GridGPU *gamma = R::getGamma();
  GridGPU *gammabar = R::getGammabar();

  GeometryGroup<T> *geom = C::getGeometry();

  dim3 blocksPerGrid, threadsPerBlock;

  int energyIdx_start = energy_iter.getCurrentValue();
  int numEnergiesPerBlock = energy_iter.getNumInterval();

  R::getKernelDimensions(blocksPerGrid, threadsPerBlock, numEnergiesPerBlock);
  int trajectoryIdx_start = 0;

  // System and gamma angle
  T system_angle = ((T)current_angle_idx / (T)numAngles) * (2.0 * PI);
  T gamma_angle = system_angle + 0.5 * PI;

  if (gamma_angle >= 2.0 * PI)
    gamma_angle -= 2.0 * PI;

  // Gives 10-15% speedup with streams
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  // 	Solve gamma by propagating riccati trajectories along positive y
  computeGammaFunctionBulk_kernel<T, typename GAMMA<T>::Gamma, POTENTIAL, 2, 3,
                                  1><<<blocksPerGrid, threadsPerBlock, 0, s1>>>(
      // Model parameters and energy values
      h, N, numEnergiesPerBlock, trajectoryIdx_start, current_angle_idx,
      system_angle, gamma_angle, B,
      energies->getDataPointer(0, Type::real) + energyIdx_start,
      energies->getDataPointer(0, Type::imag) + energyIdx_start,
      // Grids for gamma and Delta
      gamma->getDataPointer(0, Type::real),
      gamma->getDataPointer(0, Type::imag),
      deltaRot->getDataPointer(0, Type::real),
      deltaRot->getDataPointer(0, Type::imag),
      // Grids for domain info/data
      geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
      geom->getNormalsGrid()->getDataPointer(0, Type::real),
      geom->getNormalsGrid()->getDataPointer(1, Type::real),
      geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real));

  // 	Solve gammabar by propagating riccati trajectories along negative y
  computeGammaFunctionBulk_kernel<T, typename GAMMA<T>::Gammabar, POTENTIAL, 3,
                                  2, -1>
      <<<blocksPerGrid, threadsPerBlock, 0, s2>>>(
          // Model parameters and energy values
          h, N, numEnergiesPerBlock, trajectoryIdx_start, current_angle_idx,
          system_angle, gamma_angle, B,
          energies->getDataPointer(0, Type::real) + energyIdx_start,
          energies->getDataPointer(0, Type::imag) + energyIdx_start,
          // Grids for gammabar and Delta
          gammabar->getDataPointer(0, Type::real),
          gammabar->getDataPointer(0, Type::imag),
          deltaRot->getDataPointer(0, Type::real),
          deltaRot->getDataPointer(0, Type::imag),
          // Grids for domain info/data
          geom->getDomainLabelsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(0, Type::real),
          geom->getNormalsGrid()->getDataPointer(1, Type::real),
          geom->getBoundaryDistanceGrid()->getDataPointer(0, Type::real));

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}
#endif
