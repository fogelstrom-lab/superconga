//
// Collection of functions to set the initial condition of the order parameter.
// All member functions are static so this struct should not be instantiated.
//
#ifndef INITIAL_H
#define INITIAL_H

#include "OrderParameterComponent.h"
#include <cuda.h>
#include <math.h>

template <typename T> struct Initial {
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  // Computes bulk value of s-wave order parameter. Can be used as initial
  // condition.
  static double computeDeltaBulk(const double T_Tc,
                                 const double deltaConvergence = 1E-9);

  // The following functions set the deltaComponent field to...

  // ...delta bulk
  static void bulk(OrderParameterComponent<T> *deltaComponent,
                   const T mult_re = 1.0, const T mult_im = 0.0);

  // ...specified complex valued constant
  static void constant(OrderParameterComponent<T> *deltaComponent,
                       const T re = 1.0, const T im = 0.0);

  // ...the input grid argument
  static void grid(OrderParameterComponent<T> *deltaComponent,
                   const GridGPU &grid);

  // ...a given phase winding (one flux vortex has a phase winding of 2pi)
  static void phaseWinding(OrderParameterComponent<T> *deltaComponent,
                           const T numWinding, const T cutOffRadius = 0.0,
                           const T phaseShift = 0.0, const T center_x = 0.0,
                           const T center_y = 0.0);

  // ...delta bulk + a perturbation in the phase (to push solution out of
  // meta-stable initial state)
  static void random(OrderParameterComponent<T> *deltaComponent,
                     const T mix = 0.5, const T phase = 0.0);

  // Mixture between random() and phaseWinding():
  //     phaseWinding with an additional random phase factor
  static void perturbedPhaseWinding(OrderParameterComponent<T> *deltaComponent,
                                    const T numWinding, const T mix = 0.5,
                                    const T cutOffRadius = 0.0,
                                    const T phaseShift = 0.0,
                                    const T center_x = 0.0,
                                    const T center_y = 0.0);

  // ...the grid in the input file argument
  static int read(OrderParameterComponent<T> *deltaComponent,
                  const char *filename);
  static int read(OrderParameterComponent<T> *deltaComponent,
                  const char *filename, int filenum);
};

namespace {
template <typename T>
__global__ void
phaseWinding_radial_Kernel(T *data_re, T *data_im, const int Nx, const int Ny,
                           const T h, const T numWinding, const T cutOffRadius,
                           const T phaseShift, const T center_x,
                           const T center_y, T deltaBulk) {
  // Coordinates to new resampling grid
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < Nx && j < Ny) {
    const int idx = j * Nx + i;

    // Normalize coordinates to -0.5 <= x,y <= +0.5, and offset with center
    T x = ((T)i / (T)(Nx - 1) - (T)0.5) - center_x;
    T y = ((T)j / (T)(Ny - 1) - (T)0.5) - center_y;

    T phase;

    if (x >= (T)0.0 && y >= (T)0.0) {
      if (x == (T)0.0)
        phase = M_PI * (T)0.5;
      else
        phase = atan(y / x);
    } else if (x < (T)0.0 && y >= (T)0.0) {
      phase = M_PI + atan(y / x);
    } else if (x < (T)0.0 && y < (T)0.0) {
      phase = M_PI + atan(y / x);
    } else {
      phase = (T)2.0 * M_PI + atan(y / x);
    }

    phase += phaseShift;

    T radius = sqrt(x * x + y * y);

    if (radius < cutOffRadius)
      phase = (T)0.0;

    if (center_x == (T)0.0 && center_y == (T)0.0 && i == (Nx - 1) / 2 &&
        j == (Ny - 1) / 2)
      deltaBulk = (T)0.0;

    data_re[idx] = cos(phase * numWinding) * deltaBulk;
    data_im[idx] = sin(phase * numWinding) * deltaBulk;
  }
}
} // namespace

template <typename T>
double Initial<T>::computeDeltaBulk(const double T_Tc,
                                    const double deltaConvergence) {
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

template <typename T>
void Initial<T>::bulk(OrderParameterComponent<T> *deltaComponent,
                      const T mult_re, const T mult_im) {
  Parameters<T> *params = deltaComponent->getParameters();

  const double T_Tc = params->getTemperature();
  const double deltaBulk = computeDeltaBulk(T_Tc);

  GridGPU *deltaGrid = deltaComponent->getOrderParameterField();

  deltaGrid->setValue(deltaBulk * mult_re, Type::real);
  deltaGrid->setValue(deltaBulk * mult_im, Type::imag);
}

template <typename T>
void Initial<T>::constant(OrderParameterComponent<T> *deltaComponent,
                          const T re, const T im) {
  GridGPU *deltaGrid = deltaComponent->getOrderParameterField();

  deltaGrid->setValue(re, Type::real);
  deltaGrid->setValue(im, Type::imag);
}

template <typename T>
void Initial<T>::grid(OrderParameterComponent<T> *deltaComponent,
                      const GridGPU &grid) {
  GridGPU tmp = grid;

  GridGPU *delta = deltaComponent->getOrderParameterField();

  if (tmp.dimX() != delta->dimX())
    tmp.resample(delta->dimX(), delta->dimY());

  *delta = tmp;
}

template <typename T>
void Initial<T>::phaseWinding(OrderParameterComponent<T> *deltaComponent,
                              const T numWinding, const T cutOffRadius,
                              const T phaseShift, const T center_x,
                              const T center_y) {
  GridGPU *deltaGrid = deltaComponent->getOrderParameterField();

  const int N = deltaGrid->dimX();
  const T h = deltaComponent->getParameters()->getGridElementSize();
  const T deltaBulk = deltaComponent->deltaBulk();

  // GPU init
  dim3 blocksPerGrid, threadsPerBlock;
  deltaGrid->getKernelDimensions(blocksPerGrid, threadsPerBlock);

  if (numWinding > 0.0) {
    // Kernel launch
    for (int c = 0; c < deltaGrid->numFields(); ++c) {
      // Calc and set boundary grid
      phaseWinding_radial_Kernel<<<blocksPerGrid, threadsPerBlock>>>(
          deltaGrid->getDataPointer(c, Type::real),
          deltaGrid->getDataPointer(c, Type::imag), N, N, h, numWinding,
          cutOffRadius, phaseShift, center_x, center_y, deltaBulk);
    }
  } else {
    deltaGrid->setValue(deltaBulk, Type::real);
    deltaGrid->setValue(0.0, Type::imag);
  }
}

template <typename T>
void Initial<T>::random(OrderParameterComponent<T> *deltaComponent, const T mix,
                        const T phase) {
  Parameters<T> *params = deltaComponent->getParameters();
  GridGPU *deltaGrid = deltaComponent->getOrderParameterField();

  const int N = deltaGrid->dimX();
  const T h = params->getGridElementSize();
  const T deltaBulk = computeDeltaBulk(params->getTemperature());

  T *data_re, *data_im, rand_phase;
  int idx;

  // Kernel launch
  GridCPU *delta = 0;
  Grid<T>::CPU::initializeGrid(delta, deltaGrid);

  for (int c = 0; c < delta->numFields(); ++c) {
    data_re = delta->getDataPointer(c, Type::real);
    data_im = delta->getDataPointer(c, Type::imag);

    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        idx = j * N + i;

        rand_phase =
            ((double)rand() / (double)RAND_MAX - 0.5) * (T)2.0 * M_PI * mix;

        data_re[idx] = deltaBulk * cos(phase + rand_phase);
        data_im[idx] = deltaBulk * sin(phase + rand_phase);
      }
    }
  }
  *deltaGrid = *delta;
}

template <typename T>
void Initial<T>::perturbedPhaseWinding(
    OrderParameterComponent<T> *deltaComponent, const T numWinding, const T mix,
    const T cutOffRadius, const T phaseShift, const T center_x,
    const T center_y) {
  Parameters<T> *params = deltaComponent->getParameters();
  GridGPU *deltaGrid = deltaComponent->getOrderParameterField();

  const int N = deltaGrid->dimX();
  const T h = params->getGridElementSize();
  T deltaBulk = computeDeltaBulk(params->getTemperature());
  T amplitude = deltaBulk;

  T *data_re, *data_im, rand_phase;
  int idx;

  // Kernel launch
  GridCPU *delta = 0;
  Grid<T>::CPU::initializeGrid(delta, deltaGrid);

  for (int c = 0; c < delta->numFields(); ++c) {
    data_re = delta->getDataPointer(c, Type::real);
    data_im = delta->getDataPointer(c, Type::imag);

    for (int j = 0; j < N; ++j) {
      for (int i = 0; i < N; ++i) {
        idx = j * N + i;

        // Normalize coordinates to -0.5 <= x,y <= +0.5, and offset with center
        T x = ((T)i / (T)(N - 1) - (T)0.5) - center_x;
        T y = ((T)j / (T)(N - 1) - (T)0.5) - center_y;

        T phase;

        if (x >= (T)0.0 && y >= (T)0.0) {
          if (x == (T)0.0)
            phase = M_PI * (T)0.5;
          else
            phase = atan(y / x);
        } else if (x < (T)0.0 && y >= (T)0.0) {
          phase = M_PI + atan(y / x);
        } else if (x < (T)0.0 && y < (T)0.0) {
          phase = M_PI + atan(y / x);
        } else {
          phase = (T)2.0 * M_PI + atan(y / x);
        }

        phase += phaseShift;

        T radius = sqrt(x * x + y * y);

        if (radius < cutOffRadius)
          phase = (T)0.0;

        if (center_x == (T)0.0 && center_y == (T)0.0 && i == (N - 1) / 2 &&
            j == (N - 1) / 2)
          amplitude = (T)0.0;
        else
          amplitude = deltaBulk;

        rand_phase =
            ((double)rand() / (double)RAND_MAX - 0.5) * (T)2.0 * M_PI * mix;

        data_re[idx] = amplitude * cos(phase * numWinding + rand_phase);
        data_im[idx] = amplitude * sin(phase * numWinding + rand_phase);
      }
    }
  }
  *deltaGrid = *delta;
}

template <typename T>
int Initial<T>::read(OrderParameterComponent<T> *deltaComponent,
                     const char *filename) {
  GridCPU tmp;

  if (tmp.read(filename))
    return 1;

  GridGPU *delta = deltaComponent->getOrderParameterField();

  if (tmp.dimX() != delta->dimX())
    tmp.resample(delta->dimX(), delta->dimY());

  *delta = tmp;

  return 0;
}

template <typename T>
int Initial<T>::read(OrderParameterComponent<T> *deltaComponent,
                     const char *filename, int num) {
  char fn[512];
  sprintf(fn, "%s.%d.grd", filename, num);

  return read(deltaComponent, fn);
}

#endif // INITIAL_H
