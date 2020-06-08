#ifndef GRID_H_
#define GRID_H_

#ifdef __CUDACC__
#define _DEV_ __host__ __device__
#else
#define _DEV_
#endif

#include "GridData.h"
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace scg {
// Wrapper structs/typedefs for the GridData class. This makes it simpler to
// read and write GridData variable declarations. For example:
// GridData<double,thrust::host_vector<double> >    --->  Grid<double>::CPU
// GridData<double,thrust::device_vector<double> >  --->  Grid<double>::GPU
// thrust::host_vector<double>    --->  Vector<double>::CPU
// thrust::device_vector<double>  --->  Vector<double>::GPU
template <typename T> struct Vector {
  typedef thrust::host_vector<T> CPU;

#ifdef __CUDACC__
  typedef thrust::device_vector<T> GPU;
#else
  typedef thrust::host_vector<T> GPU;
#endif
};

template <typename T> struct Grid {
  typedef GridData<T, thrust::host_vector<T>> CPU;

#ifdef __CUDACC__
  typedef GridData<T, thrust::device_vector<T>> GPU;
#else
  typedef GridData<T, thrust::host_vector<T>> GPU;
#endif
};
} // namespace scg

#endif // GRID_H_
