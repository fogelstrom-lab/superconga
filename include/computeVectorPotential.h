#ifndef _COMPUTEVECTORPOTENTIAL_
#define _COMPUTEVECTORPOTENTIAL_

#include <cuda.h>
#include <Grid.h>
#include <GeometryGroup.h>

template <typename T>
void computeVectorPotential(
	GridData<T,thrust::device_vector<T> >* A,
	const GridData<T,thrust::device_vector<T> >* j,
	GridData<char,thrust::device_vector<char> >* geom,
	const T h, const T c, const int N );

#endif
