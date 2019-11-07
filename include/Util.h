//
// Useful functions not belonging to any particular part of the framework of the
// self-consistent iterations.
//
#ifndef _UTIL_H
#define _UTIL_H

#include <math.h>
#include <Grid.h>

template <typename T>
struct Util
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:

	static void	mirror_diag_lower_to_upper( typename Grid<T>::GPU* grid );

};

#endif

