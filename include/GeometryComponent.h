//
// Base class for the discrete representation of geometry, as used by the Riccati solver.
//
#ifndef _GEOMETRYCOMPONENT_H
#define _GEOMETRYCOMPONENT_H

#include <iostream>
#include <cuda.h>
#include <Type.h>
#include <Grid.h>
#include <Parameters.h>

template <typename T>
class GeometryComponent
{
	typedef typename Grid<T>::GPU GridGPU;

public:
							GeometryComponent();
	virtual					~GeometryComponent();

							GeometryComponent( const GeometryComponent<T>& other );

	virtual GeometryComponent* clone() const = 0;

	Parameters<T>*			getParameters() const { return _param; }
	void					setParameters( Parameters<T>* param ) { _param = param; }
	
	// Convenience function to get discrete dimensions of geomtetry
	void					getGridDimensions( int& Nx, int& Ny ) const;

	// Convenience function to set cuda grid dimensions
	void 					getKernelDimensions( dim3& bpg, dim3& tpb ) const;

	// Getters (pointers to data in corresponding grids)
	char*	 				getDomainLabels() const;
	T*			 			getNormals( const int dim ) const;
	T*		 				getBoundaryDistance() const;

	// Getters for pointers to grids containing discrete geometry representation
	Grid<char>::GPU* 		getDomainLabelsGrid() const { return _domainLabels; }
	GridGPU*	 			getNormalsGrid() const { return _normals; }
	GridGPU*		 		getBoundaryDistanceGrid() const { return _boundaryDistance; }

	// Create discrete representation for geometry component in rotational frame given by angle argument
	// Stores result in _domainLabels, _normals, and _boundaryDistance
	virtual void				create( const T angle ) = 0;

	// Create discrete representation with resolution NxN, and return domain labels only.
	// Does not store data in any member variables.
	virtual Grid<char>::GPU*	createDomain( const int N ) = 0;

	// Label domain nodes which are close to boundary.
	void					labelBoundaryNodes();	

	// Convenience functions to allocate and free memory for _domainLabels, _normals, and _boundaryDistance
	void					allocateMem();
	void					freeMem(); 				// Gets called by GeometryGroup.

private:
	template <typename S>
	friend void		swap( GeometryComponent<S>& A, GeometryComponent<S>& B );

	Parameters<T>*		_param;
	
	// Below are the grids defining the augmented discrete representation of the geometry.
	// These will hold the geometry in the local frame, and are written to whenever create()
	// is called. Thus, they can never be assumed to hold the geometry in reference frame (theta=0).

	// Contains labels for discrete geometry. Label codes:
	// 0=outside, 1=inside, 2=inside, near boundary entry, 3=inside, near boundary exit
	Grid<char>::GPU*	_domainLabels;

	// Field with boundary normals for discrete geometry (2 fields (x- and y-component)).
	// Only stored where nodes are labeled 2 or 3.
	GridGPU*			_normals;

	// Field with distances between the boundary nodes to the exact location of the geometry,
	// as measured along the positive gamma trajectory. Only stored where nodes are labeled 2 or 3.
	GridGPU*			_boundaryDistance;
};

#endif

