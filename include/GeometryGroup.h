//
// Container which stores and manages objects of type GeometryComponent.
// This class compounds the different components together, and produces the final
// discrete geometry output.
//
#ifndef _GEOMETRYGROUP_H
#define _GEOMETRYGROUP_H

#include <cuda.h>
#include <Type.h>
#include <cplx.h>
#include <Parameters.h>
#include <Grid.h>
#include <GeometryComponent.h>

template <typename T>
class GeometryGroup : public ContextModule<T>
{
	typedef typename Grid<T>::GPU GridGPU;

public:
							GeometryGroup();
	virtual					~GeometryGroup();

							GeometryGroup( const GeometryGroup<T>& other );
							GeometryGroup( Context<T>* ctx );

	GeometryGroup<T>&		operator=( GeometryGroup<T> other );

public:
	// Allocates memory for member variables (grids)
	virtual int 			initialize();

	// Add/set geometry component to contribute to resulting compounded geometry
	int						add( GeometryComponent<T>* newComponent, Type::type operation );
	void					set( GeometryComponent<T>* newComponent, Type::type operation, int geomIdx );

	// Get a pointer to the specified individual geometry component
	GeometryComponent<T>*	getGeometryComponent( const int comp ) const;

	int 					getNumGeometryComponents() const { return _geomComponentList.size(); }

	// Create discrete representation for compounded geometry in rotational frame given by 
	// angle argument. Stores result in _domainLabels, _normals, and _boundaryDistance
	void					create( const T angle );

	// Create discrete representation with resolution NxN, and return domain labels only.
	// Does not store data in any member variables.
	Grid<char>::GPU* 		createDomain( const int N );
	
	// Extrapolates values outwards from boundary, with numExtend lattice sites.
	// This is used prior to rotating e.g. the order parameter lattice into a rotational frame
	// in the momentum integral. The reason is that the rotation together with the linear
	// interpolation will cause the boundary lattice sites to (wrongly) have fractional values,
	// i.e. less than they should be. By extrapolating the boundary values, this artifact is eliminated.
	void					extendValuesAtBoundary( GridGPU* target, const int numExtend ) const;

	// Because of the extrapolation in extendValuesAtBoundary(), the computed quantities will
	// have some non-zero values slightly outside of the actual simulation domain. These do not belong
	// to the solution, and can be removed with the zeroValuesOutsideDomain() function.
	void					zeroValuesOutsideDomain( GridGPU* target );
	
	// Computes the area of the discrete geometry. Returned in physical units by default.
	// Set discrete=1 to retrieve the area in number of lattice nodes.
	T						area( int discrete = 0 );

	// Computes the average of the specified field in the inpute grid, only using the values
	// inside the actual geometry definition.
	cplx<T>					fieldAverage( GridGPU* data, const int field = 0 );

	// Getters	
	void					getGridDimensions( int & Nx, int & Ny ) const;
	int						getNumAngles() const;
	T						getCurrentAngle() const { return _currentAngle; }

	// Getters (pointers to data in corresponding grids)
	char*		 			getDomainLabels() const;
	T*			 			getNormals( const int dim ) const;
	T*			 			getBoundaryDistance() const;

	// Getters for pointers to grids containing discrete geometry representation
	Grid<char>::GPU* 		getDomainLabelsGrid() const { return _domainLabels; }
	GridGPU*	 			getNormalsGrid() const { return _normals; }
	GridGPU*		 		getBoundaryDistanceGrid() const { return _boundaryDistance; }

	// Get domain labels grid for the geometry in reference frame.
	Grid<char>::GPU*		getDomainGrid() const { return _domain_ref; }

	// Helper functions to find locations of geometry boundary crossings. Only used by the BoundaryStorage
	// class so far, when determining how much memory is required to store all the boundary values
	// for the coherence functions.
	void					countBoundariesAlongTrajectory( int Nx, int Ny,	char* dom_ptr,
								int* idx_in_ptr, int* idx_out_ptr, int* numBndIsect_ptr, int& sum_idx_in, int& sum_idx_out );
	void					countBoundariesAlongNegativeTrajectory( int Nx, int Ny,	char* dom_ptr,
								int* idx_in_ptr, int* idx_out_ptr, int* numBndIsect_ptr, int& sum_idx_in, int& sum_idx_out );
	
	void					findBoundaryCrossingsAlongTrajectory( int Nx, int Ny, char* dom_p, int*& coord_ptr );
	void					findBoundaryCrossingsAlongNegativeTrajectory( int Nx, int Ny, char* dom_p, int*& coord_ptr );


private:
	template <typename S>
	friend void 			swap( GeometryGroup<S>& A, GeometryGroup<S>& B );

	void					setGeomOpMatrix();

	void					geomOpAdd( GeometryComponent<T>* newGeom );
	void					geomOpRemove( GeometryComponent<T>* newGeom );
	void					geomOp( GeometryComponent<T>* newGeom, Grid<char>::GPU* geomOpMatrix );

	void					geomOpAdd( Grid<char>::GPU* dom,  Grid<char>::GPU* dom_local ) const;
	void					geomOpRemove( Grid<char>::GPU* dom,  Grid<char>::GPU* dom_local ) const;

	std::vector<GeometryComponent<T>*> _geomComponentList;

	std::vector<int>		_geomOpList;
	Grid<char>::GPU*		_geomOpAdd;
	Grid<char>::GPU*		_geomOpRemove;
	
	T						_currentAngle;
	
	// Below are the grids defining the augmented discrete representation of the geometry.
	// These will hold the geometry in the local frame, and are written to whenever create()
	// is called. Thus, they can never be assumed to hold the geometry in reference frame (theta=0).

	// Specifically, the grids store domain node labels (inside/outside/entry/exit), 
	// and normals at (exact) boundary location. The _boundaryDistance grid stores
	// the distance to the exact boundary location (along the trajectory),
	// for the corresponding grid point.
	// So the normals stored in grid at integer grid point (x,y) (generally not
	// coinciding with domain boundary), are actually located in real space at:
	//
	// ( x, y + abs(_boundaryDistance(x,y)) )  (when exiting)
	// ( x, y - abs(_boundaryDistance(x,y)) )  (when entering)
	//
	// which are the exact boundary domain locations.
	
	// Contains labels for discrete geometry. Label codes:
	// 0=outside, 1=inside, 2=inside, near boundary entry, 3=inside, near boundary exit
	Grid<char>::GPU*	_domainLabels;

	// Field with boundary normals for discrete geometry (2 fields (x- and y-component)).
	// Only stored where nodes are labeled 2 or 3.
	GridGPU*			_normals;

	// Field with distances between the boundary nodes to the exact location of the geometry,
	// as measured along the positive gamma trajectory. Only stored where nodes are labeled 2 or 3.
	GridGPU*			_boundaryDistance;


	// Geometry domain labels grid in reference/unrotated frame (theta=0)
	Grid<char>::GPU*	_domain_ref;
};

#endif

