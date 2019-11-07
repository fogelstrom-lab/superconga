#ifndef _BOUNDARYSTORAGE_H
#define _BOUNDARYSTORAGE_H

template <typename T> class GeometryGroup;
template <typename T> class BoundaryStorage;
template <typename T> class IntegrationIterator;
template <typename T> class ComputeProperty;

#include <iostream>
#include <Type.h>
#include <Grid.h>
#include <GeometryGroup.h>
#include <ComputeProperty.h>
#include <IntegrationIterator.h>

template <typename T>
class BoundaryStorage
{
	typedef typename Grid<T>::GPU GridGPU;
	typedef typename Grid<T>::CPU GridCPU;

public:
					BoundaryStorage();
	virtual			~BoundaryStorage();

					BoundaryStorage( GeometryGroup<T> * geom );
					

	void	initialize( const int numTrajectoriesPerAngle, const int numEnergies, const int numAngles );
	void	setBoundaryZero();
	
	__device__ static int getSparseIndex(
		const int idx_a, const int idx_e, const int idx_x, const int boundaryCount,
		const int Ne, const int Nx, int *sparseIndices );


	// Storage getters
	Grid<int>::GPU	*	getIndexSparseIn_gamma() const { return _index_sparse_in_gamma; }
	Grid<int>::GPU	*	getIndexSparseOut_gamma() const { return _index_sparse_out_gamma; }
	Grid<int>::GPU	*	getIndexSparseIn_gammabar() const { return _index_sparse_in_gammabar; }
	Grid<int>::GPU	*	getIndexSparseOut_gammabar() const { return _index_sparse_out_gammabar; }

	Grid<int>::GPU	*	getBoundaryIntersectCoords_gamma() const { return _coord_bnd_intersect_gamma; }
	Grid<int>::GPU	*	getBoundaryIntersectCoords_gammabar() const { return _coord_bnd_intersect_gammabar; }

	Grid<int>::GPU	*	getNumBoundaryIntersect_gamma() const { return _num_bnd_intersect_gamma; }
	Grid<int>::GPU	*	getNumBoundaryIntersect_gammabar() const { return _num_bnd_intersect_gammabar; }

	GridGPU		*		getBoundaryIn_gamma() const { return _boundary_in_gamma; }
	GridGPU		*		getBoundaryOut_gamma() const { return _boundary_out_gamma; }
	GridGPU		*		getBoundaryIn_gammabar() const { return _boundary_in_gammabar; }
	GridGPU		*		getBoundaryOut_gammabar() const { return _boundary_out_gammabar; }

	// Helper array getters (see description below)
	Grid<int>::CPU	*	getTrajectoryRangeX() const { return _trajectoryRange_x; }
	Grid<int>::CPU	*	getTrajectoryRangeY() const { return _trajectoryRange_y; }

	int					getNumIntersectingTrajectoriesAngle( const int angle_idx ) const;
	int					getTrajectoryStartIndexAngle( const int angle_idx ) const;
	
	
private:
	void	allocMemSparseStorage(	const int numElemGammaIn, const int numElemGammaOut,
									const int numElemGammaBarIn, const int numElemGammaBarOut,
								    const int numEnergies );

	void	computeSparseIndices( const int numEnergies, const int numAngles );
	void	computeIntersectingTrajectories( const int numAngles );


	GeometryGroup<T> * _geom;

	int		_numSparseIn_gamma;
	int		_numSparseOut_gamma;
	int		_numSparseIn_gammabar;
	int		_numSparseOut_gammabar;
	
	int		_numEnergies;
	int 	_numAngles;
	
	// These arrays are for finding indices into sparse storage for boundary gammas.
	// Arrays are of dimension (Nx,numAngles) (trajectory along y). So for each
	// column (trajectory), for each angle, an index offset is stored. This index offset
	// corresponds to the address to get the first in/out boundary gammas for
	// a particular trajectory. If there are multiple entry/exit points on one
	// trajectory, this has to be kept track of.
	//
	
	// MAYBE MORE HIGH LEVEL FUNCTIONS FOR API USE.
	
	// Indices into sparse storage arrays
	//
	Grid<int>::GPU	*	_index_sparse_in_gamma;
	Grid<int>::GPU	*	_index_sparse_out_gamma;

	Grid<int>::GPU	*	_index_sparse_in_gammabar;
	Grid<int>::GPU	*	_index_sparse_out_gammabar;
	
	// Arrays with information about on how to find the trajectories which actually intersects
	// the computational domain (geometry). This information is used when constructing threadblocks
	// to avoid unneccesary thread launching.
	
	// Start and end x- and y-indices for trajectory integration
	Grid<int>::CPU	*	_trajectoryRange_x;
	Grid<int>::CPU	*	_trajectoryRange_y;

	// Per trajectory start and end y-indices for trajectory integration
	Grid<int>::GPU  *	_trajectoryRangeAllStart_y;
	Grid<int>::GPU  *	_trajectoryRangeAllEnd_y;

	// Local (to frame of current angle) y-coordinates where trajectory intersects boundary.
	// Used for geometries where one trajectory can have multiple boundary intersections.
	Grid<int>::GPU	*	_coord_bnd_intersect_gamma;
	Grid<int>::GPU	*	_coord_bnd_intersect_gammabar;

	Grid<int>::GPU	*	_num_bnd_intersect_gamma;
	Grid<int>::GPU	*	_num_bnd_intersect_gammabar;

	// Storage for domain boundary gammas. Use _index_sparse_* to get values.
	GridGPU	*	_boundary_in_gamma;
	GridGPU	*	_boundary_out_gamma;

	GridGPU	*	_boundary_in_gammabar;
	GridGPU	*	_boundary_out_gammabar;
			
};

#endif

