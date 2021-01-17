#ifndef BOUNDARY_CONDITION_SPECULAR_H_
#define BOUNDARY_CONDITION_SPECULAR_H_

#include "BoundaryCondition.h"
#include "BoundaryStorage.h"
#include "FermiSurface.h"
#include "utils.h"
#include <cmath>
#include <cuda.h>

namespace scg {
template <typename T>
class BoundaryConditionSpecular : public BoundaryCondition<T> {
public:
  BoundaryConditionSpecular() : BoundaryCondition<T>() {}
  virtual ~BoundaryConditionSpecular() {}

  virtual void computeBoundaryCondition(Context<T> *ctx);

private:
};

namespace internal {
/// \brief Compute the reflected momentum at an edge.
///
/// \param inMomentumX The x-component of the incoming momentum.
/// \param inMomentumY The y-component of the incoming momentum.
/// \param inNormalX The x-component of the surface normal.
/// \param inNormalY The y-component of the surface normal.
/// \param outMomentumX The x-component of the reflected momentum.
/// \param outMomentumY The y-component of the reflected momentum.
///
/// \return Void.
// TODO(niclas): Identify if a corner is hit, and send out retroreflected
// momentum instead, to avoid branching and calling interpolate() twice. When
// there is separate domain label for corners, this can be done easily by
// passing the domain label to this function.
template <typename T>
__device__ void computeReflectedMomentum(const T inMomentumX,
                                         const T inMomentumY, const T inNormalX,
                                         const T inNormalY, T &outMomentumX,
                                         T &outMomentumY) {
  // Two times the dot product of the momentum and surface normal.
  const T twoDot =
      static_cast<T>(2.0) * (inMomentumX * inNormalX + inMomentumY * inNormalY);

  // The projection of the momentum on the normal is given by (p dot n)n.
  // By subtracting it twice from the incoming momentum we reverse the component
  // parallel to the normal (orthogonal to the surface), i.e. specular
  // reflection.
  outMomentumX = inMomentumX - twoDot * inNormalX;
  outMomentumY = inMomentumY - twoDot * inNormalY;
}

/// \brief Find the two closest angles, and two closest points, to a reflected
/// trajectory, and interpolate between them.
///
/// \param inIdxX The x-index.
/// \param inIdxY The y-index.
/// \param inIdxE The energy index.
/// \param inSystemAngle The rotation of the system.
/// \param inReflectedMomentumX The x-component of the reflected momentum.
/// \param inReflectedMomentumY The y-component of the reflected momentum.
/// \param inMomentumOffsetX The x-component of the momentum offset.
/// \param inMomentumOffsetY The y-component of the momentum offset.
/// \param inNumX The width of the grid.
/// \param inNumEnergies The total number of energies.
/// \param inNumAngles The total number of angles.
/// \param inCoordinateOffset The offset needed to move the origin to the
/// center.
/// \param inBoundaryDistance The exact distance to the boundary for the
/// current point.
/// \param inDirection Essentially whether we working with gamma
/// or gammaTilde.
/// \param inSystemAngles An array of all system angles.
/// \param inSparseIdxOutCoherence A magic array with indices for the
/// reflected coherence function.
/// \param inNumBoundaryIntersections A magic array of the number of boundary
/// intersections.
/// \param inCoordBoundaryIntersections A magic array containing the coordinates
/// of the boundary intersection, pehaps!?
/// \param inBoundaryOutCoherenceReal The real part of the reflected coherence
/// founction at the bounadry.
/// \param inBoundaryOutCoherenceImag The imaginary part of the reflected
/// coherence founction at the bounadry.
/// \param outInterpReal The real part of the interpolated coherence-function
/// value.
/// \param outInterpImag The imaginary part of the interpolated
/// coherence-function value.
/// \param outTotalWeight The total weight given to the different angles, and
/// points, being interpolated. If the function failed, this variable will be
/// zero.
///
/// \return Void.
template <typename T>
__device__ void interpolate(
    const int inIdxX, const int inIdxY, const int inIdxE, const T inSystemAngle,
    const T inReflectedMomentumX, const T inReflectedMomentumY,
    const T inMomentumOffsetX, const T inMomentumOffsetY, const int inNumX,
    const int inNumEnergies, const int inNumAngles, const T inCoordinateOffset,
    const T inBoundaryDistance, const int inDirection,
    const T *const inSystemAngles, const int *const inSparseIdxOutCoherence,
    const int *const inNumBoundaryIntersections,
    const int *const inCoordBoundaryIntersections,
    const T *const inBoundaryOutCoherenceReal,
    const T *const inBoundaryOutCoherenceImag, T &outInterpReal,
    T &outInterpImag, T &outTotalWeight) {

  // Rotate the reflected momentum to the global frame.
  T momentumX;
  T momentumY;
  utils::rotate(inSystemAngle, inReflectedMomentumX, inReflectedMomentumY,
                momentumX, momentumY);

  // Get the fractional angle index.
  const T fractionalAngleIdx = FermiSurface<T>::getFractionalAngleIndex(
      momentumX, momentumY, inMomentumOffsetX, inMomentumOffsetY, inNumAngles);

  // Find corresponding indices of trajectories towards boundary
  // (must interpolate from two angles).
  // Wrap index to stay inside range.
  const int interpAngleIdx0 =
      utils::wrapToRange(static_cast<int>(fractionalAngleIdx), 0, inNumAngles);
  const int interpAngleIdx1 = utils::wrapToRange(
      static_cast<int>(fractionalAngleIdx) + 1, 0, inNumAngles);
  const int interpAngleIdx[2] = {interpAngleIdx0, interpAngleIdx1};

  // Interpolate gammas from outgoing trajectory from stored
  // boundary values.

  // Get the remainder after subtracting the integer angle index.
  const T interpAngleIdxFraction =
      fractionalAngleIdx - floor(fractionalAngleIdx);

  // Weights for interpolation of discrete angles.
  const T angleWeights[2] = {static_cast<T>(1.0) - interpAngleIdxFraction,
                             interpAngleIdxFraction};

  // TODO(niclas): Is the comment below still relevant?
  // Only works for one entry/exit per trajectory

  // Boundary point to be rotated.
  const T boundaryPointX = static_cast<T>(inIdxX) - inCoordinateOffset;

  // Distance to exact boundary location.
  // TODO(niclas): For gamma we have boundaryLabel = 2, i.e. entering.
  // Which means that it is -h. For gammaTilde we have boundaryLabel = 3,
  // i.e. exiting. Thus +h. It should probably be done in a nicer way.
  const T boundaryPointY =
      static_cast<T>(inIdxY) -
      static_cast<T>(utils::sign(inDirection)) * inBoundaryDistance -
      inCoordinateOffset;

  const T rot_pt_angle[2] = {inSystemAngle - inSystemAngles[interpAngleIdx[0]],
                             inSystemAngle - inSystemAngles[interpAngleIdx[1]]};

  // NEW ALGORITHM:
  // Make energy innermost loop. Need to make numEnergies-sized arrays for
  // intermediate interpolation values.
  // Sparse storage majority: Angle, energy, trajectory

  outInterpReal = static_cast<T>(0.0);
  outInterpImag = static_cast<T>(0.0);
  outTotalWeight = static_cast<T>(0.0);

  for (int rot = 0; rot < 2; ++rot) {
    // TODO(niclas): What is the "discrete frame"?
    // Rotate boundary point to discrete frame
    // Transform coordinate to index
    T boundary_pt_rot_x;
    T boundary_pt_rot_y;
    utils::rotate(rot_pt_angle[rot], boundaryPointX, boundaryPointY,
                  boundary_pt_rot_x, boundary_pt_rot_y);
    boundary_pt_rot_x += inCoordinateOffset;
    boundary_pt_rot_y += inCoordinateOffset;

    // Interpolation (points) indices
    const int idx_interp[2] = {static_cast<int>(boundary_pt_rot_x),
                               static_cast<int>(boundary_pt_rot_x) + 1};

    // Interpolation (points) weights
    const T pt_idx_frac = boundary_pt_rot_x - floor(boundary_pt_rot_x);

    const T pt_weight[2] = {static_cast<T>(1.0) - pt_idx_frac, pt_idx_frac};

    for (int pt = 0; pt < 2; ++pt) {
      // Find the number of boundary intersections for current trajectory.
      const int num_isect =
          inNumBoundaryIntersections[interpAngleIdx[rot] * inNumX +
                                     idx_interp[pt]];

      int isect_current = 0;

      // Find which intersection corresponds to current point
      for (int isect = 0; isect < num_isect; ++isect) {
        const int sparse_idx =
            (inSparseIdxOutCoherence[interpAngleIdx[rot] * inNumX +
                                     idx_interp[pt]] +
             isect);

        const T y_coord_diff =
            abs(static_cast<T>(inCoordBoundaryIntersections[sparse_idx]) -
                boundary_pt_rot_y);

        // TODO(niclas): What does this check? Why 2?
        if (y_coord_diff < static_cast<T>(2.0)) {
          isect_current = isect;
        }
      } // for isect

      // Get sparse index for intersection
      const int idx_sparse_out = BoundaryStorage<T>::getSparseIndex(
          interpAngleIdx[rot], inIdxE, idx_interp[pt], isect_current,
          inNumEnergies, inNumX, inSparseIdxOutCoherence);

      const bool isIdxSparseOutOk = (idx_sparse_out > -1);

      if (isIdxSparseOutOk) {
        const T weight = pt_weight[pt] * angleWeights[rot];

        // Add interpolation contribution to final value.
        outInterpReal += inBoundaryOutCoherenceReal[idx_sparse_out] * weight;
        outInterpImag += inBoundaryOutCoherenceImag[idx_sparse_out] * weight;
        outTotalWeight += weight;
      }
    } // for pt
  }   // for rot
}

// TODO(niclas): Why is is the OUT pointers unchanged, and not IN? The
// nomenclature is a little bit odd, as they refer to direction to the bulk, not
// the scatterer.
template <typename T>
__global__ void boundaryConditionSpecularKernel(
    const int numX, const int numEnergies, const int numAngles,
    const int angleIdx, const int xIdxMin, const int xIdxMax, const int yIdxMin,
    const int yIdxMax, const T momentumX, const T momentumY,
    const T momentumOffsetX, const T momentumOffsetY,
    const T *const systemAngles, const int direction,
    const char *const domainLabel, const char boundaryLabel,
    const T *const boundaryNormalsX, const T *const boundaryNormalsY,
    const T *const boundaryDistances, const int *const sparseIdxInCoherence,
    const int *const sparseIdxOutCoherence,
    const int *const numBoundaryIntersections,
    const int *const coordBoundaryIntersections,
    T *const boundaryInCoherenceReal, T *const boundaryInCoherenceImag,
    const T *const boundaryOutCoherenceReal,
    const T *const boundaryOutCoherenceImag) {
  // x-coordinate index.
  const int xIdx = threadIdx.x + blockDim.x * blockIdx.x + xIdxMin;
  // Energy index.
  const int eIdx = threadIdx.y + blockDim.y * blockIdx.y;

  // Use variable 'offset' when transforming grid index to coordinates with
  // origin in center of grid matrix.
  const T coordOffset = static_cast<T>(numX - 1) * static_cast<T>(0.5);

  // Get the system angle.
  const T systemAngle = systemAngles[angleIdx];

  if (xIdx <= xIdxMax) {
    int numBoundaryEntry = 0;

    // The number of y-coordinates.
    const int numY = yIdxMax - yIdxMin + 1;

    for (int y = 0; y < numY; ++y) {
      // Depending on the coherence function, gamma (gamma tilde), we need to go
      // forward (backward) along the trajectory.
      const int yIdx = direction > 0 ? (yIdxMin + y) : (yIdxMax - y);

      // Index to 2d-grids
      const int idx = yIdx * numX + xIdx;

      // Get the node type.
      const char nodeType = domainLabel[idx];

      // At boundary, going in. Now we need to find where trajectory comes from
      if (nodeType == boundaryLabel) {
        // The exact distance from the center of the lattice site to the
        // boundary.
        const T boundaryDistance = abs(boundaryDistances[idx]);

        // Compute the reflected momentum in the local frame.
        T reflectedMomentumX;
        T reflectedMomentumY;
        computeReflectedMomentum(momentumX, momentumY, boundaryNormalsX[idx],
                                 boundaryNormalsY[idx], reflectedMomentumX,
                                 reflectedMomentumY);

        T interpReal;
        T interpImag;
        T totalWeight;
        interpolate(xIdx, yIdx, eIdx, systemAngle, reflectedMomentumX,
                    reflectedMomentumY, momentumOffsetX, momentumOffsetY, numX,
                    numEnergies, numAngles, coordOffset, boundaryDistance,
                    direction, systemAngles, sparseIdxOutCoherence,
                    numBoundaryIntersections, coordBoundaryIntersections,
                    boundaryOutCoherenceReal, boundaryOutCoherenceImag,
                    interpReal, interpImag, totalWeight);

        // Small value. If the total weight is less than this we consider the
        // interpolation a failure.
        const T epsilon = static_cast<T>(1e-7);

        // TODO(niclas): This is a horrible branching that will slow down this
        // kernel. We should label the corners with a separate domain label.
        if (totalWeight < epsilon) {
          // This happens when a trajectory hits a corner more or less dead on.

          // Try retroreflection.
          // TODO(niclas): For right-angle corners this is probably correct. For
          // other types of corners we need to do some more math.
          const T retroReflectedMomentumX = -momentumX;
          const T retroReflectedMomentumY = -momentumY;
          interpolate(
              xIdx, yIdx, eIdx, systemAngle, retroReflectedMomentumX,
              retroReflectedMomentumY, momentumOffsetX, momentumOffsetY, numX,
              numEnergies, numAngles, coordOffset, boundaryDistance, direction,
              systemAngles, sparseIdxOutCoherence, numBoundaryIntersections,
              coordBoundaryIntersections, boundaryOutCoherenceReal,
              boundaryOutCoherenceImag, interpReal, interpImag, totalWeight);
        }

        const int idx_sparse_in = BoundaryStorage<T>::getSparseIndex(
            angleIdx, eIdx, xIdx, numBoundaryEntry, numEnergies, numX,
            sparseIdxInCoherence);

        if (totalWeight > static_cast<T>(0.0)) {
          // Success!
          boundaryInCoherenceReal[idx_sparse_in] = interpReal / totalWeight;
          boundaryInCoherenceImag[idx_sparse_in] = interpImag / totalWeight;
        } else {
          // This should never happen.
          printf("WARNING! Specular boundary condition failed.");
          boundaryInCoherenceReal[idx_sparse_in] = static_cast<T>(0.0);
          boundaryInCoherenceImag[idx_sparse_in] = static_cast<T>(0.0);
        }

        numBoundaryEntry++;
      }
    }
  }
}
} // namespace internal

template <typename T>
void BoundaryConditionSpecular<T>::computeBoundaryCondition(Context<T> *ctx) {
  IntegrationIterator<T> *iterator = ctx->getIntegrationIterator();
  GeometryGroup<T> *geom = ctx->getGeometry();
  RiccatiSolver<T> *riccati = ctx->getRiccatiSolver();
  BoundaryStorage<T> *bnd = riccati->getBoundaryStorage();
  FermiSurface<T> *fermiSurface = ctx->getFermiSurface();

  const int numAngles = ctx->getParameters()->getAngularResolution();
  const int numEnergies = iterator->getNumEnergiesInterval();

  int numX, numY;
  geom->getGridDimensions(numX, numY);

  // Somewhat faster with streams
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  for (int angleIdx = 0; angleIdx < numAngles; ++angleIdx) {
    // Get the sample angle for this angle index.
    const T systemAngle = fermiSurface->getSystemAngle(angleIdx);

    // Get the Fermi momentum in the global frame.
    T momentumX;
    T momentumY;
    fermiSurface->getFermiMomentum(angleIdx, momentumX, momentumY);

    // Rotate the momentum to the local frame.
    utils::rotate(-systemAngle, momentumX, momentumY);

    // Set per angle optimized kernel dimensions
    const int xIdxMin =
        (bnd->getTrajectoryRangeX()->getDataPointer(0))[angleIdx];
    const int xIdxMax =
        (bnd->getTrajectoryRangeX()->getDataPointer(1))[angleIdx];
    const int yIdxMin =
        (bnd->getTrajectoryRangeY()->getDataPointer(0))[angleIdx];
    const int yIdxMax =
        (bnd->getTrajectoryRangeY()->getDataPointer(1))[angleIdx];

    const int numTrajectories = xIdxMax - xIdxMin + 1;

    dim3 bpg, tpb;
    GridGPU<T>::getKernelDimensions(bpg, tpb, numTrajectories, numEnergies);

    // Create geometry for current angle
    geom->create(-systemAngle);

    const char *const domainLabel =
        geom->getDomainLabelsGrid()->getDataPointer();
    const T *const boundaryNormalsX = geom->getNormalsGrid()->getDataPointer(0);
    const T *const boundaryNormalsY = geom->getNormalsGrid()->getDataPointer(1);
    const T *const boundaryDistance =
        geom->getBoundaryDistanceGrid()->getDataPointer();

    const T *const systemAnglesPtr = fermiSurface->getSystemAnglePointer();

    T momentumOffsetX;
    T momentumOffsetY;
    fermiSurface->getMomentumOffset(momentumOffsetX, momentumOffsetY);

    // Compute reflection
    // TODO(niclas): Don't check the numerical value, use enum!
    const char boundaryLabel = 2;
    const int direction = 1;
    internal::boundaryConditionSpecularKernel<T><<<bpg, tpb, 0, s1>>>(
        // Model parameters
        numX, numEnergies, numAngles, angleIdx, xIdxMin, xIdxMax, yIdxMin,
        yIdxMax, momentumX, momentumY, momentumOffsetX, momentumOffsetY,
        systemAnglesPtr, direction,
        // Grids for domain info/data
        domainLabel, boundaryLabel, boundaryNormalsX, boundaryNormalsY,
        boundaryDistance,
        // Data relating to sparse storage for gammas
        bnd->getIndexSparseIn_gamma()->getDataPointer(),
        bnd->getIndexSparseOut_gamma()->getDataPointer(),
        bnd->getNumBoundaryIntersect_gamma()->getDataPointer(),
        bnd->getBoundaryIntersectCoords_gamma()->getDataPointer(),
        bnd->getBoundaryIn_gamma()->getDataPointer(0, Type::real),
        bnd->getBoundaryIn_gamma()->getDataPointer(0, Type::imag),
        bnd->getBoundaryOut_gamma()->getDataPointer(0, Type::real),
        bnd->getBoundaryOut_gamma()->getDataPointer(0, Type::imag));

    // TODO(niclas): Don't check the numerical value, use enum!
    const char boundaryLabelTilde = 3;
    const int directionTilde = -1;
    internal::boundaryConditionSpecularKernel<T><<<bpg, tpb, 0, s2>>>(
        // Model parameters
        numX, numEnergies, numAngles, angleIdx, xIdxMin, xIdxMax, yIdxMin,
        yIdxMax, momentumX, momentumY, momentumOffsetX, momentumOffsetY,
        systemAnglesPtr, directionTilde,
        // Grids for domain info/data
        domainLabel, boundaryLabelTilde, boundaryNormalsX, boundaryNormalsY,
        boundaryDistance,
        // Data relating to sparse storage for gammas
        bnd->getIndexSparseIn_gammabar()->getDataPointer(),
        bnd->getIndexSparseOut_gammabar()->getDataPointer(),
        bnd->getNumBoundaryIntersect_gammabar()->getDataPointer(),
        bnd->getBoundaryIntersectCoords_gammabar()->getDataPointer(),
        bnd->getBoundaryIn_gammabar()->getDataPointer(0, Type::real),
        bnd->getBoundaryIn_gammabar()->getDataPointer(0, Type::imag),
        bnd->getBoundaryOut_gammabar()->getDataPointer(0, Type::real),
        bnd->getBoundaryOut_gammabar()->getDataPointer(0, Type::imag));
  }

  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
}
} // namespace scg

#endif // BOUNDARY_CONDITION_SPECULAR_H_
