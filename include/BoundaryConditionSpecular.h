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
  typedef typename Grid<T>::GPU GridGPU;
  typedef typename Grid<T>::CPU GridCPU;

public:
  BoundaryConditionSpecular() : BoundaryCondition<T>() {}
  virtual ~BoundaryConditionSpecular() {}

  virtual void computeBoundaryCondition(Context<T> *ctx);

private:
};

namespace internal {
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
    const T *const boundaryDistance, const int *const sparseIdxInCoherence,
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
        // Compute the reflected momentum in the local frame.
        T reflectedMomentumX;
        T reflectedMomentumY;
        computeReflectedMomentum(momentumX, momentumY, boundaryNormalsX[idx],
                                 boundaryNormalsY[idx], reflectedMomentumX,
                                 reflectedMomentumY);

        // Rotate the reflected momentum to the global frame.
        utils::rotate(systemAngle, reflectedMomentumX, reflectedMomentumY);

        // Get the fractional angle index.
        const T fractionalAngleIdx = FermiSurface<T>::getFractionalAngleIndex(
            reflectedMomentumX, reflectedMomentumY, momentumOffsetX,
            momentumOffsetY, numAngles);

        // Find corresponding indices of trajectories towards boundary
        // (must interpolate from two angles).
        // Wrap index to stay inside range.
        const int interpAngleIdx0 = utils::wrapToRange(
            static_cast<int>(fractionalAngleIdx), 0, numAngles);
        const int interpAngleIdx1 = utils::wrapToRange(
            static_cast<int>(fractionalAngleIdx) + 1, 0, numAngles);
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
        const T boundaryPointX = static_cast<T>(xIdx) - coordOffset;

        // Distance to exact boundary location.
        // TODO(niclas): For gamma we have boundaryLabel = 2, i.e. entering.
        // Which means that it is -h. For gammaTilde we have boundaryLabel = 3,
        // i.e. exiting. Thus +h. It should probably be done in a nicer way.
        const T h = abs(boundaryDistance[idx]);
        const T boundaryPointY = static_cast<T>(yIdx) -
                                 static_cast<T>(utils::sign(direction)) * h -
                                 coordOffset;

        const T rot_pt_angle[2] = {
            systemAngle - systemAngles[interpAngleIdx[0]],
            systemAngle - systemAngles[interpAngleIdx[1]]};

        // NEW ALGORITHM:
        // Make energy innermost loop. Need to make numEnergies-sized arrays for
        // intermediate interpolation values.
        // Sparse storage majority: Angle, energy, trajectory

        T val_bnd_interp_real = static_cast<T>(0.0);
        T val_bnd_interp_imag = static_cast<T>(0.0);
        T totalWeight = static_cast<T>(0.0);

        for (int rot = 0; rot < 2; ++rot) {
          // TODO(niclas): What is the "discrete frame"?
          // Rotate boundary point to discrete frame
          // Transform coordinate to index
          T boundary_pt_rot_x;
          T boundary_pt_rot_y;
          utils::rotate(rot_pt_angle[rot], boundaryPointX, boundaryPointY,
                        boundary_pt_rot_x, boundary_pt_rot_y);
          boundary_pt_rot_x += coordOffset;
          boundary_pt_rot_y += coordOffset;

          // Interpolation (points) indices
          const int idx_interp[2] = {static_cast<int>(boundary_pt_rot_x),
                                     static_cast<int>(boundary_pt_rot_x) + 1};

          // Interpolation (points) weights
          const T pt_idx_frac = boundary_pt_rot_x - floor(boundary_pt_rot_x);

          const T pt_weight[2] = {static_cast<T>(1.0) - pt_idx_frac,
                                  pt_idx_frac};

          for (int pt = 0; pt < 2; ++pt) {
            // Find the number of boundary intersections for current trajectory.
            const int num_isect =
                numBoundaryIntersections[interpAngleIdx[rot] * numX +
                                         idx_interp[pt]];

            int isect_current = 0;

            // Find which intersection corresponds to current point
            for (int isect = 0; isect < num_isect; ++isect) {
              const int sparse_idx =
                  (sparseIdxOutCoherence[interpAngleIdx[rot] * numX +
                                         idx_interp[pt]] +
                   isect);

              const T y_coord_diff =
                  abs(static_cast<T>(coordBoundaryIntersections[sparse_idx]) -
                      boundary_pt_rot_y);

              // TODO(niclas): What does this check? Why 2?
              if (y_coord_diff < static_cast<T>(2.0)) {
                isect_current = isect;
              }
            } // for isect

            // Get sparse index for intersection
            const int idx_sparse_out = BoundaryStorage<T>::getSparseIndex(
                interpAngleIdx[rot], eIdx, idx_interp[pt], isect_current,
                numEnergies, numX, sparseIdxOutCoherence);

            // TODO(niclas): What does it mean?
            const bool isIdxSparseOutOk = (idx_sparse_out > -1);

            // Maybe pre-load these [bnd_out_gamma...] to shared memory.
            const T val_bnd_real =
                isIdxSparseOutOk ? boundaryOutCoherenceReal[idx_sparse_out]
                                 : static_cast<T>(0.0);
            const T val_bnd_imag =
                isIdxSparseOutOk ? boundaryOutCoherenceImag[idx_sparse_out]
                                 : static_cast<T>(0.0);
            totalWeight += isIdxSparseOutOk
                               ? (pt_weight[pt] * angleWeights[rot])
                               : static_cast<T>(0.0);

            // Add interpolation contribution to final value
            val_bnd_interp_real +=
                val_bnd_real * pt_weight[pt] * angleWeights[rot];
            val_bnd_interp_imag +=
                val_bnd_imag * pt_weight[pt] * angleWeights[rot];
          } // for pt
        }   // for rot

        const int idx_sparse_in = BoundaryStorage<T>::getSparseIndex(
            angleIdx, eIdx, xIdx, numBoundaryEntry, numEnergies, numX,
            sparseIdxInCoherence);

        if (totalWeight > static_cast<T>(0.0)) {
          boundaryInCoherenceReal[idx_sparse_in] =
              val_bnd_interp_real / totalWeight;
          boundaryInCoherenceImag[idx_sparse_in] =
              val_bnd_interp_imag / totalWeight;
        } else {
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
    GridGPU::getKernelDimensions(bpg, tpb, numTrajectories, numEnergies);

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
