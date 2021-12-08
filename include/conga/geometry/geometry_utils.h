//===----- geometry_utils.h - Useful geometry helper functions. --===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Useful geometry helper functions.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_GEOMETRY_UTILS_H_
#define CONGA_GEOMETRY_UTILS_H_

#include "DiscGeometry.h"
#include "GeometryGroup.h"
#include "PolygonGeometry.h"

#include <json/json.h>

#include <limits>
#include <string>
#include <vector>

namespace conga {
namespace geometry_utils {
/// \brief Calculate scaling and offset for user-provided geometry.
///
/// Parse geometry dictionary for a list of geometry components, provided e.g.
/// via simulation parameters from user. Calculate the bounding box
/// of the compound geometry, and the offset to the geometry center point. The
/// bounding box is used to calculate the scaling factor to internal
/// coordinates. Return scaling and offsets.
///
/// \param inSimulationParameters User-defined simulation parameters.
/// \param outScalingFactor Scaling factor for geometry to internal coordinates.
/// \param outOffsetX Distance to geometry center point along x.
/// \param outOffsetY Distance to geometry center point along y.
///
/// \return void.
template <typename T>
void computeGeometryScaling(const Json::Value &inSimulationParameters,
                            T &outScalingFactor, T &outOffsetX, T &outOffsetY) {
  // Access geometry dictionary in simulation paramters.
  const Json::Value geometryComponentVector =
      inSimulationParameters["geometry"];

  // Find out bounding box for compound geometry.
  T xMax = std::numeric_limits<T>::min();
  T xMin = std::numeric_limits<T>::max();
  T yMax = std::numeric_limits<T>::min();
  T yMin = std::numeric_limits<T>::max();
  for (int i = 0; i < geometryComponentVector.size(); ++i) {
    // Fetch geometry component type (e.g. disc).
    const std::vector<std::string> geometryComponentNames =
        geometryComponentVector[i].getMemberNames();
    const std::string geometryComponentName = geometryComponentNames[0];

    // Only consider components that are added (not removed).
    if (!geometryComponentVector[i][geometryComponentName]["add"].asBool()) {
      continue;
    }

    // Update maxima and minima of geometry extent.
    if (geometryComponentName == "disc") {
      //***************************************************
      // DISC: GET EXTREMUM COORDINATES
      //***************************************************
      const T radius = static_cast<T>(
          0.5 * geometryComponentVector[i][geometryComponentName]["diameter"]
                    .asDouble());
      const T centerX = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_x"]
              .asDouble());
      const T centerY = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_y"]
              .asDouble());

      xMax = std::max(xMax, centerX + radius);
      xMin = std::min(xMin, centerX - radius);
      yMax = std::max(yMax, centerY + radius);
      yMin = std::min(yMin, centerY - radius);
    } else if (geometryComponentName == "polygon") {
      //***************************************************
      // POLYGON: GET EXTREMUM COORDINATES
      //***************************************************
      // Loop over vertex coordinates.
      const Json::Value xVerticesDict =
          inSimulationParameters["geometry"][i][geometryComponentName]
                                ["vertices_x"];
      const Json::Value yVerticesDict =
          inSimulationParameters["geometry"][i][geometryComponentName]
                                ["vertices_y"];

      // Check that the length is the same for the x and y vertices.
      if (xVerticesDict.size() != yVerticesDict.size()) {
        // Unknown geometry component. Print error and exit.
        std::cerr << "-- ERROR! Different number of x- and y-components given "
                     "for polygon vertices: "
                  << xVerticesDict.size() << " != " << yVerticesDict.size()
                  << "." << std::endl;
        std::exit(EXIT_FAILURE);
      }

      // Note, having n as std::size_t causes a bug in JsonCpp...
      for (int n = 0; n < xVerticesDict.size(); ++n) {
        const T x = static_cast<T>(xVerticesDict[n].asDouble());
        const T y = static_cast<T>(yVerticesDict[n].asDouble());

        xMax = std::max(xMax, x);
        xMin = std::min(xMin, x);
        yMax = std::max(yMax, y);
        yMin = std::min(yMin, y);
      }
    } else if (geometryComponentName == "regular_polygon") {
      //***************************************************
      // REGULAR POLYGON: GET EXTREMUM COORDINATES
      //***************************************************
      const T sideLength = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["side_length"]
              .asDouble());
      const T centerX = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_x"]
              .asDouble());
      const T centerY = static_cast<T>(
          geometryComponentVector[i][geometryComponentName]["center_y"]
              .asDouble());
      const T rotation = static_cast<T>(
          2.0 * M_PI *
          geometryComponentVector[i][geometryComponentName]["rotation"]
              .asDouble());
      const T numEdges =
          geometryComponentVector[i][geometryComponentName]["num_edges"]
              .asInt();

      const T angleStep = static_cast<T>(2.0 * M_PI) / static_cast<T>(numEdges);
      const T radius =
          sideLength / std::hypot(std::cos(angleStep) - static_cast<T>(1),
                                  std::sin(angleStep));
      const T angleOffset = static_cast<T>(0.5) * angleStep + rotation;

      for (int n = 0; n < numEdges; ++n) {
        const T angle = angleOffset + angleStep * static_cast<T>(n);
        const T x = centerX + radius * std::cos(angle);
        const T y = centerY + radius * std::sin(angle);

        xMax = std::max(xMax, x);
        xMin = std::min(xMin, x);
        yMax = std::max(yMax, y);
        yMin = std::min(yMin, y);
      }
    } else {
      // Unknown geometry component. Print error and exit.
      std::cout << "-- ERROR! Unknown geometry type: '" << geometryComponentName
                << "'." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // Calculate extent for scaling to internal coordinate system.
  const T xExtent = xMax - xMin;
  const T yExtent = yMax - yMin;
  // Calculate offset for translating to internal coordinate system.
  outOffsetX = xMin + static_cast<T>(0.5) * xExtent;
  outOffsetY = yMin + static_cast<T>(0.5) * yExtent;

  // Calculate scaling factor for scaling to internal coordinate system.
  // Consequently, this is the side length of the compound geometry - this is
  // returned so that it can be used for the Parameter class instance.
  outScalingFactor = std::max(xExtent, yExtent);
}

/// \brief Create geometry from list of geometry components.
///
/// Parse geometry dictionary for a list of geometry components, provided e.g.
/// via simulation parameters from user. Start by calculating the bounding box
/// of the compound geometry. Translate and scale the compound geometry to fit
/// in internal geometry coordinate system. Loop through list and add/remove
/// geometry objects as specified.
///
/// \param outGeometry Geometry container: instance of GeometryGroup class.
/// \param inSimulationParameters User-defined simulation parameters.
/// \param inScalingFactor Factor to scale geometry to internal coordinates.
/// \param inOffsetX Distance along x to translate geometry to origin.
/// \param inOffsetY Distance along y to translate geometry to origin.
/// \param inGrainFraction Simulation space size, minus margin.
///
/// \return void.
template <typename T>
void addGeometryComponents(GeometryGroup<T> *outGeometry,
                           const Json::Value &inSimulationParameters,
                           const T inScalingFactor, const T inOffsetX,
                           const T inOffsetY, const T inGrainFraction) {
  // Access geometry dictionary in simulation paramters.
  const Json::Value geometryComponentVector =
      inSimulationParameters["geometry"];

  // Loop over list of geometry components.
  for (int i = 0; i < geometryComponentVector.size(); ++i) {
    // Fetch geometry component type (e.g. disc).
    const std::vector<std::string> geometryComponentNames =
        geometryComponentVector[i].getMemberNames();
    const std::string geometryComponentName = geometryComponentNames[0];

    // Check whether component is to be added or removed.
    const Type::type addOrRemove =
        (geometryComponentVector[i][geometryComponentName]["add"].asBool())
            ? Type::add
            : Type::remove;

    // Add/remove the appropriate geometry component.
    if (geometryComponentName == "disc") {
      //***************************************************
      // DISC: ADD/REMOVE
      //***************************************************

      // Translate and scale center coordinates.
      const T xCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["disc"]["center_x"].asDouble()) -
           inOffsetX) /
          inScalingFactor;
      const T yCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["disc"]["center_y"].asDouble()) -
           inOffsetY) /
          inScalingFactor;
      // Disc diameter in internal coordinate system.
      const T diameter =
          static_cast<T>(
              geometryComponentVector[i]["disc"]["diameter"].asDouble()) /
          inScalingFactor;

      // Add/remove disc.
      outGeometry->add(new DiscGeometry<T>(inGrainFraction, diameter,
                                           xCenterTranslatedScaled,
                                           yCenterTranslatedScaled),
                       addOrRemove);
    } else if (geometryComponentVector[i].isMember("polygon")) {
      //***************************************************
      // POLYGON: ADD/REMOVE
      //***************************************************
      // Check whether component is to be added or removed.

      // Fetch vertices.
      const int numVertices =
          int(geometryComponentVector[i]["polygon"]["vertices_x"].size());

      T xVertices[numVertices];
      T yVertices[numVertices];

      for (int vertex_idx = 0; vertex_idx < numVertices; ++vertex_idx) {
        // Translate and scale polygon vertex coordinates.
        xVertices[vertex_idx] =
            (static_cast<T>(
                 geometryComponentVector[i]["polygon"]["vertices_x"][vertex_idx]
                     .asDouble()) -
             inOffsetX) /
            inScalingFactor;
        yVertices[vertex_idx] =
            (static_cast<T>(
                 geometryComponentVector[i]["polygon"]["vertices_y"][vertex_idx]
                     .asDouble()) -
             inOffsetY) /
            inScalingFactor;
      }

      // Add/remove polygon.
      outGeometry->add(new PolygonGeometry<T>(inGrainFraction, numVertices,
                                              xVertices, yVertices),
                       addOrRemove);
    } else if (geometryComponentVector[i].isMember("regular_polygon")) {
      //***************************************************
      // REGULAR POLYGON: ADD/REMOVE
      //***************************************************
      // Check whether component is to be added or removed.

      // Translate and scale center coordinates.
      const T xCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["regular_polygon"]["center_x"]
                   .asDouble()) -
           inOffsetX) /
          inScalingFactor;
      const T yCenterTranslatedScaled =
          (static_cast<T>(
               geometryComponentVector[i]["regular_polygon"]["center_y"]
                   .asDouble()) -
           inOffsetY) /
          inScalingFactor;
      // Side length of the regular polygon, in internal coordinate system.
      const T sideLength =
          static_cast<T>(
              geometryComponentVector[i]["regular_polygon"]["side_length"]
                  .asDouble()) /
          inScalingFactor;

      const int numEdges =
          geometryComponentVector[i]["regular_polygon"]["num_edges"].asInt();

      const T rotation = static_cast<T>(
          2.0 * M_PI *
          geometryComponentVector[i]["regular_polygon"]["rotation"].asDouble());

      // Add/remove regular polygon.
      outGeometry->add(new PolygonGeometry<T>(
                           inGrainFraction, numEdges, sideLength, rotation,
                           xCenterTranslatedScaled, yCenterTranslatedScaled),
                       addOrRemove);
    } else {
      // Unknown geometry component. Print error and exit.
      const std::vector<std::string> geometryComponentNames =
          geometryComponentVector[i].getMemberNames();
      std::cout << "-- ERROR! Unknown geometry type: '"
                << geometryComponentNames[i][0] << "'." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}
} // namespace geometry_utils
} // namespace conga

#endif // CONGA_GEOMETRY_UTILS_H_
