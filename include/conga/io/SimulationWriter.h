// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
#ifndef CONGA_SIMULATION_WRITER_H_
#define CONGA_SIMULATION_WRITER_H_

#include "../Context.h"
#include "../grid.h"
#include "DataFormat.h"
#include "csv.h"
#include "json.h"
#ifdef CONGA_USE_HDF5
#include "h5.h"
#endif // CONGA_USE_HDF5
#include "print.h"

#include <filesystem>

namespace conga {
namespace io {
/// \brief Class for writing simulation data and parameters to file.
template <typename T> class SimulationWriter {
public:
  /// \brief Constructor.
  ///
  /// \param inDataPath Path to data directory.
  /// \param inDataFormat Data format of large data files.
  /// \param inVerbose Whether to print to terminal or not.
  SimulationWriter(const std::filesystem::path &inDataPath,
                   const DataFormat inDataFormat, const bool inVerbose)
      : m_dataPath(inDataPath), m_dataFormat(inDataFormat),
        m_verbose(inVerbose) {}

  /// \brief Write data and paramters.
  ///
  /// \param inContext The context object.
  ///
  /// \return Void.
  void write(const Context<T> &inContext) {
    writeParameters(inContext);
    writeData(inContext);
  }

private:
  /// \brief Save config params, internal params, and results to JSON files.
  ///
  /// This function saves two json files: one for internal parameters, and one
  /// for scalar results. Motivation, there are some internal parameters that
  /// are not possible to infer without running the program (the exact number of
  /// lattice points inside the grain), these are very useful to have access to
  /// when post-processing (e.g. for doing area normalization). Finally, there
  /// are some scalar results (e.g. area-averaged quantities) that are more
  /// appropriate and convenient to have in json rather than csv.
  ///
  /// \param inContext The context object.
  ///
  /// \return Void.
  void writeParameters(const Context<T> &inContext) {
    // Number of lattice sites in grain.
    const int numGrainLatticeSites =
        inContext.getGeometry()->getNumGrainLatticeSites();

    // Side length of lattice site.
    const T latticeElementSize =
        inContext.getParameters()->getGridElementSize();

    // Json for various results.
    Json::Value resultsJSON;

    // Free energy per unit area.
    resultsJSON["free_energy_per_unit_area"] =
        inContext.getComputeFreeEnergy()->getFreeEnergyPerUnitArea();
    resultsJSON["magnetic_free_energy_per_unit_area"] =
        inContext.getComputeFreeEnergy()->getMagneticFreeEnergyPerUnitArea();

    // Average order-parameter magnitude.
    // TODO(niclas): Is this correct for multi-component OP?
    resultsJSON["order_parameter_average_magnitude"] =
        inContext.getComputeOrderParameter()->getResult()->sumMagnitude() /
        static_cast<T>(numGrainLatticeSites);

    // Average current-density magnitude.
    resultsJSON["current_density_average_magnitude"] =
        inContext.getComputeCurrent()->getResult()->sumMagnitude() /
        static_cast<T>(numGrainLatticeSites);

    // Magnetic moment per unit area.
    resultsJSON["magnetic_moment_per_unit_area"] =
        inContext.getComputeCurrent()->getResult()->moment(latticeElementSize) /
        inContext.getGeometry()->area();

    // Get the vector potential. It is used both for the average vector
    // potential, and average flux density.
    GridGPU<T> vectorPotential = *(inContext.getInducedVectorPotential());

    // Average flux density.
    // TODO(niclas): Should this be computed outside of the grain?
    GridGPU<T> fluxDensity = vectorPotential.curl(latticeElementSize);
    inContext.getGeometry()->zeroValuesOutsideDomainRef(&fluxDensity);
    resultsJSON["flux_density_average"] =
        fluxDensity.sumField() / static_cast<T>(numGrainLatticeSites);

    // Average vector-potential magnitude.
    // TODO(niclas): Should this be computed outside of the grain?
    inContext.getGeometry()->zeroValuesOutsideDomainRef(&vectorPotential);
    resultsJSON["vector_potential_average_magnitude"] =
        vectorPotential.sumMagnitude() / static_cast<T>(numGrainLatticeSites);

    // Save various results to JSON file.
    const std::filesystem::path resultsPath =
        m_dataPath / "simulation_results.json";
    io::write(resultsPath, resultsJSON);
    printWrite(m_verbose, true, "scalar results", resultsPath);

    // Internal parameters useful for post-processing.
    Json::Value internalParamJSON;
    internalParamJSON["area_compute"] = inContext.getGeometry()->area();
    internalParamJSON["internal_lattice_points"] = numGrainLatticeSites;
    internalParamJSON["grain_margin"] = constants::GRAIN_SIZE_MARGIN;
    const int numLatticeSitesPerSide =
        inContext.getParameters()->getGridResolutionBase();
    internalParamJSON["num_lattice_points"] =
        numLatticeSitesPerSide * numLatticeSitesPerSide;

    // Save internal parameters to JSON file.
    const std::filesystem::path internalParametersPath =
        m_dataPath / "internal_parameters.json";
    io::write(internalParametersPath, internalParamJSON);
    printWrite(m_verbose, true, "internal parameters", internalParametersPath);
  }

  /// \brief Write simulation data to file(s).
  ///
  /// \param inContext The context object.
  ///
  /// \return Void.
  void writeData(const Context<T> &inContext) {
    // All objects to save.
    const GridGPU<char> &domain = *(inContext.getGeometry()->getDomainGrid());
    const OrderParameter<T> &orderParameter = *(inContext.getOrderParameter());
    const GridGPU<T> &currentDensity =
        *(inContext.getComputeCurrent()->getResult());

    // Scale vector potential to our external units.
    const T penetrationDepth = inContext.getParameters()->getPenetrationDepth();
    const T vectorPotentialScale =
        penetrationDepth > static_cast<T>(0)
            ? static_cast<T>(1) / (penetrationDepth * penetrationDepth)
            : static_cast<T>(0);
    const GridGPU<T> vectorPotential =
        *(inContext.getInducedVectorPotential()) * vectorPotentialScale;

    // Compute flux density.
    const T latticeSpacing = inContext.getParameters()->getGridElementSize();
    const GridGPU<T> fluxDensity = vectorPotential.curl(latticeSpacing);

    switch (m_dataFormat) {
#ifdef CONGA_USE_HDF5
    case DataFormat::H5: {
      const std::filesystem::path fileName =
          m_dataPath / std::string("simulation.h5");
      auto file = fileOverWrite(fileName);

      // Write.
      io::write(domain, "domain", file);
      io::write(orderParameter, "order_parameter", file);
      io::write(currentDensity, "current_density", file);
      io::write(vectorPotential, "vector_potential", file);
      io::write(fluxDensity, "magnetic_flux_density", file);

      // Can it never fail?
      const bool success = true;
      printWrite(m_verbose, success, "all simulation data", fileName);
      break;
    }
#endif // CONGA_USE_HDF5
    case DataFormat::CSV: {
      // CSV is the default.
    }
    default: {
      // Write domain.
      {
        const std::filesystem::path fileName = m_dataPath / "domain.csv";
        const bool success = io::write(domain, {"domain"}, fileName);
        printWrite(m_verbose, success, "domain", fileName);
      }

      // Write order parameter.
      {
        // This is an ugly hack due to the CSVWriter not accepting an order
        // parameter as input.
        const GridGPU<T> &orderParameterGrid =
            *(inContext.getComputeOrderParameter()->getResult());
        const std::filesystem::path fileName =
            m_dataPath / "order_parameter.csv";
        const bool success =
            io::write(orderParameterGrid, orderParameter.getTypes(), fileName);
        printWrite(m_verbose, success, "order parameter", fileName);
      }

      // Write curent density.
      {
        const std::filesystem::path fileName =
            m_dataPath / "current_density.csv";
        const bool success =
            io::write(currentDensity, {"j_x", "j_y"}, fileName);
        printWrite(m_verbose, success, "current density", fileName);
      }

      // Write vector potential.
      {
        const std::filesystem::path fileName =
            m_dataPath / "vector_potential.csv";
        const bool success =
            io::write(vectorPotential, {"a_x", "a_y"}, fileName);
        printWrite(m_verbose, success, "vector potential", fileName);
      }

      // Write flux density.
      {
        const std::filesystem::path fileName =
            m_dataPath / "magnetic_flux_density.csv";
        const bool success = io::write(fluxDensity, {"b_z"}, fileName);
        printWrite(m_verbose, success, "magnetic flux density", fileName);
      }
      break;
    }
    }
  }

  // Path to data directory.
  const std::filesystem::path m_dataPath;

  // Data format of large data files.
  const DataFormat m_dataFormat;

  // Whether to print to terminal or not.
  const bool m_verbose;
};
} // namespace io
} // namespace conga

#endif // CONGA_SIMULATION_WRITER_H_
