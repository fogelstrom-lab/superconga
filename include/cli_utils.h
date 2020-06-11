//===----- cli_utils.h - Command line parsing (using JSON) --===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the default CLI11 parser, for parsing both command-line
/// options and JSON-configuration files.
///
//===----------------------------------------------------------------------===//
#ifndef CLI_UTILS_H_
#define CLI_UTILS_H_

#include "nlohmann_json_wrapper.h"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace cli_utils {
namespace internal {
// Copyright (c) 2017-2020, University of Cincinnati, developed by Henry
// Schreiner under NSF AWARD 1414736 and by the respective contributors. All
// rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/// A CLI11 custom config-formatter to read JSON config-files.
///
/// It does two things:
/// 1. Convert CLI11 application options (i.e. command line inputs) to a config
/// string
/// 2. Convert input stream from file to a vector of config items
class ConfigJSON : public CLI::Config {
public:
  /// \brief Parse command-line options from application using CLI11
  ///
  /// \param inApp - Pointer to the CLI11 application whose options are being
  ///                parsed
  /// \param inDefaultAlso - Whether there are default values
  /// \param unused
  /// \param unused
  ///
  /// \return - A CLI11 config as a string
  std::string to_config(const CLI::App *inApp, bool inDefaultAlso, bool,
                        std::string) const override {

    // The JSON that will be populated by the option names as keys, and passed
    // values.
    nlohmann::json parametersJSON;

    // Loop over all command-line options in the application, and check if they
    // are set by the user. If so populate the JSON.
    for (const CLI::Option *opt : inApp->get_options({})) {
      // Only process option with a long-name and if configurable.
      if (!opt->get_lnames().empty() && opt->get_configurable()) {
        // Get the name of the option.
        std::string name = opt->get_lnames()[0];

        if (opt->get_type_size() != 0) {
          if (opt->count() == 1) {
            // Option: one passed.
            parametersJSON[name] = opt->results().at(0);
          } else if (opt->count() > 1) {
            // Option: multiple passed.
            parametersJSON[name] = opt->results();
          } else if (inDefaultAlso && !opt->get_default_str().empty()) {
            // Option: not present, but there is a default value.
            parametersJSON[name] = opt->get_default_str();
          }
        } else if (opt->count() == 1) {
          // Flag: one passed.
          parametersJSON[name] = true;
        } else if (opt->count() > 1) {
          // Flag: multiple passed.
          parametersJSON[name] = opt->count();
        } else if (opt->count() == 0 && inDefaultAlso) {
          // Flag: not present.
          parametersJSON[name] = false;
        }
      }
    }

    // If there are subcommands, parse them recursively, i.e. call this method
    // on the subcommand.
    for (const CLI::App *subcom : inApp->get_subcommands({})) {
      parametersJSON[subcom->get_name()] =
          nlohmann::json(to_config(subcom, inDefaultAlso, false, ""));
    }

    // Convert to the JSON object to a string (i.e. serialize).
    const int indent = 4;
    const std::string config = parametersJSON.dump(indent);

    return config;
  }

  /// \brief Parse CLI11 options from JSON-config file
  ///
  /// The long option names (e.g. --long-option-name) and the JSON keys must be
  /// identical (except for '_' which is automatically converted to '-').
  ///
  /// \param input - Input stream from the JSON-file to parse.
  ///
  /// \return - A vector of config items
  std::vector<CLI::ConfigItem> from_config(std::istream &input) const override {
    // Read JSON file.
    nlohmann::json parametersJSON;
    input >> parametersJSON;

    // Convert to a vector of config items.
    return _from_config(parametersJSON);
  }

private:
  /// \brief Parse CLI11 options from JSON object
  ///
  /// \param inParametersJSON - JSON object
  /// \param inName - Name of the config item
  /// \param inPrefix - Names of parent items
  ///
  /// \return - A vector of config items
  std::vector<CLI::ConfigItem>
  _from_config(nlohmann::json inParametersJSON, std::string inName = "",
               std::vector<std::string> inPrefix = {}) const {
    std::vector<CLI::ConfigItem> results;

    if (inParametersJSON.is_object()) {
      // If inParametersJSON is an 'object', and not just a single item, the
      // contents cant iterated over.
      for (nlohmann::json::iterator item = inParametersJSON.begin();
           item != inParametersJSON.end(); ++item) {

        // TODO(niclas): I am not sure what is happening here.
        auto copyPrefix = inPrefix;
        if (!inName.empty()) {
          copyPrefix.push_back(inName);
        }

        // Get the key for this item, and replace '_' with '-', so one can have
        // snake_case in the external JSON-files.
        std::string key = item.key();
        std::replace(key.begin(), key.end(), '_', '-');

        // Process this item recursively.
        const auto subResults = _from_config(*item, key, copyPrefix);

        // Insert subResults into the total results.
        results.insert(results.end(), subResults.begin(), subResults.end());
      }
    } else if (!inName.empty()) {
      // If inParametersJSON is not an object, and there is a name, this
      // means that this method was called in the recursive sub-result
      // extraction above.

      // Expand the results vector by one and get the new, empty,
      // CLI::ConfigItem.
      results.emplace_back();
      CLI::ConfigItem &res = results.back();
      res.name = inName;
      res.parents = inPrefix;

      // Simply copy inParametersJSON value to the config item.
      if (inParametersJSON.is_boolean()) {
        res.inputs = {inParametersJSON.get<bool>() ? "true" : "false"};
      } else if (inParametersJSON.is_number()) {
        std::stringstream ss;
        ss << inParametersJSON.get<double>();
        res.inputs = {ss.str()};
      } else if (inParametersJSON.is_string()) {
        res.inputs = {inParametersJSON.get<std::string>()};
      } else if (inParametersJSON.is_array()) {
        for (std::string ival : inParametersJSON)
          res.inputs.push_back(ival);
      } else {
        throw CLI::ConversionError("Failed to convert " + inName);
      }
    } else {
      throw CLI::ConversionError(
          "You must make all top level values objects in json!");
    }

    return results;
  }
};
} // namespace internal

void makeDefaultApp(CLI::App &outApp, nlohmann::json &outParametersJSON,
                    bool &outPrintVersion, bool &outDumpConfig) {
  // Set message generated by the --help flag.
  outApp.description(
      {"SuprConGPU\nMain file for setting up self-consistent simulations in a "
       "confined\ngeometry (a mesoscopic grain). Different superconducting "
       "order-paramers and\ngeometries can be specified, as well as physical "
       "parameters such as temperature\nand external fields. To set these, do "
       "NOT modify the source file (i.e. the .cu\nfile) directly. Instead, to "
       "set up simulation properties and parameters, use\nthe command-line "
       "options specified below, or set up a .json parameter-file as\nin "
       "examples/ and give the path to this file with the -f option (see "
       "below).\nCommand-line options have highest priority."});

  // Specify the JSON-config formatter to use when parsing the JSON-parameter
  // files with CLI11.
  outApp.config_formatter(std::make_shared<internal::ConfigJSON>());

  // Format the help output.
  // Set the size of the left column (i.e. the options).
  const int helpOptionsColumnWidth = 45;
  outApp.get_formatter()->column_width(helpOptionsColumnWidth);

  // Set default values for user-modifieable parameters.
  outParametersJSON["data_path"] = std::string("data");
  outParametersJSON["temperature"] = 0.5;
  outParametersJSON["external_flux_quanta"] = 0.0;
  outParametersJSON["grain_side_length"] = 30.0;
  outParametersJSON["crystal_axes_rotation"] = 0.0;
  outParametersJSON["convergence_criterion"] = 1e-6;
  outParametersJSON["lattice_side_length"] = 201;
  outParametersJSON["num_ozaki_energies"] = 20;
  outParametersJSON["num_fermi_momenta"] = 55;
  outParametersJSON["num_iterations_max"] = 1000;
  outParametersJSON["num_iterations_min"] = 20;
  outParametersJSON["dwave_symmetry"] = true;
  outParametersJSON["disc_geometry"] = false;
  outParametersJSON["solenoid"] = false;
  outParametersJSON["visualize"] = true;
  outParametersJSON["verbose"] = true;
  outParametersJSON["save_data"] = false;
  outParametersJSON["save_data_interval"] = 200;

  // Tell CLI11 to the read JSON-parameter file, if specified.
  const std::string defaultFile = "";
  outApp
      .set_config("-f,--file", defaultFile,
                  "Path to JSON file with parameters specifying the "
                  "simulation settings.")
      ->check(CLI::ExistingFile);

  // Command-line options (argument followed by value)
  outApp.add_option<std::string>(
      "-o,--data-path",
      outParametersJSON["data_path"].get_ref<nlohmann::json::string_t &>(),
      "Base folder for output (data and parameter files).");
  outApp
      .add_option<double>("-T,--temperature",
                          outParametersJSON["temperature"]
                              .get_ref<nlohmann::json::number_float_t &>(),
                          "Temperature in units of T_c.")
      ->check(CLI::PositiveNumber);
  // TODO(niclas): Rename to "-F"?
  outApp
      .add_option<double>(
          "-B,--external-flux-quanta",
          outParametersJSON["external_flux_quanta"]
              .get_ref<nlohmann::json::number_float_t &>(),
          "External magnetic field, perpendicular to sample, in units of "
          "flux quanta.\nFor non-solenoid field, it is distributed uniformly "
          "over the sample.")
      ->check(CLI::Number);
  outApp
      .add_option<double>("-s,--grain-side-length",
                          outParametersJSON["grain_side_length"]
                              .get_ref<nlohmann::json::number_float_t &>(),
                          "Side length of grain in units of coherence lengths "
                          "(hbar*v_F)/(2*pi*k_B*T_c).")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<double>("-r,--crystal-axes-rotation",
                          outParametersJSON["crystal_axes_rotation"]
                              .get_ref<nlohmann::json::number_float_t &>(),
                          "Rotation between grain and crystal ab-axes.")
      ->check(CLI::Number);
  outApp
      .add_option<double>("-c,--convergence-criterion",
                          outParametersJSON["convergence_criterion"]
                              .get_ref<nlohmann::json::number_float_t &>(),
                          "Convergence criteron for self-consistency.")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<int64_t>(
          "-N,--lattice-side-length",
          outParametersJSON["lattice_side_length"]
              .get_ref<nlohmann::json::number_integer_t &>(),
          "Number of discrete lattice points in the grain. Set to odd value "
          "for best convergence.\nRecommended to be at least 10 times the "
          "grain "
          "size.")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<int64_t>(
          "-e,--num-ozaki-energies",
          outParametersJSON["num_ozaki_energies"]
              .get_ref<nlohmann::json::number_integer_t &>(),
          "Number of Ozaki energies (similar to Matsubara energies, see "
          "the documentation).")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<int64_t>(
          "-p,--num-fermi-momenta",
          outParametersJSON["num_fermi_momenta"]
              .get_ref<nlohmann::json::number_integer_t &>(),
          "number of discrete angles in Fermi-surface average. Set to "
          "odd value for best convergence.\nRecommended minumum: 101.")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<int64_t>(
          "-M,--num-iterations-max",
          outParametersJSON["num_iterations_max"]
              .get_ref<nlohmann::json::number_integer_t &>(),
          "Maximum number of self-consistent iterations to run.")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<int64_t>(
          "-m,--num-iterations-min",
          outParametersJSON["num_iterations_min"]
              .get_ref<nlohmann::json::number_integer_t &>(),
          "Minimum number of self-consistent iterations to run.")
      ->check(CLI::PositiveNumber);
  outApp
      .add_option<int64_t>("--save-data-interval",
                           outParametersJSON["save_data_interval"]
                               .get_ref<nlohmann::json::number_integer_t &>(),
                           "How frequently to save data.")
      ->check(CLI::PositiveNumber);

  // Command-line flags (not followed by value: set something to true/false)
  outApp.add_flag("--dwave-symmetry,!--swave-symmetry",
                  outParametersJSON["dwave_symmetry"]
                      .get_ref<nlohmann::json::boolean_t &>(),
                  "Run with d-wave pairing symmetry (otherwise s-wave).");
  outApp.add_flag(
      "--disc-geometry,!--square-geometry",
      outParametersJSON["disc_geometry"].get_ref<nlohmann::json::boolean_t &>(),
      "Run with disc geometry (otherwise square).");
  outApp.add_flag(
      "--solenoid,!--no-solenoid",
      outParametersJSON["solenoid"].get_ref<nlohmann::json::boolean_t &>(),
      "Set up a solenoid, and use solenoid gauge for external field.");
  outApp.add_flag(
      "--visualize,!--no-visualize",
      outParametersJSON["visualize"].get_ref<nlohmann::json::boolean_t &>(),
      "Run without real-time plotting.");
  outApp.add_flag(
      "-v,--verbose,!--no-verbose",
      outParametersJSON["verbose"].get_ref<nlohmann::json::boolean_t &>(),
      "Print iteration status and other output to terminal.");
  // TODO(niclas): Rename to "--no-save-data"?
  outApp.add_flag(
      "--save-data,!--no-save",
      outParametersJSON["save_data"].get_ref<nlohmann::json::boolean_t &>(),
      "Save data files to disk.");

  // Arguments not part of the main JSON.
  outApp.add_flag("--version", outPrintVersion,
                  "Print version number and exit.");
  outApp.add_flag("--dump-config", outDumpConfig,
                  "Dump configuration options to the terminal, or to file "
                  "with '> config.json'.");
}

} // namespace cli_utils

#endif // CLI_UTILS_H_
