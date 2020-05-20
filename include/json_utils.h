// Copyright (c) 2017-2020, University of Cincinnati, developed by Henry
// Schreiner under NSF AWARD 1414736 and by the respective contributors. All
// rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause
#ifndef JSON_UTILS_H_
#define JSON_UTILS_H_

#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace json_utils {

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
} // namespace json_utils

#endif // JSON_UTILS_H_
