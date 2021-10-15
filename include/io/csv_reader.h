//===---- csv_reader.h - functionality for reading data from csv files. ---===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Functions and classes for reading data from csv files.
///
//===----------------------------------------------------------------------===//
#ifndef CSV_READER_H_
#define CSV_READER_H_

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace conga {
namespace io {
template <typename T> class CSVReader {
private:
  // The delimiter character.
  const char m_delimiter = ',';

  // The names of columns to exclude. Our current CSV files all store the
  // xy-coordinates, in addition to the field values, so we should exclude them.
  const std::vector<std::string> m_excludeNames = {std::string("x_idx"),
                                                   std::string("y_idx")};

  // When reading the CSV file the result is put in this map. The key is the
  // same as the column name in file.
  std::map<std::string, std::vector<T>> m_map;

  // Whether the read was successful.
  bool m_success = false;

  /// \brief Get the column names in the CSV file.
  ///
  /// \param inOutStream The ifstream of the file.
  ///
  /// \return The column names as a vector of strings.
  std::vector<std::string> getColumnNames(std::ifstream &inOutStream) const;

public:
  /// \brief Default constructor.
  CSVReader() = default;

  /// \brief Constructor.
  ///
  /// \param inFileName Path to the file to read.
  CSVReader(const std::filesystem::path &inFileName) { read(inFileName); }

  /// \brief Read a CSV file.
  ///
  /// \param inFileName Path to file.
  ///
  /// \return Whether the read succeeded or not.
  bool read(const std::filesystem::path &inFileName);

  /// \brief Check if the read was successful.
  ///
  /// \return Whether the read succeeded or not.
  bool success() const { return m_success; }

  /// \brief Get the contents of the CSV file as a map.
  ///
  /// \return A map with keys equal to the column names in the file.
  const std::map<std::string, std::vector<T>> &getMap() const { return m_map; }

  /// \brief Check if the map contains a certain key.
  ///
  /// \param inKey The key to look for.
  ///
  /// \return Whether the key was found or not.
  bool contains(const std::string &inKey) const;
};

template <typename T>
std::vector<std::string>
CSVReader<T>::getColumnNames(std::ifstream &inOutStream) const {
  // Vector with all column names.
  std::vector<std::string> columnNames;

  // Get the first line (header).
  std::string line;
  std::getline(inOutStream, line);

  // Create stream for this line.
  std::stringstream ss(line);

  std::string columnName;
  while (std::getline(ss, columnName, m_delimiter)) {
    columnNames.push_back(columnName);
  }
  return columnNames;
}

template <typename T>
bool CSVReader<T>::read(const std::filesystem::path &inFileName) {
  // Set success to false.
  m_success = false;

  // Create stream;
  std::ifstream ifile(inFileName);

  // Abort early if not good.
  if (ifile.good()) {
    std::cout << "-- Reading file " << inFileName << "." << std::endl;
  } else {
    std::cerr << "-- ERROR! Could not open file " << inFileName << "."
              << std::endl;
    return m_success;
  }

  // Read column names from header.
  const std::vector<std::string> columnNames = getColumnNames(ifile);

  // Create keys in map for the columns we are interested in.
  for (const auto &columnName : columnNames) {
    bool excludeColumn = false;
    for (const auto &exclude : m_excludeNames) {
      excludeColumn |= (columnName == exclude);
    }
    if (!excludeColumn) {
      m_map[columnName] = {};
    }
  }

  // Continue reading the file, line by line.
  std::string line;
  while (std::getline(ifile, line)) {
    // Create stream for this line.
    std::stringstream ss(line);

    // Keep track of column index.
    int colIdx = 0;

    // The column value.
    T val;
    while (ss >> val) {
      // Get column name.
      const std::string columnName = columnNames[colIdx];

      // Check if the column should be excluded.
      // TODO(niclas): When moving to C++20 we can simply check if the key
      // exists in the map.
      bool excludeColumn = false;
      for (const auto &exclude : m_excludeNames) {
        excludeColumn |= (columnName == exclude);
      }

      // If the column should not be excluded, push back the value.
      if (!excludeColumn) {
        std::vector<T> &column = m_map[columnName];
        column.push_back(val);
      }

      // If the next token is a delimiter, ignore it and move on.
      if (ss.peek() == m_delimiter) {
        ss.ignore();
      }

      // Increment the column index.
      colIdx++;
    }
  }

  // Close the file.
  ifile.close();

  // Everything succeeded.
  m_success = true;

  return m_success;
}

template <typename T>
bool CSVReader<T>::contains(const std::string &inKey) const {
  // TODO(niclas): In C++20 we can use the method 'contains' instead.
  for (const auto [key, value] : m_map) {
    if (key == inKey) {
      return true;
    }
  }
  return false;
}
} // namespace io
} // namespace conga

#endif // CSV_READER_H_
