//===-------- io.h - Utility functions for reading/writing data. ----------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utility functions for reading/writing data.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_IO_H_
#define CONGA_IO_H_

#include "io/DataFormat.h"
#include "io/csv.h"
#ifdef CONGA_USE_HDF5
#include "io/h5.h"
#endif // CONGA_USE_HDF5
#include "io/json.h"
#include "io/print.h"

#endif // CONGA_IO_H_
