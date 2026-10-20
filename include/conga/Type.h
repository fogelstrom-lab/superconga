//===--------------------- Type.h - Enum definitions. ---------------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Convenient enum definitions, to make code and function arguments easier to
/// read.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_TYPE_H_
#define CONGA_TYPE_H_

namespace conga {
struct Type {
  enum type {
    real = 1,
    imag = 2,
    complex = 3,
    modulus = 4,
    all = 5,
    add = 6,
    remove = 7
  };
  enum location { start = 1, begin = 1, end = 2 };
};
} // namespace conga

#endif // CONGA_TYPE_H_
