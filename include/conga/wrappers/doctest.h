//===------- doctest_wrapper.h - wrapper for doctest inclusion. -------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Wrapper for including doctest, such that pragmas and definitions only have
/// to be written once, rather than in every file that includes doctest.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_DOCTEST_H_
#define CONGA_DOCTEST_H_

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

// Turn off irrelevant warnings coming from doctest internally.
#pragma GCC system_header

#include <doctest/doctest.h>

#endif // CONGA_DOCTEST_H_
