// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
#ifndef CONGA_CSV_ITERATION_WRITER_H_
#define CONGA_CSV_ITERATION_WRITER_H_

#include "../Context.h"

namespace conga {
namespace io {
/// \brief CSV iteration writer base class.
template <typename T> class CSVIterationWriter {
public:
  /// \brief Constructor.
  CSVIterationWriter() {}

  /// \brief Destructor.
  virtual ~CSVIterationWriter() {}

  /// \brief Write convergence info for this iteration to file.
  ///
  /// \param inContext Context object.
  ///
  /// \return Success.
  virtual bool write(const Context<T> &inContext) const = 0;
};
} // namespace io
} // namespace conga

#endif // CONGA_CSV_ITERATION_WRITER_H_
