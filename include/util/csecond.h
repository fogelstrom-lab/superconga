#include <sys/time.h>
#include <sys/types.h>

// TODO(patric): This file should be replaced with more standardized/best
// practice functionality to measure time.
inline double csecond() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1.0e-6;
}
