#ifndef COLORMAPS_H_
#define COLORMAPS_H_

#include <cuda.h>

#ifndef _HD_
#define _HD_ __host__ __device__
#endif

namespace conga {
namespace colormaps {
/// \brief Hot colormap.
struct Hot {
  /// \brief Convert grayscale pixel in the range [0,1] to color.
  ///
  /// \param inVal The grayscale value.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  _HD_ static void compute(const float inVal, float &outRed, float &outGreen,
                           float &outBlue) {
    // Set colors.
    const float scaledVal = 3.0f * inVal;
    outRed = min(max(scaledVal, 0.0f), 1.0f);
    outGreen = min(max(scaledVal - 1.0f, 0.0f), 1.0f);
    outBlue = min(max(scaledVal - 2.0f, 0.0f), 1.0f);
  }
};

/// \brief Red-white-blue colormap (zero is mapped to white).
struct RedWhiteBlue {
  /// \brief Convert grayscale pixel in the range [-1,+1] to color.
  ///
  /// \param inVal The grayscale value.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  _HD_ static void compute(const float inVal, float &outRed, float &outGreen,
                           float &outBlue) {
    // TODO(niclas): Change so that it conforms with Matplotlibs RdBu.

    // For convenience.
    const bool positive = inVal > 0.0f;
    const float tmp = 1.0f - abs(inVal);

    // Set colors.
    outRed = positive ? tmp : 1.0f;
    outGreen = tmp;
    outBlue = positive ? 1.0f : tmp;
  }
};

/// \brief Twilight colormap (cyclic).
struct Twilight {
  /// \brief Convert grayscale pixel in the range [-1,+1] to color.
  ///
  /// \param inVal The grayscale value.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  _HD_ static void compute(const float inVal, float &outRed, float &outGreen,
                           float &outBlue) {
    // TODO(niclas): This is a bit horrible. The values were obtained by fitting
    // 5-degree polynomials to Matplotlibs Twilight colormap.

    // Red negative.
    const float rn5 = 6.30819466f;
    const float rn4 = 12.57317797f;
    const float rn3 = 5.47363281f;
    const float rn2 = -1.3760424f;
    const float rn1 = -1.40483219f;
    const float rn0 = 0.12429275f;

    // Red positive.
    const float rp5 = -0.11196211f;
    const float rp4 = 2.91030803f;
    const float rp3 = -4.77245237f;
    const float rp2 = 1.6682225f;
    const float rp1 = 1.1296943f;
    const float rp0 = 0.13486743f;

    // Green negative.
    const float gn5 = -0.57212334f;
    const float gn4 = 0.64850747f;
    const float gn3 = 3.88774111f;
    const float gn2 = 3.71270168f;
    const float gn1 = 0.15889372f;
    const float gn0 = 0.03691657f;

    // Green positive.
    const float gp5 = 4.030221f;
    const float gp4 = -10.97781645f;
    const float gp3 = 9.8657821f;
    const float gp2 = -2.60150958f;
    const float gp1 = 0.58478138f;
    const float gp0 = 0.02210853f;

    // Blue negative.
    const float bn5 = 4.80058585f;
    const float bn4 = 14.12307721f;
    const float bn3 = 12.84216484f;
    const float bn2 = 2.8212216f;
    const float bn1 = -1.49734293f;
    const float bn0 = 0.15358266f;

    // Blue positive.
    const float bp5 = -6.91048841f;
    const float bp4 = 13.70431717f;
    const float bp3 = -6.18112154f;
    const float bp2 = -0.83681335f;
    const float bp1 = 1.01764403f;
    const float bp0 = 0.15516054f;

    // Powers of the grayscale intensity.
    const float val1 = inVal;
    const float val2 = inVal * inVal;
    const float val3 = val2 * inVal;
    const float val4 = val2 * val2;
    const float val5 = val2 * val3;

    // For convenience.
    const bool positive = inVal > 0.0f;

    // Set colors.
    outRed = positive ? (rp5 * val5 + rp4 * val4 + rp3 * val3 + rp2 * val2 +
                         rp1 * val1 + rp0)
                      : (rn5 * val5 + rn4 * val4 + rn3 * val3 + rn2 * val2 +
                         rn1 * val1 + rn0);
    outGreen = positive ? (gp5 * val5 + gp4 * val4 + gp3 * val3 + gp2 * val2 +
                           gp1 * val1 + gp0)
                        : (gn5 * val5 + gn4 * val4 + gn3 * val3 + gn2 * val2 +
                           gn1 * val1 + gn0);
    outBlue = positive ? (bp5 * val5 + bp4 * val4 + bp3 * val3 + bp2 * val2 +
                          bp1 * val1 + bp0)
                       : (bn5 * val5 + bn4 * val4 + bn3 * val3 + bn2 * val2 +
                          bn1 * val1 + bn0);
  }
};
} // namespace colormaps
} // namespace conga

#endif // COLORMAPS_H_
