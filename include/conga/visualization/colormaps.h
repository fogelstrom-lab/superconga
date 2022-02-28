#ifndef COLORMAPS_H_
#define COLORMAPS_H_

#include <cuda.h>

#ifndef _HD_
#define _HD_ __host__ __device__
#endif

namespace conga {
namespace colormaps {
/// \brief YlGnBu colormap.
struct YlGnBu {
  /// \brief Convert grayscale pixel in the range [0,1] to color.
  ///
  /// \param inVal The grayscale value.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  _HD_ static void compute(const float inVal, float &outRed, float &outGreen,
                           float &outBlue) {
    // The values below were obtained by fitting a polynomial to the Matplotlib
    // colormap. It is not exact, but reasonably close.

    // Red.
    const float r6 = 31.14985763f;
    const float r5 = -105.51096449f;
    const float r4 = 126.50620633f;
    const float r3 = -61.81929185f;
    const float r2 = 9.84384196f;
    const float r1 = -1.15995827f;
    const float r0 = 1.00879115f;

    // Green.
    const float g6 = -5.91477522f;
    const float g5 = 23.71285119f;
    const float g4 = -32.7233085f;
    const float g3 = 20.09376464f;
    const float g2 = -6.4745131f;
    const float g1 = 0.44749983f;
    const float g0 = 0.98621283f;

    // Blue.
    const float b6 = -13.67309768f;
    const float b5 = 35.34259532f;
    const float b4 = -29.63803511f;
    const float b3 = 4.38614756f;
    const float b2 = 4.83524471f;
    const float b1 = -1.77880347f;
    const float b0 = 0.85812289f;

    // Powers of the grayscale intensity.
    const float val1 = max(inVal, 0.0f);
    const float val2 = val1 * val1;
    const float val3 = val2 * val1;
    const float val4 = val2 * val2;
    const float val5 = val2 * val3;
    const float val6 = val3 * val3;

    // Set colors.
    outRed = r6 * val6 + r5 * val5 + r4 * val4 + r3 * val3 + r2 * val2 +
             r1 * val1 + r0;
    outGreen = g6 * val6 + g5 * val5 + g4 * val4 + g3 * val3 + g2 * val2 +
               g1 * val1 + g0;
    outBlue = b6 * val6 + b5 * val5 + b4 * val4 + b3 * val3 + b2 * val2 +
              b1 * val1 + b0;
  }
};

/// \brief Red-white-blue colormap (zero is mapped to white).
struct RdBu {
  /// \brief Convert grayscale pixel in the range [-1,+1] to color.
  ///
  /// \param inVal The grayscale value.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  _HD_ static void compute(const float inVal, float &outRed, float &outGreen,
                           float &outBlue) {
    // The values below were obtained by fitting a polynomial to the Matplotlib
    // colormap. It is not exact, but reasonably close.

    // Red.
    const float r6 = -0.70345843f;
    const float r5 = 0.31917521f;
    const float r4 = 1.48151386f;
    const float r3 = 0.00316337f;
    const float r2 = -1.53259179f;
    const float r1 = -0.48950088f;
    const float r0 = 0.96017916f;

    // Green.
    const float g6 = -0.28988233f;
    const float g5 = -0.35716038f;
    const float g4 = 1.10727099f;
    const float g3 = 0.34213596f;
    const float g2 = -1.6637647f;
    const float g1 = 0.10106193f;
    const float g0 = 0.94316445f;

    // Blue.
    const float b6 = -1.31362931f;
    const float b5 = -0.27708307f;
    const float b4 = 2.42015189f;
    const float b3 = -0.04662716f;
    const float b2 = -1.81557643f;
    const float b1 = 0.44075829f;
    const float b0 = 0.93207682f;

    // Powers of the grayscale intensity.
    const float val1 = inVal;
    const float val2 = val1 * val1;
    const float val3 = val2 * val1;
    const float val4 = val2 * val2;
    const float val5 = val2 * val3;
    const float val6 = val3 * val3;

    // Set colors.
    outRed = r6 * val6 + r5 * val5 + r4 * val4 + r3 * val3 + r2 * val2 +
             r1 * val1 + r0;
    outGreen = g6 * val6 + g5 * val5 + g4 * val4 + g3 * val3 + g2 * val2 +
               g1 * val1 + g0;
    outBlue = b6 * val6 + b5 * val5 + b4 * val4 + b3 * val3 + b2 * val2 +
              b1 * val1 + b0;
  }
};

/// \brief Twilight shifted reversed colormap (cyclic).
struct TwilightShiftedReversed {
  /// \brief Convert grayscale pixel in the range [-1,+1] to color.
  ///
  /// \param inVal The grayscale value.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  _HD_ static void compute(const float inVal, float &outRed, float &outGreen,
                           float &outBlue) {
    // The values below were obtained by fitting a polynomial to the Matplotlib
    // colormap. It is not exact, but reasonably close.

    // Red.
    const float rs3 = 0.024342f;
    const float rc3 = 0.05702154f;
    const float rs2 = -0.02748823f;
    const float rc2 = -0.00080247f;
    const float rs1 = -0.13564209f;
    const float rc1 = 0.2683719f;
    const float r0 = 0.54373334f;

    // Green.
    const float gs3 = -0.01150307f;
    const float gc3 = 0.00871481f;
    const float gs2 = 0.0046331f;
    const float gc2 = 0.02714641f;
    const float gs1 = 0.05230238f;
    const float gc1 = 0.38335767f;
    const float g0 = 0.41835701f;

    // Blue.
    const float bs3 = -0.01967491f;
    const float bc3 = 0.07132136f;
    const float bs2 = -0.00237468f;
    const float bc2 = 0.01179835f;
    const float bs1 = 0.18933604f;
    const float bc1 = 0.2378788f;
    const float b0 = 0.5418722f;

    // The different angles.
    const float val1 = inVal * static_cast<float>(M_PI);
    const float val2 = 2.0f * val1;
    const float val3 = 3.0f * val1;

    // Sin/cos of the angles.
    const float s3 = sin(val3);
    const float c3 = cos(val3);
    const float s2 = sin(val2);
    const float c2 = cos(val2);
    const float s1 = sin(val1);
    const float c1 = cos(val1);

    // Set colors.
    outRed =
        rs3 * s3 + rc3 * c3 + rs2 * s2 + rc2 * c2 + rs1 * s1 + rc1 * c1 + r0;
    outGreen =
        gs3 * s3 + gc3 * c3 + gs2 * s2 + gc2 * c2 + gs1 * s1 + gc1 * c1 + g0;
    outBlue =
        bs3 * s3 + bc3 * c3 + bs2 * s2 + bc2 * c2 + bs1 * s1 + bc1 * c1 + b0;
  }
};
} // namespace colormaps
} // namespace conga

#endif // COLORMAPS_H_
