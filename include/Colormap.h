//===----- Colormap.h - Colormaps for heatmaps in live visualization. -----===//
//
// Part of the SuprConGPU Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Colormaps used in the heatmaps in the OpenGL live visualization.
///
//===----------------------------------------------------------------------===//
#ifndef COLORMAP_H_
#define COLORMAP_H_

#include <algorithm>
#include <iostream>
#include <vector>

namespace scg {
class Colormap {
private:
  // Simple color struct.
  struct Color {
    float red;
    float green;
    float blue;

    /// \brief Color constructor.
    ///
    /// \param inRed The red intensity.
    /// \param inGreen The green intensity.
    /// \param inBlue The blue intensity.
    Color(const float inRed, const float inGreen, const float inBlue)
        : red(inRed), green(inGreen), blue(inBlue) {}
  };

  // The colormap defined by Mikael Håkansson.
  const std::vector<Color> m_jet = {
      Color(0.0f, 0.0f, 0.0f), Color(0.0f, 0.0f, 1.0f),
      Color(0.0f, 1.0f, 1.0f), Color(1.0f, 1.0f, 0.0f),
      Color(1.0f, 0.0f, 0.0f), Color(0.7f, 0.6f, 1.0f)};

  // Taken from Matplotlib. Not all values though, but equally spaced.
  const std::vector<Color> m_hot = {
      Color(0.0416f, 0.0f, 0.0f),     Color(0.2989711f, 0.0f, 0.0f),
      Color(0.56663705f, 0.0f, 0.0f), Color(0.82400815f, 0.0f, 0.0f),
      Color(1.0f, 0.09166748f, 0.0f), Color(1.0f, 0.34902f, 0.0f),
      Color(1.0f, 0.61666662f, 0.0f), Color(1.0f, 0.87401915f, 0.0f),
      Color(1.0f, 1.0f, 0.2124992f),  Color(1.0f, 1.0f, 0.59852901f),
      Color(1.0f, 1.0f, 1.0f)};

  // Taken from Matplotlib. Not all values though, but equally spaced.
  const std::vector<Color> m_rdbu = {
      Color(0.40392157f, 0.0f, 0.12156863f),
      Color(0.6922722f, 0.0922722f, 0.16770473),
      Color(0.83921569f, 0.37647059f, 0.30196078f),
      Color(0.95455594f, 0.64175317f, 0.50572857f),
      Color(0.99215686f, 0.85882353f, 0.78039216f),
      Color(0.96908881f, 0.96647443f, 0.96493656f),
      Color(0.81960784f, 0.89803922f, 0.94117647f),
      Color(0.57739331f, 0.77500961f, 0.87197232f),
      Color(0.2627451f, 0.57647059f, 0.76470588f),
      Color(0.13202614f, 0.40346021f, 0.67627835f),
      Color(0.01960784f, 0.18823529f, 0.38039216f)};

public:
  /// \brief Colormap constructor.
  Colormap() {}

  /// \brief Enum of the supported colormap types.
  enum class Type { JET, HOT, RDBU };

  /// \brief Get the color for a specific intensity, and colormap type.
  ///
  /// \param inIntensity A value in the range [0, 1].
  /// \param inColormapType Enum, which colormap to use.
  /// \param outRed The resulting red intensity.
  /// \param outGreen The resulting green intensity.
  /// \param outBlue The resulting blue intensity.
  void getColor(const float inIntensity, const Type inColormapType,
                float &outRed, float &outGreen, float &outBlue) const {
    // Clamp, just in case.
    const float intensity = std::clamp(inIntensity, 0.0f, 1.0f);

    // Set colormap.
    std::vector<Color> colormap;
    switch (inColormapType) {
    case Type::JET: {
      colormap = m_jet;
      break;
    }
    case Type::HOT: {
      colormap = m_hot;
      break;
    }
    case Type::RDBU: {
      colormap = m_rdbu;
      break;
    }
    default: {
      std::cerr << "-- ERROR! Colormap not implemented!" << std::endl;
    }
    }

    // The number of colors in this colormap.
    const int numColors = static_cast<int>(colormap.size());

    // Fractional index.
    const float fracIdx = intensity * static_cast<float>(numColors - 1);

    // The indices of the two colors to interpolate between.
    const int lowIdx = static_cast<int>(fracIdx);
    const int highIdx = lowIdx < (numColors - 1) ? lowIdx + 1 : lowIdx;

    // The weights of the respective colors.
    const float highWeight = fracIdx - static_cast<float>(lowIdx);
    const float lowWeight = 1.0f - highWeight;

    // Get the colors to interpolate between.
    const Color low = colormap[lowIdx];
    const Color high = colormap[highIdx];

    // Interpolate the colors.
    outRed = low.red * lowWeight + high.red * highWeight;
    outGreen = low.green * lowWeight + high.green * highWeight;
    outBlue = low.blue * lowWeight + high.blue * highWeight;
  }
};
} // namespace scg

#endif // COLORMAP_H_
