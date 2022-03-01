#ifndef VISUALIZER_H_
#define VISUALIZER_H_

#include "../Type.h"
#include "../grid.h"
#include "colormaps.h"

#include <cuda.h>
#include <thrust/device_vector.h>

#include <forge.h>
#define USE_FORGE_CPU_COPY_HELPERS
#include <fg/compute_copy.h>

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace conga {
/// \brief Colormap enum.
enum class Colormap { ANGLE, DEFAULT, MAGNITUDE };

template <typename T> class Visualizer {
private:
  // The window.
  forge::Window m_window;

  // The OpenGL handle.
  GfxHandle *m_handle;

  // The window title.
  const std::string m_title;

  // The number of subplot columns and rows.
  const int m_cols;
  const int m_rows;

  // The domain labels.
  const GridGPU<char> m_domain;

  // The precision of printed floating-point numbers.
  const int m_precision;

  /// \brief Draw raw (color) data to a subplot.
  ///
  /// \param inTitle The subplot title.
  /// \param inCol The column index to plot it.
  /// \param inRow The row index to plot it.
  /// \param inData The raw pixel data.
  /// \param inWidth The width of the data in pixels.
  /// \param inHeight The height of the data in pixels.
  ///
  /// \return Void.
  void draw(const std::string &inTitle, const int inCol, const int inRow,
            const float *const inData, const int inWidth, const int inHeight);

  /// \brief Draw a grid to a subplot.
  ///
  /// \param inTitle The title of the subplot (the range will be appended).
  /// \param inCol The column index to plot it.
  /// \param inRow The row index to plot it.
  /// \param inGrid The grid to plot.
  /// \param inField The field of the grid to plot.
  /// \param inColormap Which colormap to use.
  /// \param inErase Whether to erase the values outside of the domain.
  /// \param inRangeScale Factor to scale the printed range with.
  ///
  /// \return Void.
  void draw(const std::string &inTitle, const int inCol, const int inRow,
            const GridGPU<T> &inGrid, const int inField,
            const Colormap inColormap, const bool inErase,
            const T inRangeScale);

public:
  /// \brief Constructor of the visualizer class.
  ///
  /// \param inTitle The window title.
  /// \param inColumns The number of subplot columns.
  /// \param inRows The number of subplot rows.
  /// \param inDomain The domain labels.
  /// \param inPrecision The precision of printed floating-point numbers.
  Visualizer(const std::string inTitle, const int inColumns, const int inRows,
             const GridGPU<char> &inDomain, const int inPrecision = 2);

  /// \brief Destructor of the visualizer class.
  ~Visualizer() { releaseGLBuffer(m_handle); };

  /// \brief Show the window if hidden.
  ///
  /// \return Void.
  void show() { m_window.show(); }

  /// \brief Draw a grid to a subplot column. If complex the magnitude and the
  /// phase will be plotted along the column. If real and 2D the magnitude and
  /// xy-components will be plotted. If real and 1D the grid itself will be
  /// plotted.
  ///
  /// \param inTitle The title of the subplots (the range will be appended).
  /// \param inCol The column index to plot it.
  /// \param inGrid The grid to plot.
  /// \param inErase Whether to erase the values outside of the domain.
  /// \param inRangeScale Factor to scale the printed range with.
  ///
  /// \return Void.
  void draw(const std::string &inTitle, const int inCol,
            const GridGPU<T> &inGrid, const bool inErase, const T inRangeScale);

  /// \brief Draw a grid to a subplot column. If complex the magnitude and the
  /// phase will be plotted along the column. If real and 2D the magnitude and
  /// xy-components will be plotted. If real and 1D the grid itself will be
  /// plotted.
  ///
  /// \param inTitle The title of the subplots (the range will be appended).
  /// \param inCol The column index to plot it.
  /// \param inGrid The grid to plot.
  /// \param inErase Whether to erase the values outside of the domain.
  ///
  /// \return Void.
  void draw(const std::string &inTitle, const int inCol,
            const GridGPU<T> &inGrid, const bool inErase);

  /// \brief Set additional information that will be plotted in the window
  /// title.
  ///
  /// \param inRunning Is the simulation still running?
  ///
  /// \return Void.
  void setInfo(const bool inRunning);

  /// \brief Swap buffers of the window, i.e. update the graphics.
  ///
  /// \return Void.
  void swapBuffers() { m_window.swapBuffers(); };

  /// \brief Check if the window is closed.
  ///
  /// \return True/false if the window is closed/open.
  bool close() { return m_window.close(); }
};

namespace internal {
/// \brief CUDA kernel for converting a grayscale image to color.
///
/// \param inDimX The width of the image.
/// \param inDimY The height of the image.
/// \param inScale Scaling factor to bring the image to [-1,+1] or [0,1].
/// \param inErase Whether to erase values outside of the domain.
/// \param inSrc The grayscale image pixels.
/// \param inDomain The domain labels.
/// \param outDst The resulting color image.
///
/// \return Void.
template <typename T, typename Colormap>
__global__ void convertToColor(const int inDimX, const int inDimY,
                               const T inScale, const bool inErase,
                               const T *const inSrc, const char *const inDomain,
                               float *const outDst) {
  // Indices of spatial coordinates.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < inDimX && y < inDimY) {
    // Source index.
    const int idxSrc = y * inDimX + x;

    // Scale value.
    const float val = static_cast<float>(inSrc[idxSrc] * inScale);

    // The destination index. Note that the y-axis is flipped. This is because
    // Forge uses standard image coordinates with the y-axis being positive
    // downward.
    const int idxDst = 3 * ((inDimY - 1 - y) * inDimX + x);

    // Set colors.
    float red;
    float green;
    float blue;
    Colormap::compute(val, red, green, blue);

    // Erase values outside of the domain, i.e. set them to gray.
    const bool outside = inDomain[idxSrc] == 0;
    const bool erase = outside && inErase;
    const float outsideVal = 0.2f;
    red = erase ? outsideVal : red;
    green = erase ? outsideVal : green;
    blue = erase ? outsideVal : blue;

    // Color the border black.
    const bool border =
        (x == 0) || (x == inDimX - 1) || (y == 0) || (y == inDimY - 1);
    const float borderVal = 0.0f;
    red = border ? borderVal : red;
    green = border ? borderVal : green;
    blue = border ? borderVal : blue;

    // Write.
    outDst[idxDst] = red;
    outDst[idxDst + 1] = green;
    outDst[idxDst + 2] = blue;
  }
}
} // namespace internal

template <typename T>
Visualizer<T>::Visualizer(const std::string inTitle, const int inColumns,
                          const int inRows, const GridGPU<char> &inDomain,
                          const int inPrecision)
    : m_title(inTitle), m_window(1366, 768, inTitle.c_str(), nullptr, true),
      m_handle(nullptr), m_domain(inDomain), m_cols(inColumns), m_rows(inRows),
      m_precision(inPrecision) {
  assert(m_cols > 0);
  assert(m_rows > 0);

  // This is the main window.
  m_window.makeCurrent();

  // Set font.
  forge::Font font;
  font.loadSystemFont("Helvetica Narrow");
  m_window.setFont(&font);
}

template <typename T> void Visualizer<T>::setInfo(const bool inRunning) {
  // Set status string.
  const std::string status =
      inRunning ? std::string("Running...") : std::string("Done!");

  // Set delimiter string.
  // TODO(niclas): Make member.
  const std::string delimiter = std::string(" | ");

  // Set total title.
  const std::string title = m_title + delimiter + status;

  // Update window title.
  m_window.setTitle(title.c_str());
}

template <typename T>
void Visualizer<T>::draw(const std::string &inTitle, const int inCol,
                         const GridGPU<T> &inGrid, const bool inErase,
                         const T inRangeScale) {
  const bool isComplex = inGrid.isComplex();
  if (isComplex) {
    assert(inGrid.numFields() == 1);

    // Draw magnitude.
    const std::string titleMagnitude =
        std::string("|") + inTitle + std::string("|");
    draw(titleMagnitude, inCol, 0, inGrid.abs(), 0, Colormap::MAGNITUDE,
         inErase, inRangeScale);

    // Draw phase.
    const std::string titlePhase =
        std::string("arg(") + inTitle + std::string(")");
    const T phaseRangeScale = static_cast<T>(1);
    draw(titlePhase, inCol, 1, inGrid.complexPhase(), 0, Colormap::ANGLE,
         inErase, phaseRangeScale);
  } else if (inGrid.numFields() == 2) {
    // Draw magnitude.
    const std::string titleMagnitude =
        std::string("|") + inTitle + std::string("|");
    draw(titleMagnitude, inCol, 0, inGrid.fieldMagnitude(), 0,
         Colormap::MAGNITUDE, inErase, inRangeScale);

    // Draw x-component.
    const std::string titleX = inTitle + std::string("_x");
    draw(titleX, inCol, 1, inGrid, 0, Colormap::DEFAULT, inErase, inRangeScale);

    // Draw y-component.
    const std::string titleY = inTitle + std::string("_y");
    draw(titleY, inCol, 2, inGrid, 1, Colormap::DEFAULT, inErase, inRangeScale);
  } else {
    assert(inGrid.numFields() == 1);
    draw(inTitle, inCol, 0, inGrid, 0, Colormap::DEFAULT, inErase,
         inRangeScale);
  }
}

template <typename T>
void Visualizer<T>::draw(const std::string &inTitle, const int inCol,
                         const GridGPU<T> &inGrid, const bool inErase) {
  const T rangeScale = static_cast<T>(1);
  draw(inTitle, inCol, inGrid, inErase, rangeScale);
}

template <typename T>
void Visualizer<T>::draw(const std::string &inTitle, const int inCol,
                         const int inRow, const GridGPU<T> &inGrid,
                         const int inField, const Colormap inColormap,
                         const bool inErase, const T inRangeScale) {
  const int width = inGrid.dimX();
  const int height = inGrid.dimY();

  assert(m_domain.dimX() == width);
  assert(m_domain.dimY() == height);
  assert(inGrid.representation() == Type::real);

  // Compute min/max values.
  const auto [minVal, maxVal] = inGrid.minmax(inField, Type::real);

  T scale = static_cast<T>(0);
  if (inColormap == Colormap::ANGLE) {
    // The range is always [-pi, pi)].
    scale = static_cast<T>(M_1_PI);
  } else {
    // Scaling factor in order to bring the values into [-1,+1] or [0,+1].
    const T lim = std::max(std::abs(maxVal), std::abs(minVal));
    const T epsilon = std::numeric_limits<T>::epsilon();
    scale = static_cast<T>(1) / std::max(lim, epsilon);
  }

  // Image pixels. Must be float.
  const int numChannels = 3;
  const int numPixels = width * height * numChannels;
  thrust::device_vector<float> imageGPU(numPixels);
  float *const ptrGPU = thrust::raw_pointer_cast(imageGPU.data());

  // Kernel parameters.
  dim3 numBlocks;
  dim3 numThreadsPerBlock;
  inGrid.getKernelDimensions(numBlocks, numThreadsPerBlock);

  switch (inColormap) {
  case Colormap::MAGNITUDE: {
    // Convert to hot and flip y-axis.
    internal::convertToColor<T, colormaps::YlGnBu>
        <<<numBlocks, numThreadsPerBlock>>>(width, height, scale, inErase,
                                            inGrid.getDataPointer(inField),
                                            m_domain.getDataPointer(), ptrGPU);
    break;
  }
  case Colormap::DEFAULT: {
    // Convert to red-white-blue and flip y-axis.
    internal::convertToColor<T, colormaps::RdBu>
        <<<numBlocks, numThreadsPerBlock>>>(width, height, scale, inErase,
                                            inGrid.getDataPointer(inField),
                                            m_domain.getDataPointer(), ptrGPU);
    break;
  }
  case Colormap::ANGLE: {
    // Convert to twlight and flip y-axis.
    internal::convertToColor<T, colormaps::TwilightShiftedReversed>
        <<<numBlocks, numThreadsPerBlock>>>(width, height, scale, inErase,
                                            inGrid.getDataPointer(inField),
                                            m_domain.getDataPointer(), ptrGPU);
    break;
  }
  default: {
    std::cerr << "-- ERROR! Colormap not supported." << std::endl;
  }
  }

  // Add range to title.
  std::stringstream stream;
  stream << std::scientific << std::setprecision(m_precision) << "["
         << minVal * inRangeScale << ", " << maxVal * inRangeScale << "]";
  const std::string range = stream.str();
  const std::string title = inTitle + std::string(" : ") + range;

  // Copy to the CPU and pass the CPU pointer to Forge. This way we can
  // visualize the simulation even if we are using different GPUs for
  // computation and visualization.
  const thrust::host_vector<float> imageCPU(imageGPU);
  const float *const ptrCPU = thrust::raw_pointer_cast(imageCPU.data());
  draw(title, inCol, inRow, ptrCPU, width, height);
}

template <typename T>
void Visualizer<T>::draw(const std::string &inTitle, const int inCol,
                         const int inRow, const float *const inData,
                         const int inWidth, const int inHeight) {
  // Index of subplot.
  const int index = inRow * m_cols + inCol;

  // Sanity checks.
  assert(index >= 0);
  assert(index < m_cols * m_rows);

  // Create image.
  const forge::Image image(inWidth, inHeight, FG_RGB, forge::f32);

  // Without releasing the memory usage grows.
  if (m_handle) {
    releaseGLBuffer(m_handle);
  }
  createGLBuffer(&m_handle, image.pixels(), FORGE_IMAGE_BUFFER);
  copyToGLBuffer(m_handle, (ComputeResourceHandle)inData, image.size());

  // Draw image.
  m_window.draw(m_rows, m_cols, index, image, inTitle.c_str());
}
} // namespace conga

#endif // VISUALIZER_H_
