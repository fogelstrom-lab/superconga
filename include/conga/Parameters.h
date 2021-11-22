//===--------------- Parameters.h - Parameter storage class. --------------===//
//
// Part of the SuperConga Project, under the GNU LGPL v3 license or higher.
// See LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Storage class for parameters, with convenience functions.
///
//===----------------------------------------------------------------------===//
#ifndef CONGA_PARAMETERS_H_
#define CONGA_PARAMETERS_H_

#include "ContextModule.h"
#include "defines.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace conga {
template <typename T> class Parameters : public ContextModule<T> {
public:
  Parameters() : m_version(2) {}
  ~Parameters() {}

  Parameters(Context<T> *inOutContext);

  Parameters &operator=(const Parameters &inOther);

  template <typename T2>
  Parameters<T> &operator=(const Parameters<T2> &inOther);

  template <typename T2>
  Parameters<T> &operatorEq(const Parameters<T2> &inOther);

  template <typename T2> operator Parameters<T2>() const;

  virtual int initialize() { return 0; }

  // Getters
  int getGridResolutionBase() const { return m_numGridBase; }
  int getGridResolutionCoherence() const { return m_numGridCoherence; }
  int getGridResolutionDeltaRotated() const { return m_numGridCoherence; }

  T getGrainWidth() const { return m_grainWidth; }
  T getGridElementSize() const {
    return static_cast<T>(1) / m_pointsPerCoherenceLength;
  }
  T getGrainFraction() const { return m_grainFraction; }
  int getAngularResolution() const { return m_numAngles; }
  T getAngleIncrement() const;

  int getNumOzakiEnergies() const;
  T getTemperature() const { return m_temperature; }
  T getNumFluxQuanta() const { return m_numFluxQuanta; }

  T getConvergenceCriterion() const { return m_convergenceCriterion; }

  T getEnergyCutoff() const { return m_energyCutoff; }

  T getEnergyMin() const { return m_energyMin; }
  T getEnergyMax() const { return m_energyMax; }
  int getNumEnergies() const { return m_numEnergies; }
  T getSmearingFactor() const { return m_smearingFactor; }

  T getPenetrationDepth() const { return m_penetrationDepth; }

  int isValid() const { return m_isValid; }

  // Setters
  // TODO: (Patric) Change name from grid to grain when referring to the
  // superconducting grain.
  // TODO: (Patric) Change all internal length units to hbar*<vF>/(2*pi*kB*Tc)
  // rather than hbar*<vF>/(kB*Tc). WARNING! This is not as trivial as it
  // seems, and will have far-reaching consequences with non-trivial scaling
  // factors.
  void setGrainWidth(const T inGrainWidth);
  void setPointsPerCoherenceLength(const T inPointsPerCoherenceLength);
  void setAngularResolution(const int inNumAngles) {
    m_numAngles = inNumAngles;
  }

  void setEnergyCutoff(const T inEnergyCutoff) {
    m_energyCutoff = inEnergyCutoff;
    m_numOzakiEnergies = computeNumOzakiEnergies(m_temperature, m_energyCutoff);
  };

  void setTemperature(const T inTemperature) {
    m_temperature = inTemperature;
    m_numOzakiEnergies = computeNumOzakiEnergies(m_temperature, m_energyCutoff);
  }
  void setNumFluxQuanta(const T inNumFluxQuanta) {
    m_numFluxQuanta = inNumFluxQuanta;
  }

  void setEnergyRange(const T inEnergyMin, const T inEnergyMax,
                      const int inNumEnergies,
                      const T inSmearingFactor = static_cast<T>(0.01));
  void setNumEnergies(const int inNumEnergies) {
    m_numEnergies = inNumEnergies;
  }

  void setConvergenceCriterion(const T inConvergenceCriterion) {
    m_convergenceCriterion = inConvergenceCriterion;
  }

  void setPenetrationDepth(const T inPenetrationDepth) {
    m_penetrationDepth = inPenetrationDepth;
  }

  void setValid() { m_isValid = 1; }
  void setInvalid() { m_isValid = 0; }

  // File I/O
  int read(const std::string &inFileName);
  int read(std::ifstream &inFile, const int inVersion);
  int read(const std::string &inFileName, const int inVersion);
  int write(const std::string &inFileName) const;

  // Helpers
  int getVersion() const { return m_version; }
  void print() const;

private:
  // =========== Discrete properties =========== //

  // Number of elements for coherence functions (gamma,gammabar) grid.
  int m_numGridCoherence;

  // Number of elements for order parameter solution.
  int m_numGridBase;

  // Side length of superconducting grain in units hbar*<vF>/(kB*Tc).
  T m_grainWidth;

  // The number of lattice points per coherence length.
  T m_pointsPerCoherenceLength;

  // The fraction of the simulation lattice maximally used for the grain.
  T m_grainFraction;

  // Number of angles to solve for.
  int m_numAngles;

  // Number of spectral energies.
  int m_numEnergies;

  // ===========Physical properties =========== //

  // Energy cutoff.
  T m_energyCutoff;

  // Temperature in units of Tc.
  T m_temperature;

  // The number of external flux-quanta.
  T m_numFluxQuanta;

  // The penetration depth used when solving for the vector potential.
  T m_penetrationDepth;

  // The number of Ozaki energies.
  T m_numOzakiEnergies;

  // Lowest energy to compute LDOS for.
  T m_energyMin;

  // Highest energy to compute LDOS for.
  T m_energyMax;

  // Energy smearing factor (for e.g. retarded properties).
  T m_smearingFactor;

  // Numerical properties.
  T m_convergenceCriterion;

  // Other
  const int m_version;
  int m_isValid;

  // A vector of the largest Ozaki pole with n+1 poles computed, where n is the
  // index of the vector.
  const std::vector<double> m_maxOzakiPole = {
      3.464101615138,      13.043193723013,    27.182805923222,
      46.319508681820,     70.526709814647,    99.819209820145,
      134.201334576035,    173.674723843717,   218.240115243641,
      267.897882477874,    322.648232120917,   382.491286239281,
      447.427120870950,    517.455785454923,   592.577313298030,
      672.791727521291,    758.099044591524,   848.499276499387,
      943.992432147387,    1044.578518259219,  1150.257539991102,
      1261.029501351330,   1376.894405494319,  1497.852254930454,
      1623.903051677801,   1755.046797373544,  1891.283493357799,
      2032.613140735255,   2179.035740424027,  2330.551293191541,
      2487.159799683865,   2648.861260447426,  2815.655675948037,
      2987.543046583090,   3164.523372695167,  3346.596654578822,
      3533.762892490810,   3726.022086653727,  3923.374237262044,
      4125.819344486381,   4333.357408476837,  4545.988429367131,
      4763.712407272654,   4986.529342298375,  5214.439234539534,
      5447.442084074518,   5685.537890980710,  5928.726655323640,
      6177.008377162879,   6430.383056551208,  6688.850693540401,
      6952.411288172675,   7221.064840486898,  7494.811350521523,
      7773.650818306208,   8057.583243876886,  8346.608627256021,
      8640.726968470750,   8939.938267543248,  9244.242524494641,
      9553.639739344733,   9868.129912114642,  10187.713042813368,
      10512.389131458542,  10842.158178074462, 11177.020182659693,
      11516.975145232140,  11862.023065802561, 12212.163944387836,
      12567.397780986943,  12927.724575614844, 13293.144328276247,
      13663.657038985704,  14039.262707740532, 14419.961334558257,
      14805.752919440674,  15196.637462387956, 15592.614963412661,
      15993.685422515518,  16399.848839707749, 16811.105214993084,
      17227.454548365906,  17648.896839834808, 18075.432089412709,
      18507.060297090567,  18943.781462892035, 19385.595586789594,
      19832.502668811354,  20284.502708945201, 20741.595707203986,
      21203.781663584734,  21671.060578106339, 22143.432450728560,
      22620.897281488116,  23103.455070385800, 23591.105817420932,
      24083.849522596051,  24581.686185888688, 25084.615807335540,
      25592.638386922201,  26105.753924657907, 26623.962420530934,
      27147.263874548138,  27675.658286698381, 28209.145657019460,
      28747.725985486431,  29291.399272088380, 29840.165516850884,
      30394.024719756279,  30952.976880822123, 31517.022000048237,
      32086.160077425615,  32660.391112956720, 33239.715106652184,
      33824.132058499723,  34413.641968492826, 35008.244836653459,
      35607.940662971036,  36212.729447455276, 36822.611190103285,
      37437.585890915667,  38057.653549855168, 38682.814167005199,
      39313.067742276238,  39948.414275748975, 40588.853767362038,
      41234.386217142870,  41885.011625110244, 42540.729991207598,
      43201.541315517512,  43867.445597926955, 44538.442838554773,
      45214.533037321766,  45895.716194298584, 46581.992309429610,
      47273.361382657822,  47969.823414089085, 48671.378403742987,
      49378.026351488850,  50089.767257488224, 50806.601121605097,
      51528.527943908797,  52255.547724346325, 52987.660462994907,
      53724.866159820733,  54467.164814822114, 55214.556427974414,
      55967.040999261437,  56724.618528703817, 57487.289016365059,
      58255.052462227519,  59027.908866262900, 59805.858228478566,
      60588.900548765567,  61377.035827283733, 62170.264063961804,
      62968.585258870233,  63771.999411872166, 64580.506523058772,
      65394.106592470736,  66212.799620066056, 67036.585605816275,
      67865.464549688913,  68699.436451832487, 69538.501312039167,
      70382.659130497530,  71231.909907065638, 72086.253641840856,
      72945.690334811923,  73810.219985944655, 74679.842595227819,
      75554.558162766305,  76434.366688360693, 77319.268172226453,
      78209.262614181425,  79104.350014321433, 80004.530372739711,
      80909.803689260734,  81820.169963918539, 82735.629196799040,
      83656.181387807985,  84581.826537007699, 85512.564644378072,
      86448.395710174183,  87389.319733820099, 88335.336715730969,
      89286.446655858366,  90242.649554185831, 91203.945410558954,
      92170.334225223487,  93141.815998147023, 94118.390728990082,
      95100.058418232264,  96086.819065539064, 97078.672670988453,
      98075.619234544327,  99077.658756581528, 100084.791236587014,
      101097.016674872531, 102114.335071418158};

  /// \brief Compute the number of Ozaki poles needed. If more than 200 is
  /// needed the exact number is approximated.
  ///
  /// \param inTemperature The temperature.
  /// \param inEnergyCutoff The energy cutoff.
  ///
  /// \return The number of poles.
  int computeNumOzakiEnergies(const T inTemperature, const T inEnergyCutoff);

  // Set the lattice resolution.
  void setLatticeResolution(const T inGrainWidth,
                            const T inPointsPerCoherenceLength);
  int computeGridResolutionRotated(const int numGridBase) const;
};

template <typename T>
Parameters<T>::Parameters(Context<T> *const inOutContext)
    : m_grainWidth(static_cast<T>(20)),
      m_pointsPerCoherenceLength(static_cast<T>(5)), m_numAngles(64),
      m_energyCutoff(static_cast<T>(10)), m_temperature(static_cast<T>(0.5)),
      m_numFluxQuanta(static_cast<T>(0)),
      m_convergenceCriterion(static_cast<T>(1e-6)), m_numEnergies(128),
      m_energyMin(static_cast<T>(0)), m_energyMax(static_cast<T>(3)),
      m_smearingFactor(static_cast<T>(0.05)), m_numOzakiEnergies(1),
      m_penetrationDepth(-static_cast<T>(1)), m_version(2), m_isValid(1) {
  ContextModule<T>::setContext(inOutContext);
  inOutContext->set(this);
  setLatticeResolution(m_grainWidth, m_pointsPerCoherenceLength);
}

template <typename T>
Parameters<T> &Parameters<T>::operator=(const Parameters<T> &inOther) {
  operatorEq(inOther);
  return *this;
}

template <typename T>
template <typename T2>
Parameters<T> &Parameters<T>::operator=(const Parameters<T2> &inOther) {
  operatorEq(inOther);
  return *this;
}

template <typename T>
template <typename T2>
Parameters<T>::operator Parameters<T2>() const {
  Parameters<T2> other;
  other = *this;
  return other;
}

// Is this necessary?
template <typename T>
template <typename T2>
Parameters<T> &Parameters<T>::operatorEq(const Parameters<T2> &inOther) {
#ifndef NDEBUG
  if (m_version != inOther.getVersion()) {
    std::cout << "Parameters<T>::operator=(): Warning, differing versions of "
                 "Parameters class\n";
  }
#endif

  m_grainWidth = inOther.getGrainWidth();
  m_numAngles = inOther.getAngularResolution();
  m_energyCutoff = inOther.getEnergyCutoff();
  m_temperature = inOther.getTemperature();
  m_numFluxQuanta = inOther.getNumFluxQuanta();
  m_convergenceCriterion = inOther.getConvergenceCriterion();
  m_numEnergies = inOther.getNumEnergies();
  m_energyMin = inOther.getEnergyMin();
  m_energyMax = inOther.getEnergyMax();
  m_smearingFactor = inOther.getSmearingFactor();
  m_isValid = inOther.isValid();

  m_numOzakiEnergies = inOther.m_numOzakiEnergies;
  m_penetrationDepth = inOther.m_penetrationDepth;

  m_grainFraction = inOther.m_grainFraction;
  m_grainWidth = inOther.m_grainWidth;
  m_pointsPerCoherenceLength = inOther.m_pointsPerCoherenceLength;

  m_numGridBase = inOther.m_numGridBase;
  m_numGridCoherence = inOther.m_numGridCoherence;

  return *this;
}

template <typename T> int Parameters<T>::getNumOzakiEnergies() const {
  return m_numOzakiEnergies;
}

template <typename T>
void Parameters<T>::setEnergyRange(const T inEnergyMin, const T inEnergyMax,
                                   const int inNumEnergies,
                                   const T inSmearingFactor) {
  m_energyMin = inEnergyMin;
  m_energyMax = inEnergyMax;
  m_smearingFactor = inSmearingFactor;

  // Do the limits make sense?
  // TODO(niclas): Shouldn't we check that min < max? And why set the number of
  // energies to 1 if max = 0?
  if (inEnergyMax == static_cast<T>(0.0) || inEnergyMax == inEnergyMin) {
    m_numEnergies = 1;
  } else {
    m_numEnergies = inNumEnergies;
  }
}

template <typename T> void Parameters<T>::setGrainWidth(const T inGrainWidth) {
  // Grain width in coherence lengths: hbar*<vF>/(kB*Tc).
  m_grainWidth = inGrainWidth;

  // Update the lattice resolution.
  setLatticeResolution(m_grainWidth, m_pointsPerCoherenceLength);
}

template <typename T>
void Parameters<T>::setPointsPerCoherenceLength(
    const T inPointsPerCoherenceLength) {
  // Points per coherence length: hbar*<vF>/(kB*Tc).
  m_pointsPerCoherenceLength = inPointsPerCoherenceLength;

  // Update the lattice resolution.
  setLatticeResolution(m_grainWidth, m_pointsPerCoherenceLength);
}

template <typename T>
int Parameters<T>::computeNumOzakiEnergies(const T inTemperature,
                                           const T inEnergyCutoff) {
  // The value to compare with in the poles data.
  const double val =
      2.0 * M_PI * static_cast<double>(inEnergyCutoff / inTemperature);

  // Get the pointer to the value of the first element larger than 'val'.
  const auto upper =
      std::upper_bound(m_maxOzakiPole.begin(), m_maxOzakiPole.end(), val);

  if (upper != m_maxOzakiPole.end()) {
    // The value was found, so we can compute the index, and thus the number
    // of needed poles.
    const int idx =
        static_cast<int>(std::distance(m_maxOzakiPole.begin(), upper));
    return idx + 1;
  } else {
#ifndef NDEBUG
    std::cerr << "-- WARNING! The temperature (cutoff) is too low (high). Will "
                 "approximate the number of Ozaki poles using a polynomial fit."
              << std::endl;
#endif
    // These parameters were obtained by fitting a second degree polynomial to
    // the poles data in Matlab. The fit is very good.
    const double c2 = 2.5465;
    const double c1 = 1.2744;
    const double c0 = 0.4651;

    // Loop until we find the approximate needed number of poles.
    int i = static_cast<int>(m_maxOzakiPole.size()) - 1;
    bool success = false;
    while (!success) {
      i++;
      const double x = static_cast<double>(i);
      const double approxVal = x * (c2 * x + c1) + c0;
      success = approxVal >= val;
    }
    return i;
  }
}

template <typename T>
void Parameters<T>::setLatticeResolution(const T inGrainWidth,
                                         const T inPointsPerCoherenceLength) {
  // The fractional number of lattice points required for the grain itself,
  // excluding safety margin.
  const T fracPointsGrain = inGrainWidth * inPointsPerCoherenceLength;

  // The total number of required points, including safety margin.
  const int numPointsTotal = static_cast<int>(std::ceil(fracPointsGrain)) +
                             2 * constants::GRAIN_SIZE_MARGIN;

  // The fraction of points for the grain.
  m_grainFraction = fracPointsGrain / static_cast<T>(numPointsTotal);

  // Set the number of points per side in the simulation.
  m_numGridBase = numPointsTotal;

  // Set the number of points per side with rotational margin.
  m_numGridCoherence = computeGridResolutionRotated(numPointsTotal);
}

template <typename T> T Parameters<T>::getAngleIncrement() const {
  return static_cast<T>(2.0 * M_PI) / static_cast<T>(m_numAngles);
}

template <typename T> int Parameters<T>::read(const std::string &inFileName) {
  std::ifstream file(inFileName.c_str(), std::ios::in | std::ios::binary);
  const int result = read(file, m_version);
  file.close();
  return result;
}

template <typename T>
int Parameters<T>::read(const std::string &inFileName, const int inVersion) {
  if (inVersion == m_version)
    return read(inFileName);
  else {
    std::ifstream file(inFileName.c_str(), std::ios::in | std::ios::binary);
    const int result = read(file, inVersion);
    file.close();
    return result;
  }
}

template <typename T>
int Parameters<T>::read(std::ifstream &inFile, const int inVersion) {
  if (inFile.is_open() && (inVersion == m_version)) {
    const int numBytes = sizeof(*this);
    inFile.read((char *)this, numBytes);
    return 0;

  } else {
    std::cout << "Could not load file, version: " << inVersion << "\n";
    return 1;
  }
}

template <typename T>
int Parameters<T>::write(const std::string &inFileName) const {
  const int numBytes = sizeof(*this);

  std::ofstream file(inFileName.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open()) {
    file.write((char *)(this), numBytes);
    file.close();
    return 0;
  } else {
    std::cout << "Could not save file: " << inFileName << std::endl;
    return 1;
  }
}

template <typename T>
int Parameters<T>::computeGridResolutionRotated(const int inNumGridBase) const {
  // The rotated size is given by:
  // N + 2x >= sqrt(2)N
  // ==>
  // x >= 0.5 * (sqrt(2) - 1) * N
  const double marginFractional =
      0.5 * (std::sqrt(2.0) - 1.0) * static_cast<double>(inNumGridBase);
  const int margin = static_cast<int>(std::ceil(marginFractional));
  const int numGridRotated = inNumGridBase + 2 * margin;
  return numGridRotated;
}

template <typename T> void Parameters<T>::print() const {
  std::cout << "Grain Resolution Base: " << m_numGridBase << std::endl;
  std::cout << "Grain Width: " << m_grainWidth << std::endl;
  std::cout << "Points per coherence length: " << m_pointsPerCoherenceLength
            << std::endl;
  std::cout << "Angular Resolution: " << m_numAngles << std::endl;
  std::cout << "Temperature: " << m_temperature << std::endl;
  std::cout << "Flux quanta: " << m_numFluxQuanta << std::endl;
  std::cout << "Spectral energy range: (" << m_energyMin << ", " << m_energyMax
            << ")" << std::endl;
  std::cout << "Spectral energy levels: " << m_numEnergies << std::endl;
  std::cout << "Convergence Criterion: " << m_convergenceCriterion << std::endl;
  std::cout << "Cutoff parameter: " << m_energyCutoff << std::endl;
}
} // namespace conga

#endif // CONGA_PARAMETERS_H_
