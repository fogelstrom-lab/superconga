#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ContextModule.h"
#include "defines.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>

template <typename, typename> class GridData;

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

  T getGrainWidthCoherenceLengths() const {
    return m_grainWidthCoherenceLengths;
  }
  T getGridElementSize() const { return m_gridElementSize; }
  int getAngularResolution() const { return m_numAngles; }
  T getAngleIncrement() const;

  int getNumMatsubaraEnergies() const;
  T getTemperature() const { return m_temperature; }
  T getMagneticFieldStrength() const { return m_magneticFieldStrength; }

  T getConvergenceCriterion() const { return m_convergenceCriterion; }

  T getCutOff() const { return m_energyCutoff; }

  T getEnergyMin() const { return m_energyMin; }
  T getEnergyMax() const { return m_energyMax; }
  int getNumEnergies() const { return m_numEnergies; }
  T getSmearingFactor() const { return m_smearingFactor; }

  T getPenetrationDepth() const { return m_penetrationDepth; }

  int isValid() const { return m_isValid; }

  // Setters
  // TODO: (Patric) Change name from grid to grain when referring to the
  // superconducting grain.
  // TODO: (Patric) Change all internal length units to hbar*vF/(2*pi*kB*Tc)
  // rather than hbar*vF/(kB*Tc). WARNING! This is not as trivial as it
  // seems, and will have far-reaching consequences with non-trivial scaling
  // factors.
  void setGrainWidthCoherenceLengths(const T inGrainWidthCoherenceLengths);
  void setGridResolution(const int inNumGridBase);
  void setAngularResolution(const int inNumAngles) {
    m_numAngles = inNumAngles;
  }

  void setCutoff(const T inCutoff) { m_energyCutoff = inCutoff; }

  void setNumMatsubaraEnergies(const int inNumMatsubaraEnergies) {
    m_numMatsubaraEnergies = inNumMatsubaraEnergies;
  }

  void setTemperature(const T inTemperature) { m_temperature = inTemperature; }
  void setMagneticFieldStrength(const T inMagneticFieldStrength) {
    m_magneticFieldStrength = inMagneticFieldStrength;
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
  static int computeGridResolutionRotated(const int numGridBase);
  int getVersion() const { return m_version; }
  void print() const;

private:
  // =========== Discrete properties =========== //

  // Number of elements for coherence functions (gamma,gammabar) grid.
  int m_numGridCoherence;

  // Number of elements for order parameter solution.
  int m_numGridBase;

  // Side length of superconducting grain in units h_bar * <vF> / (kB * Tc).
  T m_grainWidthCoherenceLengths;

  // Width of one grid element.
  T m_gridElementSize;

  // Number of angles to solve for.
  int m_numAngles;

  // Number of spectral energies.
  int m_numEnergies;

  // ===========Physical properties =========== //

  // Energy cutoff.
  T m_energyCutoff;

  // Temperature in units of Tc.
  T m_temperature;

  // Magnetic field.
  T m_magneticFieldStrength;

  // The penetration depth used when solving for the vector potential.
  // TODO(niclas): In what units? I guess coherence lengths.
  T m_penetrationDepth;

  // The number of Matsubara energies.
  T m_numMatsubaraEnergies;

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

  // Helper functions
  void setGridElementSize();
};

template <typename T>
Parameters<T>::Parameters(Context<T> *const inOutContext)
    : m_grainWidthCoherenceLengths(1.0), m_numAngles(0), m_energyCutoff(10),
      m_temperature(0.5), m_magneticFieldStrength(0.0),
      m_convergenceCriterion(0.0001), m_numEnergies(0), m_energyMin(0.0),
      m_energyMax(0.0), m_smearingFactor(0.01), m_numMatsubaraEnergies(0),
      m_penetrationDepth(100.0), m_version(2), m_isValid(1) {
  ContextModule<T>::setContext(inOutContext);
  inOutContext->set(this);
  // TODO(niclas): This is ugly.
  const int numGridBase = 101;
  setGridResolution(numGridBase);
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
  if (m_version != inOther.getVersion())
    std::cout << "Parameters<T>::operator=(): Warning, differing versions of "
                 "Parameters class\n";

  m_grainWidthCoherenceLengths = inOther.getGrainWidthCoherenceLengths();
  m_numAngles = inOther.getAngularResolution();
  m_energyCutoff = inOther.getCutOff();
  m_temperature = inOther.getTemperature();
  m_magneticFieldStrength = inOther.getMagneticFieldStrength();
  m_convergenceCriterion = inOther.getConvergenceCriterion();
  m_numEnergies = inOther.getNumEnergies();
  m_energyMin = inOther.getEnergyMin();
  m_energyMax = inOther.getEnergyMax();
  m_smearingFactor = inOther.getSmearingFactor();
  m_isValid = inOther.isValid();

  m_numMatsubaraEnergies = inOther.m_numMatsubaraEnergies;
  m_penetrationDepth = inOther.m_penetrationDepth;

  setGridResolution(inOther.getGridResolutionBase());

  return *this;
}

template <typename T> int Parameters<T>::getNumMatsubaraEnergies() const {
  if (m_numMatsubaraEnergies > 0) {
    // If it is larger than zero, then it has been set by the user.
    return m_numMatsubaraEnergies;
  } else {
    // Compute it from the cutoff and temperature instead.
    // TODO(niclas): Shouldn't it be n = (cutoff / (pi * T) - 1 ) / 2 ?
    const T ratio = m_energyCutoff / m_temperature;
    return (static_cast<int>(ratio) + 2);
  }
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
  if (inEnergyMax == 0.0 || inEnergyMax == inEnergyMin) {
    m_numEnergies = 1;
  } else {
    m_numEnergies = inNumEnergies;
  }
}

template <typename T>
void Parameters<T>::setGrainWidthCoherenceLengths(
    const T inGrainWidthCoherenceLengths) {
  // Give grain width in coherence lengths: hbar*vF/(2*pi*kB*Tc). The scaling
  // to internal length units hbar*vF/(kB*Tc) is appropriately done when
  // calculating the size of the discrete lattice spacing.
  m_grainWidthCoherenceLengths = inGrainWidthCoherenceLengths;
  setGridElementSize();
}

template <typename T>
void Parameters<T>::setGridResolution(const int inNumGridBase) {
  m_numGridBase = inNumGridBase;
  m_numGridCoherence = computeGridResolutionRotated(inNumGridBase);

  // Set new grid element width.
  setGridElementSize();
}

// Calculate the lattice spacing size in physical units of hbar*vF/(kB*Tc)
template <typename T> void Parameters<T>::setGridElementSize() {
  m_gridElementSize = m_grainWidthCoherenceLengths /
                      ((static_cast<T>(m_numGridBase) - static_cast<T>(1.0)) *
                       (static_cast<T>(2.0 * M_PI) * qcs::GRAIN_SIZE_FRACTION));
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
int Parameters<T>::computeGridResolutionRotated(const int inNumGridBase) {
  // Check if odd.
  const int baseIsOdd = inNumGridBase % 2;

  // Compute the minimum size when rotated 45 degrees.
  // Note, use ceil because casting to int will floor.
  const int minNumGridRotated = static_cast<int>(
      ceil(sqrt(static_cast<T>(2.0)) * static_cast<T>(inNumGridBase)));

  // Check if odd.
  const int rotatedIsOdd = minNumGridRotated % 2;

  // If both are not odd/equal add one.
  const int numGridRotated =
      (baseIsOdd == rotatedIsOdd) ? minNumGridRotated : minNumGridRotated + 1;

  return numGridRotated;
}

template <typename T> void Parameters<T>::print() const {
  std::cout << "Grain Resolution Base: " << m_numGridBase << std::endl;
  std::cout << "Grain Width: " << m_grainWidthCoherenceLengths << std::endl;
  std::cout << "Grain Element Size: " << m_gridElementSize << std::endl;
  std::cout << "Angular Resolution: " << m_numAngles << std::endl;
  std::cout << "Temperature: " << m_temperature << std::endl;
  std::cout << "Magnetic Field Strength: " << m_magneticFieldStrength
            << std::endl;
  std::cout << "Spectral energy range: (" << m_energyMin << ", " << m_energyMax
            << ")" << std::endl;
  std::cout << "Spectral energy levels: " << m_numEnergies << std::endl;
  std::cout << "Convergence Criterion: " << m_convergenceCriterion << std::endl;
  std::cout << "Cutoff parameter: " << m_energyCutoff << std::endl;
}

#endif // PARAMETERS_H_
