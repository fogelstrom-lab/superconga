#ifndef PARAMETERSOLD_H
#define PARAMETERSOLD_H

#include <string>

#include "Parameters.h"
#include <Grid.h>
#include <math.h>

#ifndef PI
#define PI 3.1415926535897932384
#endif

template <typename, typename> class GridData;

namespace v1 {
template <typename T> class Parameters {
public:
  Parameters() : _version(1), _energies(0) {}
  ~Parameters() {
    if (_energies && _isValid)
      delete _energies;
  }

  Parameters<T> &operator=(const Parameters<T> &right);

  template <typename T2> Parameters<T> &operator=(const Parameters<T2> &right);

  template <typename T2> Parameters<T> &operatorEq(const Parameters<T2> &right);

  template <typename T2> operator Parameters<T2>() const;

  // Getters
  int getGridResolutionBase() const { return _N_grid_base; }
  int getGridResolutionCoherence() const { return _N_grid_coherence; }
  int getGridResolutionBoundary() const { return _N_grid_boundary; }
  int getGridResolutionDeltaRotated() const { return _N_grid_coherence; }

  T getGridWidth() const { return _gridWidth; }
  T getGridElementSize() const { return _gridElementSize; }
  int getAngularResolution() const { return _angularResolution; }
  T getAngleIncrement() const;

  int getNumMatsubaraFrequencies() const { return _numFrequencies; }
  T getTemperature() const { return _T_Tc; }
  T getMagneticFieldStrength() const { return _magneticField; }

  T getConvergenceCriterion() const { return _convergence; }

  T getCutOff() const { return _cutOff; }
  int getNumEnergies() const { return _numEnergies; }

  T getLDOSEnergyMin() const { return _LDOS_energyMin; }
  T getLDOSEnergyMax() const { return _LDOS_energyMax; }
  T getSmearingFactor() const { return _smearingFactor; }

  typename Grid<T>::CPU *getEnergies() const { return _energies; }

  int isValid() const { return _isValid; }

  // Setters
  void setGridWidth(T gridWidth);
  void setGridResolution(int N_grid_base);
  void setAngularResolution(int angularResolution);

  void setCutOff(T cutOff);
  void setTemperature(T T_Tc);
  void setMagneticFieldStrength(T magneticField);

  void setSignEnergy(int sign) { _signEnergy = sign; }
  void setLDOSEnergies(T e_min, T e_max = 0.0, T smearingFactor = 0.01,
                       int numEnergies = 1);
  void incrementLDOSEnergyIndex(const int idx_inc = 1);

  void setConvergenceCriterion(T convergence);

  void setDebug();

  void setValid() { _isValid = 1; }
  void setInvalid() { _isValid = 0; }

  // File I/O
  int read(std::string filename);
  int write(std::string filename) const;

  // Helpers
  void setMatsubaraFrequencies();
  void setLDOSEnergies();
  void initEnergies();
  int getVersion() const { return _version; }
  void print() const;

private:
  // Discrete properties
  int _N_grid_boundary;  // Number of elements for boundary/initial grid
  int _N_grid_coherence; // Number of elements for coherence functions
                         // (gamma,gammabar) grid
  int _N_grid_base;      // Number of elements for order parameter solution

  T _gridWidth;       // In natural units: h_bar * vF / (kB*Tc)
  T _gridElementSize; // Width of one grid element

  int _angularResolution; // Number of angles to solve for
  int _numFrequencies;    // Number of Matsubara frequencies

  int _numEnergies; // Number of energies (Matsubara or Real)

  typename Grid<T>::CPU *_energies;

  // Physical properties
  T _cutOff;
  T _T_Tc; // T/T_c
  T _magneticField;
  T _magneticFlux;

  // Spare parameters
  T _param1, _param2, _param3, _param4, _param5;

  int _signEnergy;

  T _LDOS_energyMin; // Lowest energy to compute LDOS for
  T _LDOS_energyMax; // Highest    "     "
  T _smearingFactor;

  // Numerical properties
  T _convergence;

  // Other
  const int _version;
  int _isValid;

  // Helper functions
  void setGridElementSize();
};

template <typename T>
Parameters<T> &Parameters<T>::operator=(const Parameters<T> &right) {
  operatorEq(right);
  return *this;
}

template <typename T>
template <typename T2>
Parameters<T> &Parameters<T>::operator=(const Parameters<T2> &right) {
  operatorEq(right);
  return *this;
}

template <typename T>
template <typename T2>
Parameters<T>::operator Parameters<T2>() const {
  Parameters<T2> other;
  other = *this;
  return other;
}

template <typename T>
template <typename T2>
Parameters<T> &Parameters<T>::operatorEq(const Parameters<T2> &right) {
  if (_version != right.getVersion())
    std::cout << "Parameters<T>::operator=(): Warning, differing versions of "
                 "Parameters class\n";

  _gridWidth = right.getGridWidth();
  _angularResolution = right.getAngularResolution();
  _cutOff = right.getCutOff();
  _T_Tc = right.getTemperature();
  _magneticField = right.getMagneticFieldStrength();
  _convergence = right.getConvergenceCriterion();
  _numFrequencies = right.getNumMatsubaraFrequencies();
  _numEnergies = right.getNumEnergies();
  _LDOS_energyMin = right.getLDOSEnergyMin();
  _LDOS_energyMax = right.getLDOSEnergyMax();
  _smearingFactor = right.getSmearingFactor();
  _isValid = right.isValid();

  setGridResolution(right.getGridResolutionBase());
  initEnergies();

  return *this;
}

template <typename T>
void Parameters<T>::setLDOSEnergies(T e_min, T e_max, T smearingFactor,
                                    int numEnergies) {
  _LDOS_energyMin = e_min;
  _LDOS_energyMax = e_max;
  _smearingFactor = smearingFactor;

  if (e_max == 0.0 || e_max == e_min)
    _numEnergies = 1;
  else
    _numEnergies = numEnergies;

  setLDOSEnergies();
}

template <typename T> void Parameters<T>::setLDOSEnergies() {
  T LDOS_energyIncrement;

  if (_numEnergies > 1)
    LDOS_energyIncrement =
        (_LDOS_energyMax - _LDOS_energyMin) / (T)(_numEnergies - 1);
  else
    LDOS_energyIncrement = 0.0;

  // Enter explicit energy values into array
  if (_energies)
    delete _energies;

  _energies = new typename Grid<T>::CPU(_numEnergies, 1);

  T *energies_re = _energies->getDataPointer(0, Type::real);
  T *energies_im = _energies->getDataPointer(0, Type::imag);

  for (int i = 0; i < _numEnergies; ++i) {
    energies_re[i] = _LDOS_energyMin + i * LDOS_energyIncrement;
    energies_im[i] = _smearingFactor;
  }
}

template <typename T> void Parameters<T>::setGridWidth(T gridWidth) {
  _gridWidth = gridWidth;
  setGridElementSize();
}

template <typename T> void Parameters<T>::setGridResolution(int N_grid_base) {
  _N_grid_base = N_grid_base;
  int baseIsOdd = _N_grid_base % 2;

  // Set N_grid_coherence
  _N_grid_coherence =
      sqrt(2.0) * _N_grid_base + 1; // Add 1 to make sure rotation fits inside

  int gammaIsOdd = _N_grid_coherence % 2;

  if (baseIsOdd != gammaIsOdd)
    _N_grid_coherence += 1;

  // Set N_grid_boundary
  _N_grid_boundary = sqrt(2.0) * _N_grid_coherence +
                     1; // Add 1 to make sure rotation fits inside

  int boundaryIsOdd = _N_grid_boundary % 2;

  if (baseIsOdd != boundaryIsOdd)
    _N_grid_boundary += 1;

  // Set new grid element width
  setGridElementSize();
}

template <typename T> void Parameters<T>::setGridElementSize() {
  _gridElementSize = _gridWidth / ((T)_N_grid_base - 1);
}

template <typename T> void Parameters<T>::setMatsubaraFrequencies() {
  int mult = 1 + (int)(_signEnergy == 0);

  _numFrequencies = _cutOff / _T_Tc + 2;
  _numEnergies = mult * _numFrequencies;

  // Enter explicit energy value into array
  if (_energies)
    delete _energies;

  _energies = new typename Grid<T>::CPU(_numEnergies, 1);

  T *energies_re = _energies->getDataPointer(0, Type::real);
  T *energies_im = _energies->getDataPointer(0, Type::imag);

  if (_signEnergy == 0) {
    for (int i = 0; i < _numEnergies; ++i) {
      energies_re[i] = 0.0;
      energies_im[i] = (PI * _T_Tc) * (2.0 * (i - _numEnergies / 2) + 1.0);
    }
  } else {
    for (int i = 0; i < _numEnergies; ++i) {
      energies_re[i] = 0.0;
      energies_im[i] = _signEnergy * (PI * _T_Tc) * (2.0 * i + 1.0);
    }
  }
}

template <typename T>
void Parameters<T>::setAngularResolution(int angularResolution) {
  _angularResolution = angularResolution;
}

template <typename T> void Parameters<T>::setCutOff(T cutOff) {
  _cutOff = cutOff;

  if (getNumMatsubaraFrequencies() == getNumEnergies() &&
      getLDOSEnergyMin() == getLDOSEnergyMax())
    setMatsubaraFrequencies();
}

template <typename T> void Parameters<T>::setTemperature(T T_Tc) {
  _T_Tc = T_Tc;

  if (getNumMatsubaraFrequencies() == getNumEnergies() &&
      getLDOSEnergyMin() == getLDOSEnergyMax())
    setMatsubaraFrequencies();
}

template <typename T>
void Parameters<T>::setMagneticFieldStrength(T magneticField) {
  _magneticField = magneticField;
}

template <typename T>
void Parameters<T>::setConvergenceCriterion(T convergence) {
  _convergence = convergence;
}

template <typename T> void Parameters<T>::setDebug() {
  _numFrequencies = 1;
  _numEnergies = 1;
}

template <typename T> T Parameters<T>::getAngleIncrement() const {
  return 2.0 * PI / (T)_angularResolution;
}

template <typename T> void Parameters<T>::initEnergies() {
  _energies = 0;

  if (getNumMatsubaraFrequencies() != getNumEnergies())
    setLDOSEnergies();
  else if (getNumMatsubaraFrequencies() == getNumEnergies() &&
           getLDOSEnergyMin() != getLDOSEnergyMax())
    setLDOSEnergies();
  else
    setMatsubaraFrequencies();
}

template <typename T> int Parameters<T>::read(std::string filename) {
  int numBytes = sizeof(*this);

  std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);

  if (file.is_open()) {
    file.read((char *)this, numBytes);
    file.close();

    initEnergies();
    return 0;
  } else {
    std::cout << "Could not load file\n";
    return 1;
  }
}

template <typename T> int Parameters<T>::write(std::string filename) const {
  int numBytes = sizeof(*this);

  std::ofstream file(filename.c_str(),
                     std::ios::out | std::ios::binary | std::ios::trunc);

  if (file.is_open()) {
    file.write((char *)(this), numBytes);
    file.close();
    return 0;
  } else {
    std::cout << "Could not save file: " << filename << std::endl;
    return 1;
  }
}

template <typename T> void Parameters<T>::print() const {
  std::cout << "Grid Resolution Base: " << _N_grid_base << std::endl;
  std::cout << "Grid Width: " << _gridWidth << std::endl;
  std::cout << "Grid Element Size: " << _gridElementSize << std::endl;
  std::cout << "Angular Resolution: " << _angularResolution << std::endl;
  std::cout << "Matsubara Frequencies: " << _numFrequencies << std::endl;
  std::cout << "Energy levels: " << _numEnergies << std::endl;
  std::cout << "Temperature: " << _T_Tc << std::endl;
  std::cout << "Magnetic Field Strength: " << _magneticField << std::endl;
  std::cout << "LDOS energy range: (" << _LDOS_energyMin << ", "
            << _LDOS_energyMax << ")" << std::endl;
  std::cout << "Convergence Criterion: " << _convergence << std::endl;
  std::cout << "GridData energies: ";
}
} // namespace v1

#endif // PARAMETERSOLD_H
