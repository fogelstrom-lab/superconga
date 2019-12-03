#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <ContextModule.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>

template <typename, typename> class GridData;

// TODO(niclas): Move old into this file instead, if it is used.
#include <Parameters_old.h>

template <typename T> class Parameters : public ContextModule<T> {
public:
  Parameters() : _version(2) {}
  ~Parameters() {}

  Parameters(Context<T> *ctx);

  Parameters &operator=(const Parameters &right);

  template <typename T2> Parameters<T> &operator=(const Parameters<T2> &right);

  template <typename T2> Parameters<T> &operatorEq(const Parameters<T2> &right);

  template <typename T2> operator Parameters<T2>() const;

  virtual int initialize() { return 0; }

  // Getters
  int getGridResolutionBase() const { return _N_grid_base; }
  int getGridResolutionCoherence() const { return _N_grid_coherence; }
  int getGridResolutionDeltaRotated() const { return _N_grid_coherence; }

  T getGridWidth() const { return _gridWidth; }
  T getGridElementSize() const { return _gridElementSize; }
  int getAngularResolution() const { return _angularResolution; }
  T getAngleIncrement() const;

  int getNumFrequencies() const;
  int getNumMatsubaraFrequencies() const { return getNumFrequencies(); }
  T getTemperature() const { return _T_Tc; }
  T getMagneticFieldStrength() const { return _magneticField; }

  T getConvergenceCriterion() const { return _convergence; }

  T getCutOff() const { return _cutOff; }

  T getEnergyMin() const { return _energyMin; }
  T getEnergyMax() const { return _energyMax; }
  int getNumEnergies() const { return _numEnergies; }
  T getSmearingFactor() const { return _smearingFactor; }

  T getPenetrationDepth() const { return _param2; }

  int isValid() const { return _isValid; }

  // Setters
  void setGridWidth(T gridWidth);
  void setGridResolution(const int N_grid_base);
  void setAngularResolution(int angularResolution);

  void setCutOff(T cutOff);
  void setNumMatsubaraFrequencies(const int Ne) { _param1 = Ne; }
  void setNumFrequencies(const int Ne) { _param1 = Ne; }
  void setTemperature(T T_Tc);
  void setMagneticFieldStrength(T magneticField);

  void setEnergyRange(T e_min, T e_max, int numEnergies,
                      T smearingFactor = 0.01);
  void setNumEnergies(int numEnergies) { _numEnergies = numEnergies; }

  void setConvergenceCriterion(T convergence);

  void setPenetrationDepth(T penetrationDepth) { _param2 = penetrationDepth; }

  void setDebug();

  void setValid() { _isValid = 1; }
  void setInvalid() { _isValid = 0; }

  // File I/O
  int read(std::string filename);
  int read(std::ifstream &file, const int version);
  int read(std::string filename, const int version);
  int write(std::string filename) const;

  // Helpers
  static int calcGridResolutionRotated(const int N_base);
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
  int _numEnergies;       // Number of spectral energies

  // Physical properties
  T _cutOff;
  T _T_Tc; // T/T_c
  T _magneticField;
  T _magneticFlux;

  // Spare parameters
  T _param1, _param2, _param3, _param4, _param5;

  //	int		_signEnergy;

  T _energyMin; // Lowest energy to compute LDOS for
  T _energyMax; // Highest    "     "
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
Parameters<T>::Parameters(Context<T> *ctx)
    : _gridWidth(1.0), _angularResolution(0), _cutOff(10), _T_Tc(0.5),
      _magneticField(0.0), _convergence(0.0001), _numEnergies(0),
      _energyMin(0.0), _energyMax(0.0), _smearingFactor(0.01), _param1(0.0),
      _param2(100.0), _param3(0.0), _param4(0.0), _param5(0.0), _version(2),
      _isValid(1) {
  ContextModule<T>::setContext(ctx);
  ctx->set(this);

  setGridResolution(101);
}

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

// Is this necessary?
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
  _numEnergies = right.getNumEnergies();
  _energyMin = right.getEnergyMin();
  _energyMax = right.getEnergyMax();
  _smearingFactor = right.getSmearingFactor();
  _isValid = right.isValid();

  _param1 = right._param1;
  _param2 = right._param2;

  setGridResolution(right.getGridResolutionBase());

  return *this;
}

template <typename T> int Parameters<T>::getNumFrequencies() const {
  if (_param1 != 0.0)
    return _param1;
  else
    return (_cutOff / _T_Tc + 2);
}

template <typename T>
void Parameters<T>::setEnergyRange(T e_min, T e_max, int numEnergies,
                                   T smearingFactor) {
  _energyMin = e_min;
  _energyMax = e_max;
  _smearingFactor = smearingFactor;

  if (e_max == 0.0 || e_max == e_min)
    _numEnergies = 1;
  else
    _numEnergies = numEnergies;

  if (numEnergies == 0)
    _numEnergies = 0;
}

template <typename T> void Parameters<T>::setGridWidth(T gridWidth) {
  _gridWidth = gridWidth;
  setGridElementSize();
}

template <typename T>
void Parameters<T>::setGridResolution(const int N_grid_base) {
  _N_grid_base = N_grid_base;
  _N_grid_coherence = calcGridResolutionRotated(N_grid_base);

  // Set new grid element width
  setGridElementSize();
}

template <typename T> void Parameters<T>::setGridElementSize() {
  _gridElementSize = _gridWidth / ((T)_N_grid_base - 1);
}

template <typename T>
void Parameters<T>::setAngularResolution(int angularResolution) {
  _angularResolution = angularResolution;
}

template <typename T> void Parameters<T>::setCutOff(T cutOff) {
  _cutOff = cutOff;
}

template <typename T> void Parameters<T>::setTemperature(T T_Tc) {
  _T_Tc = T_Tc;
}

template <typename T>
void Parameters<T>::setMagneticFieldStrength(T magneticField) {
  _magneticField = magneticField;
}

template <typename T>
void Parameters<T>::setConvergenceCriterion(T convergence) {
  _convergence = convergence;
}

template <typename T> void Parameters<T>::setDebug() {}

template <typename T> T Parameters<T>::getAngleIncrement() const {
  return 2.0 * M_PI / (T)_angularResolution;
}

template <typename T> int Parameters<T>::read(std::string filename) {
  std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
  int res = read(file, _version);
  file.close();

  return res;
}

template <typename T>
int Parameters<T>::read(std::string filename, const int version) {
  int res;

  if (version == _version)
    return read(filename);
  else {
    std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);
    res = read(file, version);
    file.close();
    return res;
  }
}

template <typename T>
int Parameters<T>::read(std::ifstream &file, const int version) {
  if (file.is_open()) {
    if (version == _version) {
      int numBytes = sizeof(*this);

      file.read((char *)this, numBytes);

      return 0;
    } else {
      v1::Parameters<T> pr;
      int numBytes = sizeof(pr);

      file.read((char *)&pr, numBytes);
      pr.setInvalid();

      setGridResolution(pr.getGridResolutionBase());
      setGridWidth(pr.getGridWidth());
      setAngularResolution(pr.getAngularResolution());
      setCutOff(pr.getCutOff());
      setMagneticFieldStrength(pr.getMagneticFieldStrength());
      setTemperature(pr.getTemperature());
      setConvergenceCriterion(pr.getConvergenceCriterion());

      T e_min = pr.getLDOSEnergyMin();
      T e_max = pr.getLDOSEnergyMax();
      int numEnergies = pr.getNumEnergies();

      if (e_min != e_max && numEnergies != pr.getNumMatsubaraFrequencies())
        setEnergyRange(pr.getLDOSEnergyMin(), pr.getLDOSEnergyMax(),
                       pr.getSmearingFactor(), numEnergies);
      else
        setEnergyRange(0.0, 0.0, 0.01, 0);
      return 0;
    }
  } else {
    std::cout << "Could not load file, version: " << version << "\n";
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

template <typename T>
int Parameters<T>::calcGridResolutionRotated(const int N_grid_base) {
  int baseIsOdd = N_grid_base % 2;

  // Set N_grid_rotated
  int N_grid_rotated =
      sqrt(2.0) * N_grid_base + 1; // Add 1 to make sure rotation fits inside

  int gammaIsOdd = N_grid_rotated % 2;

  if (baseIsOdd != gammaIsOdd)
    N_grid_rotated += 1;

  return N_grid_rotated;
}

template <typename T> void Parameters<T>::print() const {
  std::cout << "Grid Resolution Base: " << _N_grid_base << std::endl;
  std::cout << "Grid Width: " << _gridWidth << std::endl;
  std::cout << "Grid Element Size: " << _gridElementSize << std::endl;
  std::cout << "Angular Resolution: " << _angularResolution << std::endl;
  std::cout << "Temperature: " << _T_Tc << std::endl;
  std::cout << "Magnetic Field Strength: " << _magneticField << std::endl;
  std::cout << "Spectral energy range: (" << _energyMin << ", " << _energyMax
            << ")" << std::endl;
  std::cout << "Spectral energy levels: " << _numEnergies << std::endl;
  std::cout << "Convergence Criterion: " << _convergence << std::endl;
  std::cout << "Cutoff parameter: " << _cutOff << std::endl;
}

#endif // PARAMETERS_H
