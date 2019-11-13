#ifndef GRADIENTACCELERATOR_H
#define GRADIENTACCELERATOR_H

#include <Grid.h>

template <typename T> class GradientAccelerator : public ContextModule<T> {
  typedef typename Grid<T>::GPU GridGPU;

public:
  GradientAccelerator();
  ~GradientAccelerator();

  GradientAccelerator(Context<T> *ctx);

  virtual int initialize();

  void reset();
  int step(T residual, GridGPU *grid);

  int isPreConverged() const { return _criterionMet; }
  int isConverged() const {
    return int(_criterionMet && _relaxFinalCount >= _relaxFinal &&
               _numLoops >= _minIter);
  }

  void resetCount() { _acount = 0; }

  // Setters
  void setMinIterations(int minIter) { _minIter = minIter; }
  void setMaxIterations(int maxIter) { _maxIter = maxIter; }
  void setRelaxCount(int relax) { _relax = relax; }
  void setRelaxFinal(int relax) { _relaxFinal = relax; }
  void setPreConverged(int is_converged) { _criterionMet = is_converged; }

private:
  int _minIter; // Minimum # of iterations at start before applying acceleration
  int _maxIter; // Maximum # of iterations until applying _relaxFinal
                // iterations, then report convergence
  int _relax;   // Iterations between applying gradient acceleration
  int _relaxFinal; // Iterations after convergence reached

  int _acount;          // Iteration count after latest acceleration
  int _relaxFinalCount; // Relaxation iteration count after convergence
  int _numLoops;        // Total # of iterations
  int _doAccel;         // Acceleration flag
  int _criterionMet;    // Is convergence criterion reached?

  T _residual;        // Residual current iteration
  T _residualLast;    // Residual last iteration
  T _convergenceRate; // Fractional change of residual
  T _rate[2];         // Stored convergence fractional rates

  GridGPU *_g1; // Stored grids from two iterations, to compute gradient
  GridGPU *_g2;
};

template <typename T>
GradientAccelerator<T>::GradientAccelerator()
    : ContextModule<T>(), _g1(0), _g2(0) {}

template <typename T>
GradientAccelerator<T>::GradientAccelerator(Context<T> *ctx)
    : ContextModule<T>(ctx), _g1(0), _g2(0), _relax(10), _relaxFinal(10),
      _relaxFinalCount(0), _doAccel(0), _criterionMet(0), _minIter(20),
      _maxIter(-1), _residualLast(1.0) {
  ctx->set(this);
}

template <typename T> GradientAccelerator<T>::~GradientAccelerator() {
  std::cout << "~GradientAccelerator()\n";
  if (_g1)
    delete _g1;
  if (_g2)
    delete _g2;
}

template <typename T> int GradientAccelerator<T>::initialize() {
  int N0 = ContextModule<T>::getParameters()->getGridResolutionBase();

  GridGPU::initializeGrid(_g1, N0, N0);
  GridGPU::initializeGrid(_g2, N0, N0);

  reset();
  return 0;
}

template <typename T> void GradientAccelerator<T>::reset() {
  _acount = 0;
  _criterionMet = 0;
  _numLoops = 0;
  _relaxFinalCount = 0;
}

template <typename T>
int GradientAccelerator<T>::step(T residual, GridGPU *grid) {
  typedef ContextModule<T> C;
  Parameters<T> *param = C::getParameters();

  _residualLast = _residual;
  _residual = residual;

  _convergenceRate = _residualLast / _residual;

  if (_acount < _relax || (_acount < _relaxFinal && _criterionMet))
    _acount++;

  if ((_acount > 5 && _residual < param->getConvergenceCriterion()) ||
      _criterionMet || (_maxIter != -1 && _numLoops > _maxIter)) {
    _criterionMet = 1;
    _relaxFinalCount++;
  }

  if (_convergenceRate < 1.5 && abs(_convergenceRate - 1.0) > 0.0001 &&
      _numLoops > _minIter && _acount >= _relax && !_criterionMet)
    _doAccel = 1;
  else
    _doAccel = 0;

  if (abs(_convergenceRate - 1.0) < 0.000001)
    _acount = 0;

  std::cout << "Loop: " << _numLoops << ", Criterion met: " << _criterionMet
            << ", Rate: " << _convergenceRate << ", acount: " << _acount
            << ", Accel: " << _doAccel << std::endl;

  if (_doAccel) {
    if (_acount == 0 + _relax) // Record iteration #1
    {
      std::cout << "----->Recording iteration #1\n";
      *_g1 = *grid;
      _rate[0] = _convergenceRate;
      _acount++;
    } else if (_acount == 1 + _relax) // Record iteration #2
    {
      std::cout << "----->Recording iteration #2\n";
      *_g2 = *grid;
      _rate[1] = _convergenceRate;
      _acount++;
    } else if (_acount == 2 + _relax) // Find iteration gradient and apply to
                                      // order parameter
    {
      *_g2 -= *_g1;
      T avgRate = _rate[1];
      T factor = 0.25 / abs(avgRate - 1.0);
      if (factor > 100.0)
        factor = 100.0;
      *_g2 *= factor;
      *grid += *_g2;

      std::cout << "----->Applying gradient, factor = " << factor << std::endl;

      _acount = 0;
    }
  }

  _numLoops++;

  return (_criterionMet && _relaxFinal >= 20);
}

#endif // GRADIENTACCELERATOR_H
