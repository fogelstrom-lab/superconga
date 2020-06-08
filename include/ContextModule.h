#ifndef CONTEXT_MODULE_H_
#define CONTEXT_MODULE_H_

#include <cuda.h>
#include <iostream>
#include <string>

namespace scg {
template <typename> class ComputeProperty;
template <typename> class Context;
template <typename> class FermiSurface;
template <typename> class GeometryGroup;
template <typename> class IntegrationIterator;
template <typename> class OrderParameter;
template <typename> class Parameters;
template <typename> class RiccatiSolver;

template <typename T> class ContextModule {
public:
  ContextModule() : _context(0) {}
  virtual ~ContextModule() {}

  // Copy constructor
  ContextModule(const ContextModule<T> &other) : _context(other._context) {}

  // Context constructor
  // Makes sure that Context knows about this ContextModule, and vice versa.
  ContextModule(Context<T> *ctx) { _context = ctx; }

  void setContext(Context<T> *ctx) { _context = ctx; }
  Context<T> *getContext() const { return _context; }

  // Helpers
  Parameters<T> *getParameters() const { return _context->getParameters(); }

  GeometryGroup<T> *getGeometry() const { return _context->getGeometry(); }
  FermiSurface<T> *getFermiSurface() const {
    return _context->getFermiSurface();
  }
  OrderParameter<T> *getOrderParameter() const {
    return _context->getOrderParameter();
  }
  RiccatiSolver<T> *getRiccatiSolver() const {
    return _context->getRiccatiSolver();
  }
  IntegrationIterator<T> *getIntegrationIterator() const {
    return _context->getIntegrationIterator();
  }

  int getNumComputeObjects() const { return _context->getNumComputeObjects(); }
  ComputeProperty<T> *getComputeObject(int idx) const {
    return _context->getComputeObject(idx);
  }

  // The function Context::initialize() calls ContextModule::initialize() for
  // all context modules. Call Context::initialize() after changing anything in
  // Parameters object.
  virtual int initialize() = 0;

  static void checkCudaError(std::string msg = "");

private:
  Context<T> *_context;
};

template <typename T> void ContextModule<T>::checkCudaError(std::string msg) {
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
    std::cout << "CUDA ERROR: " << msg << ": " << cudaGetErrorString(err)
              << std::endl;
}
} // namespace scg

#endif // CONTEXT_MODULE_H_
