#ifndef COLORMAP_H
#define COLORMAP_H

#include <algorithm> // std::swap

namespace Color {
struct Base {
  Base();
  ~Base();

  Base(const Base &other);

  Base(int numCol);

  Base &operator=(Base other) {
    swap(*this, other);
    return *this;
  }

  void setColors(float *r, float *g, float *b);
  void getColor(float param, float &r, float &g, float &b) const;

  friend void swap(Base &A, Base &B) {
    std::swap(A._numCol, B._numCol);
    std::swap(A._r, B._r);
    std::swap(A._g, B._g);
    std::swap(A._b, B._b);
  }
  int _numCol;
  float *_r, *_g, *_b;
};

Base::Base() : _r(0), _g(0), _b(0) {}

Base::Base(const Base &other) {
  _numCol = other._numCol;
  _r = new float[_numCol];
  _g = new float[_numCol];
  _b = new float[_numCol];

  setColors(other._r, other._g, other._b);
}

Base::Base(int numCol) {
  _numCol = numCol;
  _r = new float[_numCol];
  _g = new float[_numCol];
  _b = new float[_numCol];
}

Base::~Base() {
  if (_r)
    delete[] _r;
  if (_g)
    delete[] _g;
  if (_b)
    delete[] _b;
}

void Base::setColors(float *r, float *g, float *b) {
  for (int i = 0; i < _numCol; ++i) {
    _r[i] = r[i];
    _g[i] = g[i];
    _b[i] = b[i];
  }
}

inline void Base::getColor(float param, float &r, float &g, float &b) const {
  // Clamp value
  if (param < 0.0f)
    param = 0.0f;
  else if (param > 1.0f)
    param = 1.0f;

  float paramIdx = param * (_numCol - 1);

  int idx_low = paramIdx;
  int idx_high = idx_low + (int)(idx_low < (_numCol - 1));

  float high_frac = (paramIdx - (float)idx_low);
  float low_frac = 1.0f - high_frac;

  r = _r[idx_low] * low_frac + _r[idx_high] * high_frac;
  g = _g[idx_low] * low_frac + _g[idx_high] * high_frac;
  b = _b[idx_low] * low_frac + _b[idx_high] * high_frac;
}

struct Jet : Base {
  Jet();
};

Jet::Jet() : Base(6) {
  float r[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.7};
  float g[6] = {0.0, 0.0, 1.0, 1.0, 0.0, 0.6};
  float b[6] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0};

  setColors(r, g, b);
}

struct RedBlue : Base {
  RedBlue();
};

RedBlue::RedBlue() : Base(9) {
  float r[9] = {0.6, 0.2, 0.1, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
  float g[9] = {0.9, 0.6, 0.4, 0.0, 0.0, 0.0, 0.4, 0.6, 0.9};
  float b[9] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.1, 0.2, 0.6};

  setColors(r, g, b);
}

struct RedBlueWhite : Base {
  RedBlueWhite();
};

RedBlueWhite::RedBlueWhite() : Base(3) {
  float r[3] = {0.9, 1.0, 0.0};
  float g[3] = {0.0, 1.0, 0.0};
  float b[3] = {0.0, 1.0, 0.9};

  setColors(r, g, b);
}

struct Copper : Base {
  Copper();
};

Copper::Copper() : Base(5) {
  float r[5] = {0.3, 0.8, 1.0, 1.0, 1.0};
  float g[5] = {0.3, 0.2, 0.2, 0.7, 1.0};
  float b[5] = {0.0, 0.2, 0.6, 0.9, 1.0};

  setColors(r, g, b);
}

} // namespace Color

#endif // COLORMAP_H
