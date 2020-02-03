//
// Enums to make code and function arguments read more easily.
//
#ifndef TYPE_H_
#define TYPE_H_

struct Type {
  enum type {
    real = 1,
    imag = 2,
    complex = 3,
    modulus = 4,
    all = 5,
    add = 6,
    remove = 7
  };
  enum location { start = 1, begin = 1, end = 2 };
  enum coherence { singlet = 1, triplet = 4 };
};

#endif // TYPE_H_
