#ifndef NLOHMANN_JSON_WRAPPER_H_
#define NLOHMANN_JSON_WRAPPER_H_

// For some reason the nlohmann/json lib has annoying warnings that clutter our
// compiler output. So lets try to turn them off.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"

#include <nlohmann/json.hpp>

// Turn warnings back on again.
#pragma GCC diagnostic pop

#endif // NLOHMANN_JSON_WRAPPER_H_
