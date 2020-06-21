#ifndef UTIL_RANDOM_ARRAY_H
#define UTIL_RANDOM_ARRAY_H

#include <vector>
#include <random>

namespace util {

std::mt19937 create_random_engine();

template<typename T>
std::vector<T> create_random_array(const size_t size, const T rand_min, const T rand_max);

} // namespace util

#endif // UTIL_RANDOM_ARRAY_H
