/** \file type_definition.hpp
 *
 *  \brief Define some common type used in SDDK::
 */

#ifndef __TYPE_DEFINITION_HPP__
#define __TYPE_DEFINITION_HPP__

using double_complex = std::complex<double>;

// define type traits for a single template implementation of both real and complex matrix
// general case for real matrix
template <typename T>
struct Real {using type = T;};

// special case for complex matrix
template <typename T>
struct Real<std::complex<T>> {using type = T;};

template <typename T>
using real_type = typename Real<T>::type;
#endif // __TYPE_DEFINITION_HPP__