#ifndef HYSSOP_UTIL_H
#define HYSSOP_UTIL_H

#include <Eigen/Dense>

// Why can’t I separate the definition of my templates class from its declaration and put it inside a .cpp file?
// https://isocpp.org/wiki/faq/templates#templates-defn-vs-decl

template <typename T>
bool MatrixEqVector(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m, std::vector<std::vector<T>> v)  {
    for (size_t i = 0; i < m.rows(); i++) {
        for (size_t j = 0; j < m.cols(); j++) {
            if (m(i,j) != v[i][j]) {
                return false;
            };
        }
    }
    return true;
}

#endif // HYSSOP_UTILITIES_H