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

template <typename T>
bool VectorEqVector(Eigen::Vector<T, Eigen::Dynamic> ev, std::vector<T> sv)  {
    for (size_t i = 0; i < ev.size(); i++) {
        if (ev(i) != sv[i]) {
            return false;
        };
    }
    return true;
}

template <typename T>
bool ArrayEqVector(Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic> a, std::vector<std::vector<T>> v)  {
    for (size_t i = 0; i < a.rows(); i++) {
        for (size_t j = 0; j < a.cols(); j++) {
            if (a(i,j) != v[i][j]) {
                return false;
            };
        }
    }
    return true;
}

/** Compare Eigen matrices. Use with EXPECT_PRED2 */
inline bool MatricesApproxXXi(const Eigen::MatrixXi &expected,
                           const Eigen::MatrixXi &actual) {
    return expected.isApprox(actual);
}

#endif // HYSSOP_UTILITIES_H