#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "util.h"

TEST(MatrixArithmetic, Addition) {
    
    Eigen::MatrixXi a{{1, 2}, {3, 4}};
    Eigen::MatrixXi b{{1, 2}, {3, 4}};

    std::vector<std::vector<int>> v{{2, 4}, {6, 8}};

    Eigen::MatrixXi m(2,2);
    
    m = a + b;

    EXPECT_TRUE(MatrixEqVector<int>(m, v));

}

TEST(MatrixArithmetic, AdditionInPlace) {
    
    Eigen::MatrixXi a{{1, 2}, {3, 4}};
    Eigen::MatrixXi b{{1, 2}, {3, 4}};

    std::vector<std::vector<int>> v{{2, 4}, {6, 8}};

    a = a + b;
    
    EXPECT_TRUE(MatrixEqVector<int>(a, v));

}

TEST(MatrixArithmetic, Substraction) {

    Eigen::MatrixXi a{{3, 3}, {3, 3}};
    Eigen::MatrixXi b{{2, 2}, {2, 2}};

    std::vector<std::vector<int>> v{{1, 1}, {1, 1}};

    Eigen::MatrixXi m = a - b;

    EXPECT_TRUE(MatrixEqVector<int>(m, v));

}

TEST(MatrixArithmetic, SubstractionInPlace) {

    Eigen::MatrixXi a{{3, 3}, {3, 3}};
    Eigen::MatrixXi b{{2, 2}, {2, 2}};

    std::vector<std::vector<int>> v{{1, 1}, {1, 1}};

    a = a - b;

    EXPECT_TRUE(MatrixEqVector<int>(a, v));

}

TEST(MatrixArithmetic, Multiplication) {

    Eigen::MatrixXi a{{1, 2}, {2, 1}};

    std::vector<std::vector<int>> v{{2, 4}, {4, 2}};

    Eigen::MatrixXi b = a * 2;

    EXPECT_TRUE(MatrixEqVector<int>(b, v));

}

TEST(MatrixArithmetic, MultiplicationInPlace) {

    Eigen::MatrixXi a{{1, 2}, {2, 1}};

    std::vector<std::vector<int>> v{{2, 4}, {4, 2}};

    a = a * 2;

    EXPECT_TRUE(MatrixEqVector<int>(a, v));

}

TEST(MatrixArithmetic, CompoundAddition) {

    Eigen::MatrixXi a{{1, 1}, {1, 1}};

    std::vector<std::vector<int>> v {{2, 2}, {2, 2}};

    a += a;

    EXPECT_TRUE(MatrixEqVector(a, v));

}

TEST(MatrixArithmetic, CompoundSubstraction) {

    Eigen::MatrixXi a{{1, 1}, {1, 1}};

    std::vector<std::vector<int>> v {{0, 0}, {0, 0}};

    a -= a;

    EXPECT_TRUE(MatrixEqVector(a, v));

}

TEST(MatrixArithmetic, Transpose) {

    Eigen::MatrixXi a{{1, 1}, {2, 2}};

    std::vector<std::vector<int>> v {{1, 2}, {1, 2}};

    Eigen::MatrixXi b = a.transpose();

    EXPECT_TRUE(MatrixEqVector(b, v));

}

TEST(MatrixArithmetic, TransposeInPlace) {

    Eigen::MatrixXi a{{1, 1}, {2, 2}};

    std::vector<std::vector<int>> v {{1, 2}, {1, 2}};

    a.transposeInPlace();

    EXPECT_TRUE(MatrixEqVector(a, v));

}

TEST(MatrixArithmetic, Conjugate) {

    Eigen::Matrix<std::complex<int>, -1, -1> a{{std::complex<int>(1, 1), std::complex<int>(1, 1)}, {std::complex<int>(2, 2), std::complex<int>(2, 2)}};

    std::vector<std::vector<std::complex<int>>> v {{std::complex<int>(1, -1), std::complex<int>(1, -1)}, {std::complex<int>(2, -2), std::complex<int>(2, -2)}};

    Eigen::Matrix<std::complex<int>, -1, -1> b = a.conjugate();

    EXPECT_TRUE(MatrixEqVector<std::complex<int>>(b, v));

}

TEST(MatrixArithmetic, MatrixVectorMultiply) {

    Eigen::Matrix3i m{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};

    Eigen::Vector3i v{2, 2, 2};

    std::vector<int> r{6, 12, 18};

    v = m * v;

    EXPECT_TRUE(VectorEqVector<int>(v, r));
}

TEST(MatrixArithmetic, MatrixMatrixMultiply) {

    Eigen::Matrix2f m{{1.0, 2.0}, {2.0, 1.0}};
    Eigen::Matrix2f n{{1.0, 0.0}, {0.0, 1.0}};

    std::vector<std::vector<float>> v{{1.0, 2.0}, {2.0, 1.0}};

    m = m * n;

    EXPECT_TRUE(MatrixEqVector<float>(m, v));

}

TEST(MatrixArithmetic, DotProduct) {

    Eigen::Vector3f v{1.0, 2.0, 3.0};
    Eigen::Vector3f u{3.0, 2.0, 1.0};

    float dot_product_expected {10.0};

    auto dot_product_actual = v.dot(u);

    EXPECT_TRUE(dot_product_actual == dot_product_expected);

}

TEST(MatrixArithmetic, CrossProduct) {
    
    Eigen::Vector3f v{1.0, 2.0, 3.0};
    Eigen::Vector3f u{1.5, 3.0, 4.5};

    std::vector<float> w{0.0, 0.0, 0.0};

    v = v.cross(u);

    EXPECT_TRUE(VectorEqVector<float>(v, w));

}
