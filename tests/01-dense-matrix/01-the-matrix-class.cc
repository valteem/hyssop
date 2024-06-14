#include "gtest/gtest.h"

#include <cmath>

#include <Eigen/Dense>

#include "util.h"

TEST(TheMatrixClass, DefaultConstructorFloatFixedSize) {
    
    Eigen::Matrix3f a;
    
    for (size_t i = 0; i < a.rows(); i++) {
        for (size_t j = 0; j < a.cols(); j++) {
            auto value = a(i, j);
            if ( !std::isnan(value) ) { // NaN as coefficient in the middle of the underlying array
                EXPECT_NE(value, 0.0); // default constructor populates matrix coefficients with arbitrary non-zero values
            };
        };
    };

}

TEST(TheMatrixClass, DefaultConstructorIntFixedSize) {

    Eigen::Matrix3i b;

    for (size_t i = 0; i < b.rows(); i++) {
        for (size_t j = 0; j < b.cols(); j++) {
            auto value = b(i, j);
            EXPECT_NE(value, 0); // default constructor populates matrix coefficients with arbitrary non-zero values
        };
    };

}

TEST(TheMatrixClass, DefaultConstructorIntDynamicSize) {

    Eigen::MatrixXi c;

    // Default constructor for MatrixX*, VectorX* returns an object with empty underlying array
    EXPECT_EQ(c.rows(), 0);
    EXPECT_EQ(c.cols(), 0);

}

TEST(TheMatrixClass, InitWithStdVector) {

    std::vector<int> sv = {0, 1, 2, 3};
    Eigen::Vector4i ev(sv.data());

    // Eigen::Vector4i s(sourceData.data());
    // static assertion failed: YOU_CALLED_A_FIXED_SIZE_METHOD_ON_A_DYNAMIC_SIZE_MATRIX_OR_VECTOR

    for (size_t i = 0; i < ev.size(); i++) {
        int value = ev(i);
        EXPECT_EQ(value, sv[i]);
    }
}

TEST(TheMatrixClass, initMatrixRows) {

    std::vector<std::vector<int>> v = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Eigen::Matrix3i m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    EXPECT_TRUE(MatrixEqVector<int>(m, v)); // need to explicitly set template parameter for fixed-size matrices

}

TEST(TheMatrixClass, InitMatrixWithCommaSeparatedValues) {

    std::vector<std::vector<int>> v = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Eigen::Matrix3i m;

    m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    EXPECT_TRUE(MatrixEqVector<int>(m, v));

}

TEST(TheMatrixClass, ResizeDynamicPredefined) {

    Eigen::MatrixXf m(1, 2);
    
    m.resize(3, 3);

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);

}

TEST(TheMatrixClass, ResizeDynamicEmpty) {

    Eigen::MatrixXf m;
    
    m.resize(3, 3);

    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 3);

}

TEST(TheMatrixClass, ResizeShrink) {

    std::vector<std::vector<int>> v = {{1, 2}, {4, 5}};
    Eigen::MatrixXi m(3,3);

    m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    m.conservativeResize(2,2); // m.resize() re-allocates m_storage

    EXPECT_TRUE(MatrixEqVector(m, v));

}

TEST(TheMatrixClass, ResizeByAssignment) {

    Eigen::MatrixXi a(2,2);
    Eigen::Matrix3i b{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<int>> v{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    a << 101, 102, 103, 104;

    a = b;

    EXPECT_EQ(a.rows(), b.rows());
    EXPECT_EQ(a.cols(), b.cols());

    EXPECT_TRUE(MatrixEqVector(a, v));
}