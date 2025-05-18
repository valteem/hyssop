#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "hyssop/util.h"

TEST(AdvancedInitialization, CommaInitializerMatrix) {

    Eigen::Matrix3i output;
    output << 1, 2, 3, 1, 2, 3, 1, 2, 3; // row-wise

    Eigen::Matrix3i expected {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};

    EXPECT_PRED2(MatricesApproxXi, output, expected);

}

TEST(AdvancedInitialization, CommaInitializerBlockMatrix) {

    Eigen::MatrixXi input {{2, 4}, {6, 8}};

    Eigen::MatrixXi output(4, 4);
    output << input, input *2, input/2, input;

    Eigen::MatrixXi expected {{2, 4, 4, 8}, {6, 8, 12, 16}, {1, 2, 2, 4}, {3, 4, 6, 8}};

    EXPECT_PRED2(MatricesApproxXi, output, expected);

}