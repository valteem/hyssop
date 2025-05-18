#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "hyssop/util.h"

TEST(AdvancedInitialization, CommaInitializerMatrix) {

    Eigen::Matrix3i output;
    output << 1, 2, 3, 1, 2, 3, 1, 2, 3; // row-wise

    Eigen::Matrix3i expected {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}};

    EXPECT_PRED2(MatricesApproxXi, output, expected);

}