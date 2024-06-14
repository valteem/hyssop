#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "util.h"

TEST(MatrixArithmetic, Addition) {
    
    Eigen::MatrixXi a{{1, 2}, {3, 4}};
    Eigen::MatrixXi b{{1, 2}, {3, 4}};

    std::vector<std::vector<int>> v{{2, 4}, {6, 8}};

    Eigen::MatrixXi m(2,2);
    
    m = a + b;

    bool result = MatrixEqVector<int>(m, v);

    EXPECT_EQ(result, true);

}
