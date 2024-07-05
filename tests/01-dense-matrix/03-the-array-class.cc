#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "hyssop/util.h"

TEST(TheArrayClass, AddSubstractMultiply) {
    
    Eigen::ArrayXXi a{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    Eigen::ArrayXXi b{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};

    std::vector<std::vector<int>> expected_add_result{{3, 3, 3}, {3, 3, 3}, {3, 3, 3}};
    std::vector<std::vector<int>> expected_substract_result{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}};
    std::vector<std::vector<int>> expected_multiply_result{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};

    Eigen::ArrayXXi r(3, 3);
    Eigen::ArrayXXi er{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};

// Compare actual and expected results with custom matcher function

    r = a + b;
    EXPECT_TRUE(ArrayEqVector(r, expected_add_result));

    r = a - b;
    EXPECT_TRUE(ArrayEqVector(r, expected_substract_result));

    r = a * b;
    EXPECT_TRUE(ArrayEqVector(r, expected_multiply_result));

 // Compare actual and expected results with isApprox() helper function

    EXPECT_PRED2(MatricesApproxXXi, r.matrix(), er.matrix());

}