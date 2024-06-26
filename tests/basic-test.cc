#include "gtest/gtest.h"

#include <Eigen/Dense>

TEST(Primary, Basic) {
    Eigen::Matrix<int,2,2> m;
    m(0,0) = 1;
    m(1,1) = 1;
    auto tr = m(0,0) + m(1,1);
    GTEST_ASSERT_EQ(tr, 2);
};

TEST(Primary, Standalone) {
    EXPECT_EQ(3*4, 12);
}