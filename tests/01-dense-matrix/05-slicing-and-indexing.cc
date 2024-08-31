#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "hyssop/util.h"

TEST(SlicingAndIndexing, BottomRight) {

    Eigen::MatrixXi m {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};

    Eigen::MatrixXi e {{2, 3, 4}, {2, 3, 4}};

    Eigen::MatrixXi output = m(Eigen::seq(1, Eigen::placeholders::last), Eigen::seqN(1, 3));

    EXPECT_PRED2(MatricesApproxXi, output, e);

}

TEST(SlicingAndIndexing, TopLeft) {

    Eigen::MatrixXi m {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};

    Eigen::MatrixXi e {{1, 2, 3}, {1, 2, 3}};

    Eigen::MatrixXi output = m(Eigen::seq(0, Eigen::placeholders::last-1), Eigen::seqN(0, 3));

    EXPECT_PRED2(MatricesApproxXi, output, e);

}

TEST(SlicingAndIndexing, EvenColumns) {

    Eigen::MatrixXi m {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};

    Eigen::MatrixXi e {{1, 3}, {1, 3}, {1, 3}};

    // TODO: doesn't compile with seqN()
    // Eigen::MatrixXi output = m(Eigen::placeholders::all, Eigen::seqN(0, 4, 2));

    Eigen::MatrixXi output = m(Eigen::placeholders::all, Eigen::seq(0, 3, 2));

    EXPECT_PRED2(MatricesApproxXi, output, e);

}

TEST(SlicingAndIndexing, OddRows) {

    Eigen::MatrixXi m {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};

    Eigen::MatrixXi e {{1, 1, 1, 1}, {3, 3, 3, 3}};

    Eigen::MatrixXi output = m(Eigen::seq(1, 4, 2), Eigen::placeholders::all);

    EXPECT_PRED2(MatricesApproxXi, output, e);

}

TEST(SlicingAndIndexing, MiddleRow) {

    Eigen::MatrixXi m {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};

    Eigen::MatrixXi e {{2, 2, 2, 2}};

    Eigen::MatrixXi output = m(Eigen::placeholders::last/2, Eigen::placeholders::all);

    EXPECT_PRED2(MatricesApproxXi, output, e);

}