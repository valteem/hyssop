#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "hyssop/util.h"

TEST(BlockOperations, BlockSelect) {

    Eigen::MatrixXi m{{1, 1, 2, 2}, {1, 1, 2, 2}, {3, 3, 4, 4}, {3, 3, 4, 4}};

    Eigen::MatrixXi b1{{1, 1}, {1, 1}}; // top-left 2x2 block
    Eigen::MatrixXi b2{{2, 2}, {2, 2}}; // top-right 2x2 block
    Eigen::MatrixXi b3{{3, 3}, {3, 3}}; // bottom-left 2x2 block
    Eigen::MatrixXi b4{{4, 4}, {4, 4}}; // bottom-right 2x2 block

    Eigen::MatrixXi r(2, 2);

    r = m.block<2,2>(0, 0); // 2x2 block (as per template spec), start row/column as args
    EXPECT_PRED2(MatricesApproxXi, r, b1);

    r = m.block<2,2>(0, 2);
    EXPECT_PRED2(MatricesApproxXi, r, b2);

    r = m.block(2, 0, 2, 2); // 2x2 block (last two args), start row/column as first two args
    EXPECT_PRED2(MatricesApproxXi, r, b3);

    r = m.block(2, 2, 2, 2);
    EXPECT_PRED2(MatricesApproxXi, r, b4);

}

TEST(BlockOperations, RowColumn) {

    Eigen::MatrixXi m{{1, 2}, {3, 4}};

    Eigen::MatrixXi e1{{5, 2}, {11, 4}};
    Eigen::MatrixXi e2{{5, 2}, {21, 8}};

    m.col(0) += 2 * m.col(1);
    EXPECT_PRED2(MatricesApproxXi, m, e1);
    
    m.row(1) += 2 * m.row(0);
    EXPECT_PRED2(MatricesApproxXi, m, e2);

}

TEST(BlockOperations, CornerRelated) {

    Eigen::MatrixXi m{{1, 1, 2, 2}, {1, 1, 2, 2}, {3, 3, 4, 4}, {3, 3, 4, 4}};

    Eigen::MatrixXi top_left{{1, 1}, {1, 1}};
    Eigen::MatrixXi top_right{{2, 2}, {2, 2}};
    Eigen::MatrixXi bottom_left{{3, 3}, {3, 3}};
    Eigen::MatrixXi bottom_right{{4, 4}, {4, 4}};
    
    Eigen::MatrixXi top_row {{1, 1, 2, 2}};
    Eigen::MatrixXi top_2_rows {{1, 1, 2, 2}, {1, 1, 2, 2}};
    Eigen::MatrixXi middle_2_rows {{1, 1, 2, 2}, {3, 3, 4, 4}};
    Eigen::MatrixXi bottom_2_rows {{3, 3, 4, 4}, {3, 3, 4, 4}};
    Eigen::MatrixXi bottom_row {{3, 3, 4, 4}};

    Eigen::MatrixXi left_column {{1}, {1}, {3}, {3}};
    Eigen::MatrixXi left_2_columns {{1, 1}, {1, 1}, {3, 3}, {3, 3}};
    Eigen::MatrixXi middle_2_columns {{1, 2}, {1, 2}, {3, 4}, {3, 4}};
    Eigen::MatrixXi right_2_columns {{2, 2}, {2, 2}, {4, 4}, {4, 4}};
    Eigen::MatrixXi right_column {{2}, {2}, {4}, {4}};

    // silently converts (?) Eigen::Block<Eigen::Matrix<...>> to Eigen::Matrix
    EXPECT_PRED2(MatricesApproxXi, m.topLeftCorner(2, 2), top_left);
    EXPECT_PRED2(MatricesApproxXi, m.topRightCorner(2, 2), top_right);
    EXPECT_PRED2(MatricesApproxXi, m.bottomLeftCorner(2, 2), bottom_left);
    EXPECT_PRED2(MatricesApproxXi, m.bottomRightCorner(2, 2), bottom_right);
    
    EXPECT_PRED2(MatricesApproxXi, m.topRows(1), top_row);
    EXPECT_PRED2(MatricesApproxXi, m.topRows(2), top_2_rows);
    EXPECT_PRED2(MatricesApproxXi, m.middleRows(1, 2), middle_2_rows);
    EXPECT_PRED2(MatricesApproxXi, m.bottomRows(2), bottom_2_rows);
    EXPECT_PRED2(MatricesApproxXi, m.bottomRows(1), bottom_row);

    EXPECT_PRED2(MatricesApproxXi, m.leftCols(1), left_column);
    EXPECT_PRED2(MatricesApproxXi, m.leftCols(2), left_2_columns);
    EXPECT_PRED2(MatricesApproxXi, m.middleCols(1, 2), middle_2_columns);
    EXPECT_PRED2(MatricesApproxXi, m.rightCols(2), right_2_columns);
    EXPECT_PRED2(MatricesApproxXi, m.rightCols(1), right_column);
}

TEST(BlockOperations, VectorBlocks) {

    Eigen::VectorXi v{{1, 2, 3, 4}};

    Eigen::VectorXi left_1{{1}};
    Eigen::VectorXi left_2{{1, 2}};
    Eigen::VectorXi left_3{{1, 2, 3}};
    Eigen::VectorXi left_4{{1, 2, 3, 4}};

    Eigen::VectorXi right_1{{4}};
    Eigen::VectorXi right_2{{3, 4}};
    Eigen::VectorXi right_3{{2, 3, 4}};
    Eigen::VectorXi right_4{{1, 2, 3, 4}};

    Eigen::VectorXi middle{{2, 3}};

    EXPECT_PRED2(VectorsApproxXi, v.head(1), left_1);
    EXPECT_PRED2(VectorsApproxXi, v.head(2), left_2);
    EXPECT_PRED2(VectorsApproxXi, v.head(3), left_3);
    EXPECT_PRED2(VectorsApproxXi, v.head(4), left_4);

    EXPECT_PRED2(VectorsApproxXi, v.tail(1), right_1);
    EXPECT_PRED2(VectorsApproxXi, v.tail(2), right_2);
    EXPECT_PRED2(VectorsApproxXi, v.tail(3), right_3);
    EXPECT_PRED2(VectorsApproxXi, v.tail(4), right_4);

    EXPECT_PRED2(VectorsApproxXi, v.segment(1, 2), middle);

}