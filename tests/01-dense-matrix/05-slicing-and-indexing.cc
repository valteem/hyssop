#include "gtest/gtest.h"

#include "Eigen/Dense"

#include "hyssop/util.h"

#include <vector>

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

// Works for fixed size vectors only
TEST(SlicingAndIndexing, LastElementsStartingAtIndex) {

    Eigen::Vector4i input {0, 1, 2, 3};

    Eigen::Vector2i expected {2, 3};

    Eigen::Vector2i output = input(Eigen::seq(2, Eigen::placeholders::last));

    EXPECT_PRED2(VectorsApproxXi, output, expected);

}

// Does not work with input vector of non-fixed size (VectorXi)
TEST(SlicingAndIndexing, LastElementsNumber) {

    Eigen::Vector4i input(0, 1, 2, 3);

    Eigen::Vector3i expected {1, 2, 3};

    Eigen::Vector3i output = input(Eigen::seq(Eigen::placeholders::last + 1 - 3, Eigen::placeholders::last));

    EXPECT_PRED2(VectorsApproxXi, output, expected);

}

TEST(SlicingAndIndexing, ArrayOfIndices) {

    Eigen::MatrixXi input {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}};

    std::vector<int> ind {2, 1, 0, 3};

    Eigen::MatrixXi output = input(ind, Eigen::placeholders::all);

    Eigen::MatrixXi expected {{2, 2, 2, 2}, {1, 1, 1, 1}, {0, 0, 0, 0}, {3, 3, 3, 3}};

    std::cout << output << std::endl;

    EXPECT_PRED2(MatricesApproxXi, output, expected);

}

TEST(SkicingAndIndexing, CustomIndexList) {

    class pad {
        public:
        Eigen::Index size_in, size_out;
        Eigen::Index size() const {return size_out;};
        Eigen::Index operator[] (Eigen::Index i) const {return std::max<Eigen::Index>(0, i - (size_out - size_in));};
    };

    Eigen::MatrixXi input {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}};

    Eigen::MatrixXi expected {{1, 1, 2, 3, 4}, {1, 1, 2, 3, 4}, {1, 1, 2, 3, 4}, {1, 1, 2, 3, 4}, {1, 1, 2, 3, 4}};

    Eigen::MatrixXi output = input(pad{4, 5}, pad{4, 5});

    EXPECT_PRED2(MatricesApproxXi, output, expected);

}