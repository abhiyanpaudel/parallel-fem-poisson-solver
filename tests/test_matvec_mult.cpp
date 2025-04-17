//
// Created by hasanm4 on 4/15/25.
//

#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>

#include "MatVecMult.h"
#include "catch2/matchers/catch_matchers_floating_point.hpp"

using doubleView = Kokkos::View<double*>;
using intView = Kokkos::View<int*>;

TEST_CASE("Test matrix-vector multiplication") {
  Kokkos::initialize();
  {
    // M
    // [1 2 0 0]
    // [0 3 4 9]
    // [5 0 6 5]

    constexpr int N = 3;
    constexpr int M = 4;
    constexpr int nnz = 8;

    intView row_ptr("row_ptr", N + 1);
    intView col_ind("col_ind", nnz);
    doubleView val("val", nnz);

    auto row_ptr_h = Kokkos::create_mirror_view(row_ptr);
    auto col_ind_h = Kokkos::create_mirror_view(col_ind);
    auto val_h = Kokkos::create_mirror_view(val);

    row_ptr_h(0) = 0;
    row_ptr_h(1) = 2;
    row_ptr_h(2) = 5;
    row_ptr_h(3) = 8;

    col_ind_h(0) = 0;
    col_ind_h(1) = 1;
    col_ind_h(2) = 1;
    col_ind_h(3) = 2;
    col_ind_h(4) = 3;
    col_ind_h(5) = 0;
    col_ind_h(6) = 2;
    col_ind_h(7) = 3;

    val_h(0) = 1.0;
    val_h(1) = 2.0;
    val_h(2) = 3.0;
    val_h(3) = 4.0;
    val_h(4) = 9.0;
    val_h(5) = 5.0;
    val_h(6) = 6.0;
    val_h(7) = 5.0;

    Kokkos::deep_copy(row_ptr, row_ptr_h);
    Kokkos::deep_copy(col_ind, col_ind_h);
    Kokkos::deep_copy(val, val_h);

    REQUIRE(M == get_max(col_ind) + 1);

    // CSRMatrix
    auto A = CSRMatrix(row_ptr, col_ind, val);
    REQUIRE(A.get_nCols() == M);
    REQUIRE(A.get_nRows() == N);

    // x
    // [1 2 3 -1]
    doubleView x("x", M);
    auto x_h = Kokkos::create_mirror_view(x);
    x_h(0) = 1.0;
    x_h(1) = 2.0;
    x_h(2) = 3.0;
    x_h(3) = -1.0;
    Kokkos::deep_copy(x, x_h);

    // Vector
    auto V = Vector(x);

    auto y = A.multiply(V);
    auto y_h = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(y_h, y);

    // Expected result
    std::vector<double> expected_y = {5.0, 9.0, 18.0};
    printf("y size = %zu\n", y_h.size());
    REQUIRE(y_h.size() == N);
    for (int i = 0; i < N; ++i) {
      printf("Expected y[%d] = %f, Actual y[%d] = %f\n", i, expected_y[i], i,
             y_h(i));
      REQUIRE_THAT(y_h(i), Catch::Matchers::WithinAbs(expected_y[i], 1e-2));
    }
  }
  Kokkos::finalize();
}
