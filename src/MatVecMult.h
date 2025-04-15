//
// Created by hasanm4 on 4/15/25.
//

#ifndef ASSIGNMENT2_MATVECMULT_H
#define ASSIGNMENT2_MATVECMULT_H

#include <Kokkos_Core.hpp>

int get_max(Kokkos::View<int *> arr);

class Vector {
 public:
  Vector(Kokkos::View<double *> data) : data(data) {}
  Kokkos::View<double *> data;
};

class CSRMatrix {
 public:
  int nRows;
  int nCols;

  CSRMatrix(int n_rows, int n_cols, Kokkos::View<int *> row_ptr,
            Kokkos::View<int *> col_ind, Kokkos::View<double *> values)
      : nRows(n_rows),
        nCols(n_cols),
        row_ptr(row_ptr),
        col_ind(col_ind),
        values(values) {
    assert(col_ind.size() == values.size());
    assert(row_ptr.size() == n_rows + 1);
  }

  CSRMatrix(Kokkos::View<int *> row_ptr, Kokkos::View<int *> col_ind,
            Kokkos::View<double *> values)
      : row_ptr(row_ptr), col_ind(col_ind), values(values) {
    assert(col_ind.size() == values.size());
    nRows = row_ptr.size() - 1;
    nCols = get_max(col_ind) + 1;
  }

  Kokkos::View<double *> multiply(const Vector V);

 private:
  Kokkos::View<int *> row_ptr;
  Kokkos::View<int *> col_ind;
  Kokkos::View<double *> values;
};

#endif  // ASSIGNMENT2_MATVECMULT_H
