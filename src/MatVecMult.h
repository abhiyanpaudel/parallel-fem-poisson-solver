//
// Created by hasanm4 on 4/15/25.
//

#ifndef ASSIGNMENT2_MATVECMULT_H
#define ASSIGNMENT2_MATVECMULT_H

#include <Kokkos_Core.hpp>

int get_max(Kokkos::View<int *> arr);

/**
 * @brief A class representing a vector.
 */
class Vector {
 public:
  Vector(Kokkos::View<double *> data) : data(data) {}
  Kokkos::View<double *> data;
};

class CSRMatrix {
 public:
  /**
   * @brief Constructor for CSRMatrix (Shape Provided)
   */
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

  /**
   * @brief Constructor for CSRMatrix (Shape is calculated from given data)
   */
  CSRMatrix(Kokkos::View<int *> row_ptr, Kokkos::View<int *> col_ind,
            Kokkos::View<double *> values)
      : row_ptr(row_ptr), col_ind(col_ind), values(values) {
    assert(col_ind.size() == values.size());
    nRows = row_ptr.size() - 1;
    nCols = get_max(col_ind) + 1;
  }

  // Matrix-Vector multiplication
  Kokkos::View<double *> multiply(const Vector V);

  [[nodiscard]] int get_nRows() const { return nRows; }
  [[nodiscard]] int get_nCols() const { return nCols; }

 private:
  int nRows;
  int nCols;

  Kokkos::View<int *> row_ptr;
  Kokkos::View<int *> col_ind;
  Kokkos::View<double *> values;
};

#endif  // ASSIGNMENT2_MATVECMULT_H
