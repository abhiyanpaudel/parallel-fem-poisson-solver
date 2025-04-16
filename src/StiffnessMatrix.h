//
// Created by hasanm4 on 4/11/25.
//

#ifndef ASSIGNMENT2_STIFFNESSMATRIX_H
#define ASSIGNMENT2_STIFFNESSMATRIX_H

#include <Kokkos_Core.hpp>

#include "Mesh.h"

// Contains the global indices of the element stiffness matrix
// This is paired so that we can sort
// data using Kokkos bitonic sorting algorithm
struct globalIndex {
  int r;
  int c;
};

// Used by the bitonic sort algorithm
struct gIDComparator {
  KOKKOS_FUNCTION constexpr bool operator()(const globalIndex &a,
                                            const globalIndex &b) const {
    if (a.r == b.r) {
      return a.c < b.c;
    }
    return a.r < b.r;
  };
};

// It contains a sequence of global indices
// and has to be sequenced the same way as the data.
// No extra effort is needed since the mesh is read by
// the same reader and has the same element and node ordering
class ElementStiffnessMatrix {
 public:
  void createOOROOC(Mesh mesh);

  void sortDataByRowCol(Kokkos::View<double *> data);

  Kokkos::View<globalIndex *> rowColCOO_;
};

// StiffnessMatrix with the CSR format
class StiffnessMatrix {
 public:
  /**
   * @brief Constructor for StiffnessMatrix
   * @details Calls the createOOROOC function to create the COO format. User has
   * to sort and then assemble the data to create the CSR format.
   */
  StiffnessMatrix(Mesh mesh);

  ElementStiffnessMatrix elementStiffnessMatrix;

  // The row pointers of the CSR format
  [[nodiscard]]
  Kokkos::View<int *> GetRowIndex() const {
    return csrRowIds_;
  };

  // The column index of the CSR format
  [[nodiscard]]
  Kokkos::View<int *> GetColIndex() const {
    return csrColIds_;
  };

  // The values of the CSR format
  [[nodiscard]]
  Kokkos::View<double *> GetValues() const {
    return csrValues_;
  };

  /**
   * @brief Get the row or column size of the stiffness matrix
   * @details The size is equal to the number of nodes in the mesh for this
   * case.
   */
  size_t GetDim() const { return nDof_; };

  /**
   * @brief Sorts the data array using (row, col) as the key using bitonic sort
   * @details This data array has to be sent to the assemble function to create
   * the CSR format.
   */
  void sortDataByRowCol(Kokkos::View<double *> data) {
    elementStiffnessMatrix.sortDataByRowCol(data);
  }

  /**
   * @brief Assembles the *sorted* data into the CSR format
   * @details The data array has to be sorted using the sortDataByRowCol
   * function before calling this function.
   */
  void assemble(Kokkos::View<double *> data);

  /**
   * @brief Get the size of the flattened element stiffness matrix
   * @details The size is equal to the number of elements times the number of
   * nodes per element squared.
   */
  [[nodiscard]]
  size_t getElementStiffnessSize() const {
    return mesh_.GetNumElements() * mesh_.GetMeshType() * mesh_.GetMeshType();
  };

  // Prints the CSR arrays
  void printStiffnessMatrix() const;

 private:
  // ********************** Private Functions **********************
  // void createRowIndex();

  // void createColIndex();

  // ********************** Private Variables **********************
  size_t nDof_;  // it's the number of nodes here
  size_t nElem_;

  Mesh mesh_;

  // ********************** CSR Data Views **********************
  size_t csrDataSize_ = 0;
  Kokkos::View<double *> csrValues_;
  Kokkos::View<int *> csrRowIds_;
  Kokkos::View<int *> csrColIds_;
};

#endif  // ASSIGNMENT2_STIFFNESSMATRIX_H
