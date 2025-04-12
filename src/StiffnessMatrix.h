//
// Created by hasanm4 on 4/11/25.
//

#ifndef ASSIGNMENT2_STIFFNESSMATRIX_H
#define ASSIGNMENT2_STIFFNESSMATRIX_H

#include <Kokkos_Core.hpp>

#include "Mesh.h"
struct globalIndex {
  int r;
  int c;
};

class ElementStiffnessMatrix {
 public:
  void createOOROOC(Mesh mesh);
  void sortDataByRowCol(Kokkos::View<double*> data);

  Kokkos::View<globalIndex*> rowColIndex_;
};

class StiffnessMatrix {
 public:
  StiffnessMatrix(Mesh mesh);
  ElementStiffnessMatrix elementStiffnessMatrix;

  [[nodiscard]]
  Kokkos::View<int*> GetRowIndex() const {
    return rowIndex_;
  };
  size_t GetDim() const { return nDof_; };

  void sortDataByRowCol(Kokkos::View<double*> data) {
    elementStiffnessMatrix.sortDataByRowCol(data);
  }

 private:
  // ********************** Private Functions **********************
  void createRowIndex();
  void createColIndex();

  // ********************** Private Variables **********************
  size_t nDof_;  // it's the number of nodes here
  size_t nElem_;

  Mesh mesh_;

  // ********************** Data Views **********************
  Kokkos::View<double*> MData_;
  Kokkos::View<int*> rowIndex_;
  Kokkos::View<int*> colIndex_;
};

#endif  // ASSIGNMENT2_STIFFNESSMATRIX_H
