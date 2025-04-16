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

struct gIDComparator {
  KOKKOS_FUNCTION constexpr bool operator()(const globalIndex &a,
                                            const globalIndex &b) const {
    if (a.r == b.r) {
      return a.c < b.c;
    }
    return a.r < b.r;
  };
};

class ElementStiffnessMatrix {
 public:
  void createOOROOC(Mesh mesh);

  void sortDataByRowCol(Kokkos::View<double *> data);

  Kokkos::View<globalIndex *> rowColCOO_;
};

class StiffnessMatrix {
 public:
  StiffnessMatrix(Mesh mesh);

  ElementStiffnessMatrix elementStiffnessMatrix;

  [[nodiscard]]
  Kokkos::View<int *> GetRowIndex() const {
    return csrRowIds_;
  };

  [[nodiscard]]
  Kokkos::View<int *> GetColIndex() const {
    return csrColIds_;
  };

  [[nodiscard]]
  Kokkos::View<double *> GetValues() const {
    return csrValues_;
  };

  size_t GetDim() const { return nDof_; };

  void sortDataByRowCol(Kokkos::View<double *> data) {
    elementStiffnessMatrix.sortDataByRowCol(data);
  }

  void assemble(Kokkos::View<double *> data);

  [[nodiscard]]
  size_t getElementStiffnessSize() const {
    return mesh_.GetNumElements() * mesh_.GetMeshType() * mesh_.GetMeshType();
  };

  void printStiffnessMatrix() const;

 private:
  // ********************** Private Functions **********************
  void createRowIndex();

  void createColIndex();

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
