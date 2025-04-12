//
// Created by hasanm4 on 4/11/25.
//

#include "StiffnessMatrix.h"

#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>

StiffnessMatrix::StiffnessMatrix(Mesh mesh) : mesh_(mesh) {
  nDof_ = mesh_.GetNumVertices();
  nElem_ = mesh_.GetNumElements();
  Kokkos::resize(rowIndex_, nDof_ + 1);
  // createRowIndex();
  elementStiffnessMatrix.createOOROOC(mesh_);
}

void StiffnessMatrix::createRowIndex() {
  auto mesh_data = mesh_.GetData();
  int n_local_verts = mesh_.GetMeshType() + 1;
  // auto rowIndex_scatter =
  // Kokkos::Experimental::create_scatter_view(rowIndex_);
  auto rowIndex_l = rowIndex_;

  /* ! Wrong
  // TODO: MDRange/ Hierarchical
  Kokkos::parallel_for("fillrowsizes", nElem_, KOKKOS_LAMBDA (const int elem_id)
  { for (int i = 0; i < n_local_verts; ++i) {
          //auto access = rowIndex_scatter.access();
          int node_id = mesh_data(elem_id, i, 0); //TODO: Create issue to remove
  double indexing Kokkos::atomic_add(&rowIndex_l(node_id), n_local_verts);
      }
  });
  */
}

void ElementStiffnessMatrix::createOOROOC(Mesh mesh) {
  auto mesh_data = mesh.GetData();
  auto n_elems = mesh.GetNumElements();
  int n_local_verts = mesh.GetMeshType();

  int size_of_local_stiffness = n_local_verts * n_local_verts;

  Kokkos::resize(rowColIndex_, n_elems * size_of_local_stiffness);
  auto rowColIndex_l = rowColIndex_;

  // TODO : Use MDRange for better performance as it's a nested loop
  Kokkos::parallel_for(
      "filloorooc", n_elems, KOKKOS_LAMBDA(const int elem_id) {
        for (int row = 0; row < n_local_verts; ++row) {
          for (int col = 0; col < n_local_verts; ++col) {
            int node_id_i = mesh_data(elem_id, row, 0);
            int node_id_j = mesh_data(elem_id, col, 0);
            int data_id =
                elem_id * size_of_local_stiffness + row * n_local_verts + col;
            rowColIndex_l(data_id).r = node_id_i;
            rowColIndex_l(data_id).c = node_id_j;
          }
        }
      });
}

void ElementStiffnessMatrix::sortDataByRowCol(Kokkos::View<double *> data) {
  assert(rowColIndex_.size() == data.size());
  auto rowColIndex_l = rowColIndex_;

  struct gIDComparator {
    KOKKOS_INLINE_FUNCTION constexpr bool operator()(
        const globalIndex &a, const globalIndex &b) const {
      if (a.r == b.r) {
        return a.c < b.c;
      }
      return a.r < b.r;
    };
  };

  Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(),
                                    rowColIndex_, data, gIDComparator());
}

void StiffnessMatrix::assemble(Kokkos::View<double *> data) {
  auto row_index_l = rowIndex_;  // TODO: Use scatter view
  auto rowcolIndex_l = elementStiffnessMatrix.rowColIndex_;

  Kokkos::parallel_for(
      "create row index", rowcolIndex_l.size(), KOKKOS_LAMBDA(const int i) {
        int row = rowcolIndex_l(i).r;
        Kokkos::atomic_inc(&row_index_l(row + 1));
      });
  Kokkos::fence();

  Kokkos::parallel_scan(
      "create row index scan", rowcolIndex_l.size(),
      KOKKOS_LAMBDA(const int i, int &partial_sum, const bool final) {
        partial_sum += row_index_l(i);
        if (final) {
          row_index_l(i) = partial_sum;
        }
      });
}
