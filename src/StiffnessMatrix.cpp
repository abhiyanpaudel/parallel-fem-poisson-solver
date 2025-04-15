//
// Created by hasanm4 on 4/11/25.
//

#include "StiffnessMatrix.h"

#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>

StiffnessMatrix::StiffnessMatrix(Mesh mesh) : mesh_(mesh) {
  nDof_ = mesh_.GetNumVertices();
  nElem_ = mesh_.GetNumElements();
  Kokkos::resize(csrRowIds_, nDof_ + 1);
  // createRowIndex();
  elementStiffnessMatrix.createOOROOC(mesh_);
}

void StiffnessMatrix::createRowIndex() {
  auto mesh_data = mesh_.GetData();
  int n_local_verts = mesh_.GetMeshType() + 1;
  // auto rowIndex_scatter =
  // Kokkos::Experimental::create_scatter_view(csrRowIds_);
  auto rowIndex_l = csrRowIds_;

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

  Kokkos::resize(rowColCOO_, n_elems * size_of_local_stiffness);
  auto rowColIndex_l = rowColCOO_;

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
  assert(rowColCOO_.size() == data.size());
  auto rowColCoo_l = rowColCOO_;

  /*
  struct gIDComparator {
    KOKKOS_FUNCTION constexpr bool operator()(
        const globalIndex &a, const globalIndex &b) const {
      if (a.r == b.r) {
        return a.c < b.c;
      }
      return a.r < b.r;
    };
  };
   */

  Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(),
                                    rowColCoo_l, data, gIDComparator());
}

void StiffnessMatrix::assemble(Kokkos::View<double *> data) {
  auto rowColCOO_l = elementStiffnessMatrix.rowColCOO_;
  size_t coo_size = rowColCOO_l.size();

  // * Find unique row/col indices
  auto unique_pos_flag = Kokkos::View<int *>("uniqueFlag", coo_size);
  auto unique_row_start_index = Kokkos::View<size_t *>("uniquestartidx", nDof_);
  Kokkos::parallel_for(
      "MarkUnique", coo_size, KOKKOS_LAMBDA(const size_t i) {
        int row_flip_flag =
            (i == 0 || (rowColCOO_l(i).r != rowColCOO_l(i - 1).r));
        int col_flip_flag = (rowColCOO_l(i).c != rowColCOO_l(i - 1).c);
        unique_pos_flag(i) = row_flip_flag || col_flip_flag;
        if (row_flip_flag) {
          unique_row_start_index(rowColCOO_l(i).r) = i;
        }
      });

  // * Create CSR row index
  auto csrRowIds_l = csrRowIds_;  // TODO: Use scatter view
  assert(csrRowIds_l.size() == nDof_ + 1);
  Kokkos::parallel_for(
      "create row index", coo_size, KOKKOS_LAMBDA(const int i) {
        if (unique_pos_flag(i) == 1) {
          int row = rowColCOO_l(i).r;
          Kokkos::atomic_inc(&csrRowIds_l(row + 1));
        }
      });
  Kokkos::fence();

  Kokkos::parallel_scan(
      "create row index scan", csrRowIds_l.size(),
      KOKKOS_LAMBDA(const size_t i, size_t &partial_sum, const bool final) {
        partial_sum += csrRowIds_l(i);
        if (final) {
          csrRowIds_l(i) = partial_sum;
        }
      },
      csrDataSize_);
  Kokkos::fence();
  printf("Total unique entries: %zu\n", csrDataSize_);

  Kokkos::resize(csrColIds_, csrDataSize_);
  Kokkos::resize(csrValues_, csrDataSize_);
  auto csrColIds_l = csrColIds_;
  auto csrValues_l = csrValues_;

  // * Fill CSR row index
  // TODO: Multilevel parallelism
  auto nDof_l = nDof_;
  Kokkos::parallel_for(
      "fill CSR", nDof_, KOKKOS_LAMBDA(const size_t row) {
        size_t row_start = unique_row_start_index(row);
        size_t row_end =
            (row == nDof_l - 1) ? coo_size : unique_row_start_index(row + 1);
        assert(row_end > row_start);

        int col_i = -1;
        for (size_t coo_i = row_start; coo_i < row_end; ++coo_i) {
          col_i += unique_pos_flag(coo_i);
          size_t csr_data_index = csrRowIds_l(row) + col_i;
          csrColIds_l(csr_data_index) = rowColCOO_l(coo_i).c;
          csrValues_l(csr_data_index) += data(coo_i);
        }
      });
}

void StiffnessMatrix::printStiffnessMatrix() const {
  auto csr_row_host = Kokkos::create_mirror_view(csrRowIds_);
  auto csr_col_host = Kokkos::create_mirror_view(csrColIds_);
  auto csr_val_host = Kokkos::create_mirror_view(csrValues_);

  Kokkos::deep_copy(csr_row_host, csrRowIds_);
  Kokkos::deep_copy(csr_col_host, csrColIds_);
  Kokkos::deep_copy(csr_val_host, csrValues_);

  printf("Row Index:\n");
  for (int i = 0; i < csr_row_host.size(); ++i) {
    printf("%d ", csr_row_host(i));
  }
  printf("\nColumn Index:\n");
  for (int i = 0; i < csr_col_host.size(); ++i) {
    printf("%d ", csr_col_host(i));
  }
  printf("\nValues:\n");
  for (int i = 0; i < csr_val_host.size(); ++i) {
    printf("%.1f ", csr_val_host(i));
  }
  printf("\n");
}
