//
// Created by hasanm4 on 4/15/25.
//

#include "MatVecMult.h"

using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
using member_type = typename TeamPolicy::member_type;

[[nodiscard]]
int get_max(Kokkos::View<int *> arr) {
  int max = 0;
  Kokkos::Max<int> max_reducer(max);
  Kokkos::parallel_reduce(
      "maxreduce", arr.size(),
      KOKKOS_LAMBDA(const int &i, int &lmax) {
        max_reducer.join(lmax, arr(i));
      },
      max_reducer);

  return max;
}

Kokkos::View<double *> CSRMatrix::multiply(const Vector V) {
  auto row_ptr_l = row_ptr;
  auto col_ind_l = col_ind;
  auto values_l = values;
  auto n_rows = this->nRows;
  auto v_data = V.data;

  assert(row_ptr_l.size() == n_rows + 1);
  assert(col_ind_l.size() == values_l.size());

  auto result = Kokkos::View<double *>("result", n_rows);

  TeamPolicy policy(n_rows, Kokkos::AUTO());
  Kokkos::parallel_for(
      "csr vec mult", policy, KOKKOS_LAMBDA(const member_type &team) {
        const int row_id = team.league_rank();
        const int row_start = row_ptr_l(row_id);
        const int row_end = row_ptr_l(row_id + 1);

        double row_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team, row_start, row_end),
            [&](const int &elem_idx, double &lsum) {
              lsum += values_l(elem_idx) * v_data(col_ind_l(elem_idx));
            },
            row_sum);

        Kokkos::single(Kokkos::PerTeam(team),
                       [&]() { result(row_id) = row_sum; });
      });

  return result;
}