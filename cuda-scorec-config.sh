export NVCC_WRAPPER_DEFAULT_COMPILER=`which mpicxx`
cmake -B build-cuda -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=build-cuda/install \
  -DCMAKE_CXX_COMPILER=/lore/hasanm4/practice_projects/advanced_comp/assignment2_deps/kokkos/bin/nvcc_wrapper \
  -DKokkos_ROOT=../assignment2_deps/kokkos/build-cuda/install/ \
  -DAssignment_ENABLE_CUDA=ON \
  -DCatch2_ROOT=/lore/hasanm4/practice_projects/advanced_comp/assignment2_deps/Catch2/build/install/

cmake --build build-cuda -j --target install
