cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=build/install \
  -DCMAKE_CXX_COMPILER=`which mpicxx` \
  -DCMAKE_C_COMPILER=`which mpicc` \
  -DKokkos_ROOT=../assignment2_deps/kokkos/build/install/ \
  -DAssignment_ENABLE_CUDA=OFF \
  -DCatch2_ROOT=/lore/hasanm4/practice_projects/advanced_comp/assignment2_deps/Catch2/build/install/

cmake --build build -j --target install
