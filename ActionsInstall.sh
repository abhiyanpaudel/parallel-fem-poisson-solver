git clone https://github.com/catchorg/Catch2 ../Catch2
cmake -S ../Catch2 -B ../Catch2/build/ \
  -DCMAKE_INSTALL_PREFIX=../Catch2/build/install/
cmake --build ../Catch2/build/ -j 8 --target install

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:../Catch2/build/install

git clone https://github.com/kokkos/kokkos.git ../Kokkos
cmake -S ../Kokkos -B ../Kokkos/build/ \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=`which g++` \
  -DCMAKE_INSTALL_PREFIX=../Kokkos/build/install/ \
  -DKokkos_ENABLE_OPENMP=ON
cmake --build ../Kokkos/build/ -j 8 --target install

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:../Kokkos/build/install

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=`which g++` \
  -DAssignment_ENABLE_CUDA=OFF
cmake --build build -j 8