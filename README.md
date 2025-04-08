# Assignment 2
This is a part of the course "Advanced Computing for Engineering and Science Software at Scale".


## Installation
1. Install `Kokkos` with either `OpenMP` or `CUDA` backend. Here's an example cmake command:
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_PREFIX=build/install \
  -DKokkos_ENABLE_OPENMP=ON

cmake --build build -j2 --target install
```
> [!TIP]
> Don't forget to load necessary modules for compilers. On SCOREC Machines, you can use the following command:
```bash
   module use /opt/scorec/spack/rhel9/v0201_4/lmod/linux-rhel9-x86_64/Core/
   module load gcc/12.3.0-iil3lno mpich/4.1.1-xpoyz4t cuda/12.1.1-zxa4msk
   module load cmake
```

2. Installing `Catch2` is optional. If you enable testing but don't provide `Catch2_ROOT`, it will fetch it automatically.
3. To install the project, use `cmake` as usual. Use `Assignment_ENABLE_TESTING` (default ON) to enable testing. *An example configuration for SCOREC Machines is given in `scorec-config.sh` file.*
4. To run tests, use `ctest` or `make test` after building the project. **Note that some tests may not work if the test binaries are run directly from the build directory. Use `tests/` directory in that case.**


## Developer's Guide
> [!CAUTION]
> Please fork this repository and add all changes through pull requests (not directly pushing to the `main` branch).
### `clang-format`
We are using `clang-format` to enforce a consistent coding style.
Here we are using the `Google` style. To format code, run:
```bash
clang-format -i <file_name>
```
> [!TIP]
> If you do not have `clang-format` installed, you can use my installation `/lore/hasanm4/sourcestoInstall/clangd/llvm-project/build/bin/clang-format`. It should be accessible from any SCOREC machine.
> To add this directory to your `PATH` variable in your `~/.bashrc`, run:
```bash
  echo 'export PATH=$PATH:/lore/hasanm4/sourcestoInstall/clangd/llvm-project/build/bin' >> ~/.bashrc
  source ~/.bashrc
```

### Use `clang-format` as a Pre-commit Hook
> [!IMPORTANT]
> It will greatly reduce the manual labor of formatting code.

1. Do `vim .git/hooks/pre-commit` and add the following lines:
```bash
#!/bin/bash

extensions="cpp hpp h"
files=$(git diff --cached --name-only --diff-filter=ACM | grep -E "\.(${extensions// /|})$")

[ -z "$files" ] && exit 0

for file in $files; do
  if [ -f "$file" ]; then
    echo "Formatting $file"
    clang-format -i "$file"
    git add "$file"
  fi
done

exit 0
```
2. Make the hook executable:
```bash
chmod +x .git/hooks/pre-commit
```
3. Now, every time you commit, `clang-format` will automatically format *the changed `.h, .cpp, .hpp` files*.
