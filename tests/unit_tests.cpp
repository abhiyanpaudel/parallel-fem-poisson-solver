//
// Created by Fuad Hasan on 4/7/25.
//

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include "Mesh.h"


TEST_CASE("Test check_file_existence"){
    std::string filename = "assets/small_square.msh";
    check_file_existence(filename);
    printf("If it's failing, try running the test from the tests directory.\n");
}