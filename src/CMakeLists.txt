################ Sources and Headers ################

set(Assignment2_SOURCES
    main.cpp
    Mesh.cpp
	CalculateStiffnessMatrixAndLoadVector.cpp	
)

set (Assignment2_HEADERS
    Mesh.h
	Element.hpp
	TriElement.hpp
	QuadElement.hpp
	CalculateStiffnessMatrixAndLoadVector.hpp	
)


################ Executables ################
add_executable(Assignment2)
target_sources(Assignment2 PRIVATE ${Assignment2_SOURCES})
target_link_libraries(Assignment2 PRIVATE Kokkos::kokkos)
if(Assignment_ENABLE_CUDA)
    target_compile_definitions(Assignment2 PRIVATE -DASSIGNMENT_ENABLE_CUDA)
endif()


################ Install ################
target_include_directories(Assignment2
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}

    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# just install the executable
install(TARGETS Assignment2
    EXPORT Assignment2Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT Assignment2Targets
    FILE Assignment2Targets.cmake
    NAMESPACE Assignment2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Assignment2
)

