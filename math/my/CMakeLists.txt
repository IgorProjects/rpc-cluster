cmake_minimum_required(VERSION 3.17)
project(my_math)

add_executable(my_mpi src/main.cpp)

find_package(MPI REQUIRED)
target_link_libraries(my_mpi ${MPI_LIBRARIES})

find_package(Boost COMPONENTS mpi REQUIRED)
target_include_directories(my_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${Boost_INCLUDE_DIR})
target_link_libraries(my_mpi ${Boost_LIBRARIES})

find_package(OpenMP)
target_link_libraries(my_mpi OpenMP::OpenMP_CXX)

set_target_properties(my_mpi PROPERTIES CXX_STANDARD 17)