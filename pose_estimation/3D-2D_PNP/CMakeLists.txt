cmake_minimum_required( VERSION 2.8 )
project( pnp_solver )

set( CMAKE_BUILD_TYPE "-Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

find_package( OpenCV 3.1 REQUIRED )

include_directories( 
    ${PROJECT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS} 
    # "/usr/include/eigen3/" # use in linux
    /usr/local/Cellar/eigen/3.3.7/include/eigen3 # use in mac
)


add_executable(pnp_solver src/main.cpp src/pnp_solver.cc src/random_array.cc)
target_link_libraries(pnp_solver  ${OpenCV_LIBS})