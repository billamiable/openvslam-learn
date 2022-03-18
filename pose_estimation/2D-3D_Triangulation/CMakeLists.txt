cmake_minimum_required( VERSION 2.8 )
project( triangulator )

set( CMAKE_BUILD_TYPE "-Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

find_package( OpenCV REQUIRED )

include_directories( 
    ${PROJECT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS} 
    "/usr/include/eigen3/" # use in linux
    # /usr/local/Cellar/eigen/3.3.7/include/eigen3 # use in mac
)


add_executable(triangulator src/main.cpp)
target_link_libraries(triangulator  ${OpenCV_LIBS})