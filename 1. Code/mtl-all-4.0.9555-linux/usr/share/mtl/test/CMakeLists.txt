project(mtl_test)

cmake_minimum_required(VERSION 2.8)

find_package(MTL REQUIRED)
include_directories(${MTL_INCLUDE_DIRS})
add_definitions(${MTL_CXX_DEFINITIONS})

file(GLOB SOURCES *_test.cpp)

include(CTest)
foreach (source IN LISTS SOURCES)
get_filename_component (test ${source} NAME_WE)		# filename without extension
add_executable (${test} ${source})
add_test(${test} ${test})
if(MTL_LIBRARIES)
  target_link_libraries(${test} ${MTL_LIBRARIES})
endif(MTL_LIBRARIES)
if(ENABLE_VAMPIR AND VAMPIR_FOUND)
get_target_property(CURLINK ${test} LINK_FLAGS)
# message (status "Link flags are ${CURLINK}")
if(CURLINK)
  set_target_properties(${test} PROPERTIES LINK_FLAGS "${VT_LINK_FLAGS} ${CURLINK}")
else()
  set_target_properties(${test} PROPERTIES LINK_FLAGS "${VT_LINK_FLAGS}")
endif()
endif(ENABLE_VAMPIR AND VAMPIR_FOUND)
endforeach (source IN LISTS SOURCES)
