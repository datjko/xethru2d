cmake_minimum_required(VERSION 2.8)

project(xethru_map2d)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

#
# External dependencies:
#

# Enable searching for libraries in lib64 directories during 64-bit builds.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

# Boost:
#

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(REQUIRED_BOOST_COMPONETS
  atomic
  chrono
  date_time
  filesystem
  iostreams
  program_options
  regex
  #serialization
  system
  thread
)

# You should provide a minimum version number that should be used.  If
#    you provide this version number and specify the REQUIRED attribute,
#    this module will fail if it can't find the specified or a *later* version.
find_package(Boost 1.54.0 REQUIRED COMPONENTS ${REQUIRED_BOOST_COMPONETS})
if (Boost_FOUND)
  include_directories(${Boost_INCLUDES})
  include_directories(${Boost_INCLUDE_DIRS})
  #link_directories(${Boost_LIBRARY_DIRS})
  message(STATUS "Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
  message(STATUS "Boost_INCLUDES=${Boost_INCLUDES}")
  message(STATUS "Boost_LIBRARY_DIRS=${Boost_LIBRARY_DIRS}")
  message(STATUS "Boost_LIBRARIES=${Boost_LIBRARIES}")
else()
  message(SEND_ERROR "Boost not found.")
endif (Boost_FOUND)


# OpenCV:
#

find_package(OpenCV REQUIRED)



#
# Build Targets:
#


# Apps

add_executable(xethru_map2d
  map2d.cpp
)

target_link_libraries(xethru_map2d
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)
