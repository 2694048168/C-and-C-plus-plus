project(
  _framework_component2
  VERSION 1.0
  DESCRIPTION "Component 2 implementation"
  LANGUAGES CXX
)

add_library(
  _framework_component2 SHARED
) # Create a shared library target named ch2_framework_component2 (retrieved

# from project name variable)
target_include_directories(
  _framework_component2 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/
) # Add include/ to include directories with visibility of PUBLIC.
target_sources(
  _framework_component2 PRIVATE src/component2.cpp
) # Add src/component2.cpp file as sources to the target with PRIVATE

# visibility.
target_link_libraries(
  _framework_component2 PUBLIC _framework_component_interface
) # Link against ch2_framework_component_interface

target_compile_features(_framework_component2 PUBLIC cxx_std_11)

# Make specified target(s) installable. Separate them to components.
install(
  TARGETS _framework_component2
  COMPONENT ch2.libraries
)

# Install the headers
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  COMPONENT ch2.libraries
  DESTINATION include
)
