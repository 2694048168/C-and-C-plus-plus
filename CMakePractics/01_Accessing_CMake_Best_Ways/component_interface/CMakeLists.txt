project(
  _framework_component_interface
  VERSION 1.0
  DESCRIPTION "Component interface implementation"
  LANGUAGES CXX
)

add_library(_framework_component_interface INTERFACE)
target_include_directories(
  _framework_component_interface INTERFACE include/
)

target_compile_features(_framework_component_interface INTERFACE cxx_std_11)

# Make specified target(s) installable. Separate them to components.
install(TARGETS _framework_component_interface COMPONENT ch2.libraries)

# Install the headers
install(
  DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  COMPONENT ch2.libraries
  DESTINATION include
)
