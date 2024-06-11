#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lib::config_file_package" for configuration "Release"
set_property(TARGET lib::config_file_package APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lib::config_file_package PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/config_file_package.lib"
  )

list(APPEND _cmake_import_check_targets lib::config_file_package )
list(APPEND _cmake_import_check_files_for_lib::config_file_package "${_IMPORT_PREFIX}/lib/config_file_package.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
