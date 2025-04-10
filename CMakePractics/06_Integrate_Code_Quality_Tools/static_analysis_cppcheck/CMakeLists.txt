cmake_minimum_required(VERSION 3.25)

project("cppcheck_example"
  VERSION 1.0
  DESCRIPTION "A simple C++ project to demonstrate using cppcheck on an executable"
  LANGUAGES CXX
)

find_program(CPPCHECK_EXECUTABLE NAMES cppcheck)

# search for the cppcheck exectuable if it exists enable it
if(CPPCHECK_EXECUTABLE)
  option(ENABLE_CPPCHECK "Enable cppcheck" ON)

  # setting the flags for cppcheck, if --error-exitcode=1 is enabled compilation will
  # fail if an error is detected
  list(APPEND CPPCHECK_COMMAND "${CPPCHECK_EXECUTABLE}"
    "--enable=warning"
    "--inconclusive"
    "--force"
    "--inline-suppr"

    # "--error-exitcode=1"
  )
endif()

add_executable(cppcheck_example)
target_sources(cppcheck_example
  PRIVATE src/main.cpp
)

# Enable cppcheck for the target
if(ENABLE_CPPCHECK)
  set_target_properties(cppcheck_example
    PROPERTIES CXX_CPPCHECK "${CPPCHECK_COMMAND}"
  )
endif()
