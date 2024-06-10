cmake_minimum_required(VERSION 3.25)

project("00_Kickstarting_CMake"
    VERSION 2.0
    DESCRIPTION "Simple project for basic CMake usage"
    LANGUAGES CXX
)

message(STATUS [====]${PROJECT_NAME})
message(STATUS [====]${CMAKE_PROJECT_NAME})
message(STATUS [====]${PROJECT_IS_TOP_LEVEL})

# --------------------------------------------------------------------
# The following are some additional variables regarding the projects. All of them can be
# prefixed with CMAKE_ to the value for the root project. If they're not defined in the
# project() directive, the strings are empty:
# • PROJECT_DESCRIPTION: The description string of the project
# • PROJECT_HOMEPAGE_URL: The URL string for the project
# • PROJECT_VERSION: The full version that's given to the project
# • PROJECT_VERSION_MAJOR: The first number of the version string
# • PROJECT_VERSION_MINOR: The second number of the version string
# • PROJECT_VERSION_PATCH: The third number of the version string
# • PROJECT_VERSION_TWEAK: The fourth number of the version string
message(STATUS [====]${PROJECT_NAME})
message(STATUS [====]${CMAKE_PROJECT_NAME})
message(STATUS [====]${PROJECT_IS_TOP_LEVEL})
message(STATUS [====The version]${CMAKE_PROJECT_VERSION})
message(STATUS [====The major version]${CMAKE_PROJECT_VERSION_MAJOR})
message(STATUS [====The minor version]${CMAKE_PROJECT_VERSION_MINOR})
message(STATUS [====The patch version]${CMAKE_PROJECT_VERSION_PATCH})
message(STATUS [====The tweak version]${CMAKE_PROJECT_VERSION_TWEAK})

# Variables can be set using the set command and deleted using unset.
# Variable names are case-sensitive
set(MYVAR "WeiLi.Ithaca")
message(STATUS "[====set]The content of MYVAR are ${MYVAR}")
unset(MYVAR)
message(STATUS "[====unset]The content of MYVAR are ${MYVAR}")

# Variables might be scoped in the following way:
# • Function scope: Variables that are set inside a function are only visible inside
# the function.
# • Directory scope: Each of the subdirectories in a source tree binds variables and
# includes any variable bindings from the parent directory.
# • Persistent cache: Cached variables can be either system- or user-defined. These
# persist their values over multiple runs.
# • Passing the PARENT_SCOPE option to set() makes the variable visible in the
# parent scope.
# Variable references can even be nested and are evaluated inside out
set(MANJOR 4)
set(MINOR4 "4.6")
message(STATUS "[====]${MINOR${MANJOR}}")
unset(MANJOR)
unset(MINOR4)

# semicolon-separated string
set(SRC_FILES main.cpp main2.cpp main3.cpp)
set(SRC_FILE "main.cpp;main2.cpp;main3.cpp")
message(STATUS [====]${SRC_FILES})
message(STATUS [====]${SRC_FILE})

# Manipulating lists by modifying their contents, reordering,
# or finding things can be done using the list command.
list(FIND SRC_FILES main.cpp MAIN_CPP_IDX)
message(STATUS [====MAIN_CPP_IDX]${MAIN_CPP_IDX})
list(GET SRC_FILES ${MAIN_CPP_IDX} MAIN)
message(STATUS [====MAIN]${MAIN})

# To append a value to a list
list(APPEND SRC_FILES "main4.cpp")
message(STATUS [====src_files]${SRC_FILES})

# Cached variables and options
# CMakeCache.txt files. Usually, you don't have to edit them manually,
# but they are great for debugging builds that do not behave as expected.
# Most cached variables that are automatically generated are marked as advanced
set(_MYVAR foo CACHE STRING "Variable foo that configures bar")
message(STATUS [====cached Var]${_MYVAR})

# Boolean cache variables, CMake also provides the option keyword
# option has a default value of OFF unless specified otherwise
# They can also depend on each othervia the CMakeDependentOption module
option(_PRINT_LANGUAGE_EXAMPLES "Print examples for each language" OFF)
message(STATUS [====Option Boolean cache Var] ${_PRINT_LANGUAGE_EXAMPLES})

include(CMakeDependentOption)
cmake_dependent_option(_PRINT_HELLO_WORLD "print a greeting from chapter1 " ON
    _PRINT_LANGUAGE_EXAMPLES ON)
message(STATUS [====Option depend Var] ${_PRINT_HELLO_WORLD})

# Properties in CMake are values that are attached to a specific object or scope of CMake,
# such as a file, target, directory, or test case.
# https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html

# Loops and conditions
# Unary keywords are prefixed before the value:
# • COMMAND: True if the supplied value is a command
# • EXISTS: Checks whether a file or a path exists
# • DEFINED: True if the value is a defined variable
# Additionally, there are unary filesystem conditions:
# • EXISTS: True if the passed file or directory exits
# • IS_DIRECTORY: Checks whether the supplied path is a directory
# • IS_SYMLINK: True if the supplied path is a symbolic link
# • IS_ABSOULTE: Checks whether a supplied path is an absolute path
if(DEFINED MY_VAR)
    message(STATUS [====condition True]${MYY_VAR})
else()
    message(STATUS [====condition False])
endif()

set(MYVAR FOO)

# Binary tests compare two values and are placed between the values to be compared,
# The binary operators are as follows:
# • LESS, GREATER, EQUAL, LESS_EQUAL, and GREATER_EQUAL: These compare numeric values.
# • STRLESS, STREQUAL, STRGREATER, STRLESS_EQUAL, and STRGREATER_EQUAL: These lexicographically compare strings.
# • VERSION_LESS, VERSION_EQUAL, VERSION_GREATER, VERSION_LESS_EQUAL, and VERSION_GREATER_EQUAL: These compare version strings.
# • MATCHES: This compares against a regular expression.
# • IS_NEWER_THAN: Checks which of the two files that passed has been modified recently.
# • IS_NEWER_THAN: Unfortunately, this is not very precise because if both files have
# the same timestamp, it also returns true. There is also more confusion because if
# either of the files is missing, the result is also true.
if(MYVAR STREQUAL "FOO")
    message(STATUS [====binary compare True])
else()
    message(STATUS [====binary compare False])
endif()

# Finally, there's the Boolean OR, AND, and NOT operators.
# Loops are either achieved by while() and endwhile() or foreach() and
# endforeach(). Loops can be terminated using break(); continue() aborts
# the current iteration and starts the next one immediately.
set(MYVAR 0)

while(MYVAR LESS "5")
    message(STATUS "[====]: MYVAR is '${MYVAR}'")
    math(EXPR MYVAR "${MYVAR}+1")
endwhile()

set(MYLIST main.cpp main2.cpp main3.cpp)

foreach(ITEM IN LISTS MYLIST)
    message(STATUS [===== foreach] ${ITEM})
endforeach()

foreach(ITEM RANGE 0 6)
    if(ITEM EQUAL 3)
        continue()
    elseif(ITEM EQUAL 5)
        break()
    endif()

    message(STATUS [===== foreach Range] ${ITEM})
endforeach()

# Functions are defined by function()/endfunction(). Functions open a new scope
# for variables, so all the variables that are defined inside are not accessible from the outside
# unless the PARENT_SCOPE option is passed to set().
# Functions are case-insensitive and are invoked by calling function,
function(func ARG1)
    message(STATUS [function call params]${ARG1})
endfunction()

# invoke func with parameter bar
func("bar")

# CMake macros are defined using the macro()/endmacro() commands. They are a
# bit like functions, with the difference that in functions, the arguments are true variables,
# whereas in macros, they are string replacements.
# Consequently, it is highly recommended to avoid calling return() in
# macros as this would stop the scope

# Targets
# The build system of CMake is organized as a set of logical targets that correspond to an
# executable, library, or custom command or artifact, such as documentation or similar.
# There are three major ways to create a target in CMake – add_executable,
# add_library, and add_custom_target. The first two are used to create executables
# and static or shared libraries, while the third can contain almost any custom command to be executed.
# visibility modifiers such as PRIVATE, PUBLIC, or INTERFACE
# to denote which requirements are transitive

# --------------------------------------------------------------------
add_executable(00_Kickstarting_CMake)
target_sources(00_Kickstarting_CMake
    PRIVATE
    src/main.cpp
)

# Generator expressions
# Generator expressions are small statements that are evaluated during the configuration
# phase of the build. Most functions allow generator expressions to be used, with a few
# exceptions. They take the form of $<OPERATOR:VALUE>, where OPERATOR is applied or
# compared to VALUE. You can think of generator expressions as small inline if-statements.
target_compile_options(00_Kickstarting_CMake PRIVATE
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>")
