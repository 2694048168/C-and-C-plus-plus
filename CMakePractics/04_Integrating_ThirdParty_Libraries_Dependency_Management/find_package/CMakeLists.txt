cmake_minimum_required(VERSION 3.25)

project(find_package
    VERSION 1.0
    DESCRIPTION "To demonstrate searching third-party dependencies"
    LANGUAGES CXX
)

# enable debuging for finding the package (For illustrative purposes)
# set(CMAKE_FIND_DEBUG_MODE TRUE )

# winget install -e --id ShiningLight.OpenSSL
# Search the package OpenSSL and its component SSL, configuration fails if not found
# because of the keyword REQUIRED
find_package(OpenSSL REQUIRED COMPONENTS SSL)

# disable debugging again set(CMAKE_FIND_DEBUG_MODE FALSE )
add_executable(${PROJECT_NAME})
target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_17
)
target_sources(${PROJECT_NAME}
    PRIVATE src/main.cpp
)

# link the openssl library to the executable
target_link_libraries(${PROJECT_NAME}
    PRIVATE OpenSSL::SSL
)
