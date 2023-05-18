# just for *.cpp files
link_directories(${PROJECT_SOURCE_DIR}/lib)
# just for static brary
link_libraries(calc)

# .vcproj
add_executable(main main.cpp)
target_include_directories(main
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
