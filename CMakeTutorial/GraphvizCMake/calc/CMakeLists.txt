# just for *.cpp files
aux_source_directory(./ SRC)

add_library(calc STATIC ${SRC})
target_include_directories(calc
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)
