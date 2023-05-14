cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

add_executable(MatClass)
target_sources(MatClass
    PRIVATE
        "mat_class.cpp"
)

add_executable(MatTemplate)
target_sources(MatTemplate
    PRIVATE
        "mat_template.cpp"
)

add_executable(nontype_param)
target_sources(nontype_param
    PRIVATE
        "nontype_param.cpp"
)

add_executable(specializationClass)
target_sources(specializationClass
    PRIVATE
        "specialization.cpp"
)
