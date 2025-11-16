cmake_minimum_required(VERSION 3.25.0)

project("MinimalQtVTK"
    VERSION 1.0.1
    DESCRIPTION "VTK and QT via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C
)

# =============================================
add_executable(${PROJECT_NAME}
    main.cpp
    widget.cpp
    widget.ui
)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${qt_modules} 
    ${VTK_LIBRARIES}
)
# vtk_module_autoinit is needed
vtk_module_autoinit(
  TARGETS ${PROJECT_NAME}
  MODULES ${VTK_LIBRARIES}
)
