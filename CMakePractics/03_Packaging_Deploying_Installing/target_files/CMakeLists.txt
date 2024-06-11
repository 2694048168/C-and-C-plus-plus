cmake_minimum_required(VERSION 3.25)

project(target_file
    VERSION 1.0
    DESCRIPTION "TARGET install file/program"
    LANGUAGES CXX
)

include(GNUInstallDirs)

install(FILES "${CMAKE_CURRENT_LIST_DIR}/script"
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(PROGRAMS "${CMAKE_CURRENT_LIST_DIR}/script.py"
    # DESTINATION "${CMAKE_INSTALL_BINDIR}" RENAME script
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
