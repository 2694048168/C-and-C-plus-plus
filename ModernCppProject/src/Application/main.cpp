#include "Application/Application.hpp"
#include "Version/version.h"

#include <iostream>

int main(int argc, char *argv[])
{
    Application::getInstance().Run();

    std::cout << "Version NO: " << Ithaca::version::VERSION_MAJOR << "." << Ithaca::version::VERSION_MINOR << "."
              << Ithaca::version::VERSION_PATCH << std::endl;
    std::cout << "Version: " << Ithaca::version::VERSION_STRING << std::endl;
    std::cout << "Build Time: " << Ithaca::version::BUILD_TIMESTAMP << std::endl;
    std::cout << "Version-Build Time: " << Ithaca::version::VERSION_BUILD_STRING << std::endl;
    return 0;
}
