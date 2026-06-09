#include "Application/Application.hpp"

#include <iostream>

Application::Application() {}

Application::~Application() {}

Application &Application::getInstance()
{
    static Application instance;
    return instance;
}

void Application::Run()
{
    std::cout << "Running application..." << std::endl;
}
