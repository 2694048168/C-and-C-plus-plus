Main Page {#mainpage}
=========

# Chapter num - Example 01

This example is intended to illustrate integration between CMake and the Doxygen.

## Project structure

Project contains a static library and an executable target. Static library consist of two header files and one source file (@ref include/calc/calculator.hpp "calculator.hpp", @ref include/calc/calculator_interface.hpp "calculator_interface.hpp", @ref src/calculator.cpp "calculator.cpp"), whereas executable only contains a single source file (@ref src/main.cpp "main.cpp").

### Static library (doxygen_doc_lib)

An example library that provides a class named @ref ProjectName::Calc::calculator "calculator" . This class contains four static functions named @ref ProjectName::Calc::calculator::sum "sum(...)", @ref ProjectName::Calc::calculator::sub "sub(...)", @ref ProjectName::Calc::calculator::div "div(...)", @ref ProjectName::Calc::calculator::mul "mul(...)", and a member variable named @ref ProjectName::Calc::calculator::last_result "last_result", which keeps the result of the last performed operation. In order to be able to illustrate documentation generation, functions and variables are properly documented in Doxygen JavaDoc format.

### Example application(ch6_ex01_doxdocgen_exe)

The application that consumes the @ref ProjectName::Calc::calculator "calculator" class and prints basic four arithmetic operation outputs to the stdout. Example application is not important for this example's purpose. It is included for completeness.
