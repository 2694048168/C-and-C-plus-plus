#include "PrintModuleLib/Module.hpp"

void testPrintDebug(PrintModule *p_print)
{
    p_print->printDebug("test Debug level");
    p_print->printDebug("test Debug level", 42);
}

void testPrintInfo(PrintModule *p_print)
{
    p_print->printInfo("test Info level");
    p_print->printInfo("test Info level", 42);
}

void testPrintError(PrintModule *p_print)
{
    p_print->printError("test Error level");
    p_print->printError("test Error level", 42);
}

// =====================================
int main(int argc, const char **argv)
{
    PrintModule *p_print = PrintModule::getInstance();

    testPrintDebug(p_print);
    testPrintInfo(p_print);
    testPrintError(p_print);

    return 0;
}
