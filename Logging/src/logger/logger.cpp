// #define UNICODE
#include "logger.hpp"

#include "log4cplus/configurator.h"
#include "log4cplus/layout.h"
#include "log4cplus/logger.h"

Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("logmain"));

void initLogger(bool isDebug)
{
    if (isDebug)
    {
        PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("./log4cplus_d.conf"));
    }
    else
    {
        PropertyConfigurator::doConfigure(LOG4CPLUS_TEXT("./log4cplus.conf"));
    }
}

void shutDown()
{
    log4cplus::Logger::shutdown();
}

void close()
{
    log4cplus::deinitialize();
}
