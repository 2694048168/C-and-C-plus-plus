#pragma once

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using namespace log4cplus;
using namespace log4cplus::helpers;

// global object
extern Logger logger;

// define some macros for simplicity
#define LOG_TRACE(logEvent) LOG4CPLUS_TRACE(logger, logEvent)
#define LOG_DEBUG(logEvent) LOG4CPLUS_DEBUG(logger, logEvent)
#define LOG_DEBUG_F(...)    LOG4CPLUS_DEBUG_FMT(logger, __VA_ARGS__)
#define LOG_INFO(logEvent)  LOG4CPLUS_INFO(logger, logEvent)
#define LOG_WARN(logEvent)  LOG4CPLUS_WARN(logger, logEvent)
#define LOG_ERROR(logEvent) LOG4CPLUS_ERROR(logger, logEvent)
#define LOG_FATAL(logEvent) LOG4CPLUS_FATAL(logger, logEvent)

extern void initLogger(bool isDebug);
extern void shutDown();
extern void close();
