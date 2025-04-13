/**
 * @file AbstractBase.h
 * @brief Defines the interface for all concrete implementations in the application.
 */

#pragma once

#include <string>

/**
 * @class IProcessor
 * @brief Interface for processing data within the application.
 *
 * This abstract class defines the standard operations required for data processing
 * across different modules of the application. Concrete classes must implement all
 * the pure virtual functions defined here.
 */
class IProcessor
{
public:
    /**
     * @brief Initializes the processor with necessary resources.
     * @return True if initialization is successful, false otherwise.
     * @note This method must be implemented by derived classes.
     */
    virtual bool initialize() = 0;

    /**
     * @brief Processes the data and produces results.
     * @param data Input data to be processed.
     * @return True if data processing is successful, false otherwise.
     * @note This method must be implemented by derived classes.
     */
    virtual bool process(const std::string &data) = 0;

    /**
     * @brief Cleans up resources used by the processor.
     * @note This method must be implemented by derived classes.
     */
    virtual void cleanup() = 0;

    /**
     * @todo Implement logging of processing errors and status.
     */

    virtual ~IProcessor() {} // Virtual destructor for safe polymorphic use.
};
