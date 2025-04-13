/**
 * @file ConcreteImplementation.cpp
 * @brief Implementation of the IProcessor interface for specific data processing.
 */

#include "AbstractBase.h"

#include <iostream>

/**
 * @class ConcreteProcessor
 * @brief Implements the IProcessor interface to process string data.
 *
 * ConcreteProcessor provides a specific algorithm to handle string manipulation
 * and processing based on the IProcessor interface.
 */
class ConcreteProcessor : public IProcessor
{
public:
    /**
     * @brief Initializes the processor specifically for string data.
     * @return True if initialization is successful, false otherwise.
     */
    bool initialize() override
    {
        // Specific initialization logic for string processing
        return true;
    }

    /**
     * @brief Processes the input string data and modifies it.
     * @param data The string data to be processed.
     * @return True if processing is successful, false otherwise.
     */
    bool process(const std::string &data) override
    {
        std::cout << "Processing data: " << data << std::endl;
        // Example processing logic
        return true;
    }

    /**
     * @brief Overloaded version of process specifically for integer data.
     * @param data The integer data to be processed.
     * @return True if processing is successful, false otherwise.
     */
    bool process(int data)
    {
        std::cout << "Processing integer data: " << data << std::endl;
        // Processing logic for integers
        return true;
    }

    /**
     * @brief Cleans up any resources used by the string processor.
     */
    void cleanup() override
    {
        // Cleanup logic
    }

    /**
     * @brief Virtual destructor ensures proper cleanup of derived classes.
     */
    virtual ~ConcreteProcessor() {}
};
