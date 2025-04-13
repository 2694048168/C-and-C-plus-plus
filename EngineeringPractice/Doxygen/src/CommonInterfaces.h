/**
 * @file CommonInterfaces.h
 * @brief Defines interfaces and common structures used across the application.
 */

#pragma once

#include <string>

/**
 * @interface IDataProcessor
 * @brief Interface for data processing across different modules.
 *
 * This interface defines a set of methods for data processing that all concrete
 * processors must implement, ensuring consistency across different parts of the application.
 */
class IDataProcessor
{
public:
    /**
     * @brief Process data and return the results.
     * @param data The data to be processed.
     * @return Processed data.
     * @throw DataProcessingException if data cannot be processed.
     */
    virtual std::string process(const std::string &data) = 0;

    virtual ~IDataProcessor() = default; // Ensure proper cleanup with a virtual destructor.
};

/**
 * @struct DataPacket
 * @brief Structure to hold data for transmission.
 *
 * This structure is used throughout the application to transfer data between
 * components. It encapsulates all the necessary information in one place.
 */
struct DataPacket
{
    std::string sender;   /**< Identifier of the sender. */
    std::string receiver; /**< Identifier of the receiver. */
    std::string content;  /**< The actual content of the message. */
};
