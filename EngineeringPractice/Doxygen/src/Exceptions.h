/**
 * @file Exceptions.h
 * @brief Defines custom exceptions used throughout the application.
 */

#pragma once

#include <exception>
#include <string>

/**
 * @class NetworkException
 * @brief Exception for network-related errors.
 *
 * This exception is thrown when there are network failures such as disconnections or
 * timeouts.
 */
class NetworkException : public std::exception
{
public:
    /**
     * @brief Constructs a new Network Exception with the given message.
     * @param message The error message that describes the failure.
     */
    explicit NetworkException(const std::string &message)
        : msg(message)
    {
    }

    /**
     * @brief Returns a description of the exception.
     * @return A const character pointer to the error message.
     */
    virtual const char *what() const noexcept override
    {
        return msg.c_str();
    }

private:
    std::string msg;
};

/**
 * @brief Attempts to connect to a remote server.
 * @throws NetworkException If the connection cannot be established.
 */
void connectToServer()
{
    if (true) // connectionFailed
    {
        throw NetworkException("Unable to connect to the server.");
    }
}
