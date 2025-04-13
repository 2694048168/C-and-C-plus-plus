/**
 * @file BusinessLogic.h
 * @brief Defines the BusinessLogic class and associated functionality.
 */

#pragma once

#include <string>
#include <vector>

/**
 * @class BusinessLogic
 * @brief Manages the main operations of the application.
 *
 * This class encapsulates all the business logic of the application. It handles user
 * interactions, data processing, and maintains the state of the application.
 */
class BusinessLogic
{
public:
    /**
     * @brief Initializes the business logic layer.
     * @return True if initialization is successful, false otherwise.
     */
    static bool initialize();

    /**
     * @brief Runs the main processing loop of the business logic.
     */
    static void run();

private:
    std::vector<std::string> data; /**< Maintains internal state of processed data. */
};
