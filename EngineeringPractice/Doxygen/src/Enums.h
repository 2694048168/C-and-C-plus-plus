/**
 * @file Enums.h
 * @brief Defines traditional enumeration types used throughout the application.
 *
 * @brief Defines strongly-typed enumeration types used throughout the application.
 */

#pragma once

/**
 * @enum Color
 * @brief Represents color settings used in the application interface.
 */
enum Color
{
    Red,   /**< Indicates the color red. */
    Green, /**< Indicates the color green. */
    Blue   /**< Indicates the color blue. */
};

/**
 * @enum class ErrorCode
 * @brief Represents specific error codes for operations within the application.
 */
enum class ErrorCode
{
    None = 0,        /**< No error has occurred. */
    NetworkError,    /**< Error due to network failure. */
    DiskFull,        /**< Error due to insufficient disk space. */
    PermissionDenied /**< Error due to inadequate permissions. */
};
