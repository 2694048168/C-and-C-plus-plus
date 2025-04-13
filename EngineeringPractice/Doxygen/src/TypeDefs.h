/**
 * @file TypeDefs.h
 * @brief Provides type definitions for common data structures in the application.
 */

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @typedef std::vector<std::string> StringList
 * @brief Represents a list of strings.
 */
typedef std::vector<std::string> StringList;

/**
 * @using StringStringMap = std::map<std::string, std::string>
 * @brief Defines a map from string to string for configuration settings.
 */
using StringStringMap = std::map<std::string, std::string>;

/**
 * @using EnumStringMap = std::unordered_map<std::string, std::string>
 * @brief Defines a map from Enum to string for configuration settings.
 */
using EnumStringMap = std::unordered_map<std::string, std::string>;
