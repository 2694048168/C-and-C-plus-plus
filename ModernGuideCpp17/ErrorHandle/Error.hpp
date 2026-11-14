/**
 * @file Error.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-11-14
 * 
 * @copyright Copyright (c) 2026
 * 
 * Provide a modern error handling system via Result<T, E>
 * It encourages explicit error handling and provides
 * a type-safe way to propagate errors through the call stack.
 */

#pragma once

#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace Ithaca {

/**
 * @brief Standard error codes used throughout Project
 * 
 */
enum class ErrorCode : uint32_t
{
    // Success
    Ok = 0,

    // Memory errors (1-99)
    OutOfMemory      = 1,
    InvalidAlignment = 2,
    NullPointer      = 3,
    BufferOverflow   = 4,

    // I/O errors (100-199)
    FileNotFound      = 100,
    FileReadError     = 101,
    FileWriteError    = 102,
    InvalidFileFormat = 103,

    // Operator errors (200-499)
    InvalidOperator        = 400,
    OperatorNotImplemented = 401,
    InvalidOperand         = 402,

    // Module errors (300-899)

    // Generic errors (900-999)
    InvalidArgument = 900,
    NotImplemented  = 901,
    InternalError   = 902,
    Unknown         = 999,
};

/**
 * @brief Convert ErrorCode to string representation
 * 
 * @param ErrorCode code 
 * @return constexpr std::string_view 
 */
constexpr std::string_view error_code_str(ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::Ok:
        return "Ok";

    // Memory
    case ErrorCode::OutOfMemory:
        return "OutOfMemory";
    case ErrorCode::InvalidAlignment:
        return "InvalidAlignment";
    case ErrorCode::NullPointer:
        return "NullPointer";
    case ErrorCode::BufferOverflow:
        return "BufferOverflow";

    // I/O
    case ErrorCode::FileNotFound:
        return "FileNotFound";
    case ErrorCode::FileReadError:
        return "FileReadError";
    case ErrorCode::FileWriteError:
        return "FileWriteError";
    case ErrorCode::InvalidFileFormat:
        return "InvalidFileFormat";

    // Operator
    case ErrorCode::InvalidOperator:
        return "InvalidOperator";
    case ErrorCode::OperatorNotImplemented:
        return "OperatorNotImplemented";
    case ErrorCode::InvalidOperand:
        return "InvalidOperand";

    // Module

    // Generic
    case ErrorCode::InvalidArgument:
        return "InvalidArgument";
    case ErrorCode::NotImplemented:
        return "NotImplemented";
    case ErrorCode::InternalError:
        return "InternalError";
    case ErrorCode::Unknown:
        return "Unknown";

    default:
        return "UnknownErrorCode";
    }
}

/**
 * @brief Represents an error with code and optional message
 * 
 */
class Error
{
private:
    ErrorCode   m_code;
    std::string m_message;

public:
    /**
     * @brief Get the error code
     * 
     * @return constexpr ErrorCode 
     */
    [[nodiscard]] constexpr ErrorCode code() const noexcept
    {
        return m_code;
    }

    /**
     * @brief Get the error message
     * 
     * @return * const std::string& 
     */
    [[nodiscard]] const std::string &message() const noexcept
    {
        return m_message;
    }

    /**
     * @brief Check if this is an Ok (no error)
     * 
     * @return true 
     * @return false 
     */
    [[nodiscard]] constexpr bool is_ok() const noexcept
    {
        return m_code == ErrorCode::Ok;
    }

    /**
     * @brief Check if this is an error (not Ok)
     * 
     * @return true 
     * @return false 
     */
    [[nodiscard]] constexpr bool is_err() const noexcept
    {
        return m_code != ErrorCode::Ok;
    }

    /**
     * @brief Get full error description
     * 
     * @return std::string 
     */
    [[nodiscard]] std::string to_string() const
    {
        if (is_ok())
        {
            return "Ok";
        }
        if (m_message.empty())
        {
            return std::string(error_code_str(m_code));
        }
        return std::string(error_code_str(m_code)) + ": " + m_message;
    }

    /**
     * @brief Boolean conversion for easy checking
     * 
     * @return true 
     * @return false 
     */
    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return is_ok();
    }

    /* Rule of Zero for modern C++ */
public:
    explicit Error() noexcept
        : m_code(ErrorCode::Ok)
        , m_message(error_code_str(ErrorCode::Ok))
    {
    }

    explicit Error(ErrorCode code) noexcept
        : m_code(code)
        , m_message(error_code_str(code))
    {
    }

    Error(ErrorCode code, std::string message)
        : m_code(code)
        , m_message(std::move(message))
    {
    }

    ~Error() = default;

    // copy-constructor
    Error(const Error &obj) = default; // delete
    // copy-assignment
    Error &operator=(const Error &obj) = default; // delete
    // move-constructor
    Error(Error &&obj) = default; // delete
    // move-assignment
    Error &operator=(Error &&obj) = default; // delete
};

/**
 * @brief A Result type via Result<T, E>
 * 
 * Result<T, E> represents either success (containing a value of type T) or
 * failure (containing an error of type E). This forces explicit error handling
 * and makes error propagation visible in the type system.
 * 
 * @tparam T The type of the success value
 * @tparam E The type of the error (defaults to Error)
 */
template<typename T, typename E = Error>
class [[nodiscard]] Result
{
private:
    std::variant<T, E> m_storage;

public:
    using value_type = T;
    using error_type = E;

    /**
     * @brief Construct a successful Result with a value
     */
    constexpr Result(T value)
        : m_storage(std::move(value))
    {
    }

    /**
     * @brief Construct a failed Result with an error
     */
    constexpr Result(E error)
        : m_storage(std::move(error))
    {
    }

    /**
     * @brief Check if Result contains a value (is Ok)
     */
    [[nodiscard]] constexpr bool is_ok() const noexcept
    {
        return std::holds_alternative<T>(m_storage);
    }

    /**
     * @brief Check if Result contains an error (is Err)
     */
    [[nodiscard]] constexpr bool is_err() const noexcept
    {
        return std::holds_alternative<E>(m_storage);
    }

    /**
     * @brief Boolean conversion (true if Ok)
     */
    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return is_ok();
    }

    /**
     * @brief Get the value (must be Ok, otherwise undefined behavior)
     */
    [[nodiscard]] constexpr T &value() &
    {
        return std::get<T>(m_storage);
    }

    [[nodiscard]] constexpr const T &value() const &
    {
        return std::get<T>(m_storage);
    }

    [[nodiscard]] constexpr T &&value() &&
    {
        return std::get<T>(std::move(m_storage));
    }

    [[nodiscard]] constexpr const T &&value() const &&
    {
        return std::get<T>(std::move(m_storage));
    }

    /**
     * @brief Get the error (must be Err, otherwise undefined behavior)
     */
    [[nodiscard]] constexpr E &error() &
    {
        return std::get<E>(m_storage);
    }

    [[nodiscard]] constexpr const E &error() const &
    {
        return std::get<E>(m_storage);
    }

    [[nodiscard]] constexpr E &&error() &&
    {
        return std::get<E>(std::move(m_storage));
    }

    [[nodiscard]] constexpr const E &&error() const &&
    {
        return std::get<E>(std::move(m_storage));
    }

    /**
     * @brief Get value or return default if error
     */
    template<typename U>
    [[nodiscard]] constexpr T value_or(U &&default_value) const &
    {
        return is_ok() ? value() : static_cast<T>(std::forward<U>(default_value));
    }

    template<typename U>
    [[nodiscard]] constexpr T value_or(U &&default_value) &&
    {
        return is_ok() ? std::move(value()) : static_cast<T>(std::forward<U>(default_value));
    }

    /**
     * @brief Unwrap value (panics if error)
     */
    [[nodiscard]] constexpr T unwrap() &&
    {
        if (is_err())
        {
            // In production code, this should use a proper assertion/panic mechanism
            std::terminate();
        }
        return std::move(value());
    }

    /**
     * @brief Unwrap error (panics if ok)
     */
    [[nodiscard]] constexpr E unwrap_err() &&
    {
        if (is_ok())
        {
            std::terminate();
        }
        return std::move(error());
    }
};

} // namespace Ithaca
