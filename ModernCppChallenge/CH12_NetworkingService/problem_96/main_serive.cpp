/**
 * @file main_serive.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Client-server Fizz-Buzz
 * @version 0.1
 * @date 2024-01-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <array>
#include <iostream>
#include <string_view>

#define ASIO_STANDALONE
#include "asio/asio.hpp"

/**
 * @brief Client-server Fizz-Buzz
 * 
 * Write a client-server application that can be used for playing the Fizz-Buzz game.
 * The client sends numbers to the server that answer back with fizz, buzz, fizz-buzz,
 * or the number itself, according to the game rules. 
 * Communication between the client and the server must be done over TCP. 
 * The server should run indefinitely. The client should run as long as the user enters
 * numbers between 1 and 99. Fizz-Buzz is a game for children, intended to teach them 
 * arithmetic division. A player must say a number and another player should answer with:
 * 1. Fizz, if the number is divisible by 3
 * 2. Buzz, if the number is divisible by 5
 * 3. Fizz-buzz, if the number is divisible by both 3 and 5
 * The number itself in all other cases
 * 
 */

/**
 * @brief Solution: Asio C++ Library
 https://think-async.com/Asio/
 https://github.com/chriskohlhoff/asio/
------------------------------------------------------ */
std::string fizzbuzz(const int number)
{
    if (number != 0)
    {
        auto m3 = number % 3;
        auto m5 = number % 5;
        if (m3 == 0 && m5 == 0)
            return "fizzbuzz";
        else if (m5 == 0)
            return "buzz";
        else if (m3 == 0)
            return "fizz";
    }

    return std::to_string(number);
}

class session : public std::enable_shared_from_this<session>
{
public:
    session(asio::ip::tcp::socket socket)
        : tcp_socket(std::move(socket))
    {
    }

    void start()
    {
        read();
    }

private:
    void read()
    {
        auto self(shared_from_this());

        tcp_socket.async_read_some(asio::buffer(data, data.size()),
                                   [this, self](const std::error_code ec, const std::size_t length)
                                   {
                                       if (!ec)
                                       {
                                           auto number = std::string(data.data(), length);
                                           auto result = fizzbuzz(std::atoi(number.c_str()));

                                           std::cout << number << " -> " << result << std::endl;

                                           write(result);
                                       }
                                   });
    }

    void write(std::string_view response)
    {
        auto self(shared_from_this());

        tcp_socket.async_write_some(asio::buffer(response.data(), response.length()),
                                    [this, self](const std::error_code ec, const std::size_t)
                                    {
                                        if (!ec)
                                        {
                                            read();
                                        }
                                    });
    }

    std::array<char, 1024> data;
    asio::ip::tcp::socket  tcp_socket;
};

class server
{
public:
    server(asio::io_context &context, const short port)
        : tcp_acceptor(context, asio::ip::tcp::endpoint(asio::ip::tcp::v4(), port))
        , tcp_socket(context)
    {
        std::cout << "server running on port " << port << std::endl;

        accept();
    }

private:
    void accept()
    {
        tcp_acceptor.async_accept(tcp_socket,
                                  [this](std::error_code ec)
                                  {
                                      if (!ec)
                                      {
                                          std::make_shared<session>(std::move(tcp_socket))->start();
                                      }

                                      accept();
                                  });
    }

    asio::ip::tcp::acceptor tcp_acceptor;
    asio::ip::tcp::socket   tcp_socket;
};

void run_server(const short port)
{
    try
    {
        asio::io_context context;

        server srv(context, port);

        context.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    run_server(11234);

    return 0;
}
