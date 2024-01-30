/**
 * @file main_client.cpp
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
void run_client(std::string_view host, const short port)
{
    try
    {
        asio::io_context        context;
        asio::ip::tcp::socket   tcp_socket(context);
        asio::ip::tcp::resolver resolver(context);
        asio::connect(tcp_socket, resolver.resolve({host.data(), std::to_string(port)}));

        while (true)
        {
            std::cout << "number [1-99]: ";

            int number;
            std::cin >> number;
            if (std::cin.fail() || number < 1 || number > 99)
                break;

            auto request = std::to_string(number);
            tcp_socket.write_some(asio::buffer(request, request.length()));

            std::array<char, 1024> reply;
            auto                   reply_length = tcp_socket.read_some(asio::buffer(reply, reply.size()));

            std::cout << "reply is: ";
            std::cout.write(reply.data(), reply_length);
            std::cout << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    run_client("localhost", 11234);

    return 0;
}
