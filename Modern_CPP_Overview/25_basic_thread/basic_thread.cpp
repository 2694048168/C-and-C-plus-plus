/**
 * @file basic_thread.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief Basic of Parallelism: thread (not process in Linux system programming)
 * @version 0.1
 * @date 2022-01-14
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <thread>

int main(int argc, char** argv)
{
    std::thread thread_1([](){
        std::cout << "Hello World form thread." << std::endl;
    });

    thread_1.join();
    
    return 0;
}
