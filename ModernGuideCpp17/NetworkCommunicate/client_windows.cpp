/**
 * @file server.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Windows OS 网络通信客户端
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// 使用包含的头文件
#include <winsock2.h>
#include <ws2tcpip.h>

// 使用的套接字库
// ws2_32.dll

// ===================================
int main(int argc, const char **argv)
{
    // 0. 初始化Winsock库
    WSAData wsa;
    // 初始化套接字库
    WSAStartup(MAKEWORD(2, 2), &wsa);

    // 1. 创建通信的套接字
    SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET)
    {
        std::cout << "create listen socket NOT successfully\n";
        return -1;
    }

    // 2. 连接服务器
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(10000); // 大端端口
    inet_pton(AF_INET, "172.31.98.10", &addr.sin_addr.s_addr);

    int ret = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (ret == SOCKET_ERROR)
    {
        std::cout << "connect SOCKET_ERROR\n";
        exit(0);
    }

    // 3. 和服务器端通信
    int number = 0;
    while (true)
    {
        // 发送数据
        char buf[1024];
        sprintf(buf, "你好, 服务器...%d\n", number++);
        send(fd, buf, strlen(buf) + 1, 0);

        // 接收数据
        memset(buf, 0, sizeof(buf));
        int len = recv(fd, buf, sizeof(buf), 0);
        if (len > 0)
        {
            std::cout << "服务器say: " << buf << '\n';
        }
        else if (len == 0)
        {
            std::cout << "服务器断开了连接...\n";
            break;
        }
        else
        {
            std::cout << "read SOCKET_ERROR\n";
            break;
        }
        Sleep(1000); // 每隔1s发送一条数据
    }

    ret = closesocket(fd);
    if (ret == SOCKET_ERROR)
    {
        std::cout << "close communication socket NOT successfully\n";
        return -1;
    }

    // 注销Winsock相关库
    WSACleanup();

    return 0;
}
