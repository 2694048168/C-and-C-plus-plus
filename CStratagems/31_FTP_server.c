/**
 * @file 31_FTP_server.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 空城计: 服务器伪装防御-迷惑攻击者
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 * gcc 31_FTP_server.c -o 31_FTP_server.exe
 * clang 31_FTP_server.c -o 31_FTP_server.exe
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

#define PORT 8888
#define BACKLOG 10
#define FAKE_BANNER "220 (vsFTPd 2.3.4)\r\n"
#define FAKE_DIR "drwxr-xr-x 2 ftp ftp 4096 Apr  8 10:00 fake_dir\r\n"

// 处理客户端连接（发送伪造响应）
void handle_client(SOCKET client_fd)
{
    char buffer[256];
    // 发送伪造 banner（空城计：冒充有漏洞的旧版本）
    send(client_fd, FAKE_BANNER, (int)strlen(FAKE_BANNER), 0);

    while (1)
    {
        memset(buffer, 0, sizeof(buffer));
        int len = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        if (len <= 0)
            break;

        // 去除换行符
        buffer[strcspn(buffer, "\r\n")] = '\0';
        printf("[诱饵] 收到命令: %s\n", buffer);

        // 伪造响应（无论什么命令都返回看似正常的回复）
        if (_strnicmp(buffer, "USER", 4) == 0)
        {
            send(client_fd, "331 Please specify the password.\r\n", 36, 0);
        }
        else if (_strnicmp(buffer, "PASS", 4) == 0)
        {
            // 空城计：任何密码都允许登录
            send(client_fd, "230 Login successful.\r\n", 24, 0);
            // 发送伪造的目录列表
            send(client_fd, "150 Here comes the directory listing.\r\n", 40, 0);
            send(client_fd, FAKE_DIR, (int)strlen(FAKE_DIR), 0);
            send(client_fd, "226 Directory send OK.\r\n", 25, 0);
        }
        else if (_strnicmp(buffer, "QUIT", 4) == 0)
        {
            send(client_fd, "221 Goodbye.\r\n", 14, 0);
            break;
        }
        else
        {
            // 其他命令返回通用错误，但不断开连接
            send(client_fd, "500 Unknown command.\r\n", 22, 0);
        }
    }
    closesocket(client_fd);
}

int main(void)
{
    SetConsoleOutputCP(65001);

    WSADATA wsaData;
    // 初始化 WinSock
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        fprintf(stderr, "WSAStartup 失败\n");
        return EXIT_FAILURE;
    }

    SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == INVALID_SOCKET)
    {
        fprintf(stderr, "socket 创建失败: %d\n", WSAGetLastError());
        WSACleanup();
        return EXIT_FAILURE;
    }

    // 允许地址重用（Windows 中通过 setsockopt 设置 SO_REUSEADDR）
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR)
    {
        fprintf(stderr, "bind 失败: %d\n", WSAGetLastError());
        closesocket(listen_fd);
        WSACleanup();
        return EXIT_FAILURE;
    }

    if (listen(listen_fd, BACKLOG) == SOCKET_ERROR)
    {
        fprintf(stderr, "listen 失败: %d\n", WSAGetLastError());
        closesocket(listen_fd);
        WSACleanup();
        return EXIT_FAILURE;
    }

    printf("[空城计] 虚假 FTP 服务启动，端口 %d，返回 vsFTPd 2.3.4 标识（已弃用版本）\n", PORT);
    printf("[迷惑] 攻击者会以为存在漏洞，实际无任何后门。\n");

    while (1)
    {
        struct sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        SOCKET client_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd == INVALID_SOCKET)
        {
            fprintf(stderr, "accept 失败: %d\n", WSAGetLastError());
            continue;
        }

        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
        printf("[连接] 来自 %s:%d\n", ip, ntohs(client_addr.sin_port));

        handle_client(client_fd);
    }

    closesocket(listen_fd);
    WSACleanup();
    return 0;
}
