/**
 * @file 03_event_drive.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 以逸待劳: 事件驱动与异步编程
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 03_event_drive 03_event_drive.c
 *
 */

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#define MAX_EVENTS 64
#define PORT 8888
#define BUFFER_SIZE 1024

// 设置文件描述符为非阻塞模式（现代异步编程必备）
static int set_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1)
        return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// 创建并绑定TCP监听socket
static int create_listen_socket(int port)
{
    int sock = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (sock < 0)
    {
        perror("socket");
        return -1;
    }

    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY};
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("bind");
        close(sock);
        return -1;
    }

    if (listen(sock, SOMAXCONN) < 0)
    {
        perror("listen");
        close(sock);
        return -1;
    }
    return sock;
}

// 处理新连接：接受客户端，设置为非阻塞，加入epoll
static void accept_new_connection(int listen_fd, int epoll_fd)
{
    while (1)
    {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept4(listen_fd, (struct sockaddr *)&client_addr, &client_len, SOCK_NONBLOCK);
        if (client_fd == -1)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                break; // 没有更多连接
            perror("accept");
            break;
        }

        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
        printf("[以逸待劳] 新客户端 %s:%d，fd=%d\n", ip, ntohs(client_addr.sin_port), client_fd);

        struct epoll_event ev = {.events = EPOLLIN | EPOLLET, // 边缘触发，配合非阻塞I/O
                                 .data.fd = client_fd};
        if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev) == -1)
        {
            perror("epoll_ctl add client");
            close(client_fd);
        }
    }
}

// 处理客户端数据：读数据并原样写回（回射）
static void handle_client_data(int client_fd)
{
    char buffer[BUFFER_SIZE];
    ssize_t n = read(client_fd, buffer, sizeof(buffer) - 1);
    if (n <= 0)
    {
        if (n == 0)
        {
            printf("客户端 %d 断开连接\n", client_fd);
        }
        else if (errno != EAGAIN && errno != EWOULDBLOCK)
        {
            perror("read error");
        }
        close(client_fd);
        return;
    }

    buffer[n] = '\0';
    printf("[事件触发] 收到 %d 字节: %s", (int)n, buffer);
    // 原样写回（异步写也可能阻塞，为简洁示例忽略非阻塞写处理）
    write(client_fd, buffer, n);
}

// 处理标准输入命令（演示自定义事件）
static void handle_stdin_command(void)
{
    char cmd[128];
    if (fgets(cmd, sizeof(cmd), stdin))
    {
        cmd[strcspn(cmd, "\n")] = '\0';
        if (strcmp(cmd, "stats") == 0)
        {
            printf("[命令] 当前服务器运行中...\n");
        }
        else if (strcmp(cmd, "exit") == 0)
        {
            printf("[命令] 退出事件循环\n");
            exit(0);
        }
        else
        {
            printf("[命令] 未知命令: %s\n", cmd);
        }
    }
}

int main(void)
{
    // 忽略SIGPIPE，避免写断开连接时崩溃
    signal(SIGPIPE, SIG_IGN);

    int listen_fd = create_listen_socket(PORT);
    if (listen_fd < 0)
        return EXIT_FAILURE;

    // 创建epoll实例
    int epoll_fd = epoll_create1(0);
    if (epoll_fd < 0)
    {
        perror("epoll_create1");
        close(listen_fd);
        return EXIT_FAILURE;
    }

    // 将监听socket加入epoll
    struct epoll_event ev = {.events = EPOLLIN, .data.fd = listen_fd};
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev) == -1)
    {
        perror("epoll_ctl add listen");
        close(listen_fd);
        close(epoll_fd);
        return EXIT_FAILURE;
    }

    // 将标准输入也加入epoll（非阻塞需要额外处理，这里仅演示多事件源）
    // 注意：stdin默认行缓冲，epoll配合普通stdin效果不好，为演示多个fd，将stdin设为非阻塞较复杂，改用简单独立线程?
    // 简便方法：单独开一个线程读stdin，但为了纯epoll演示，我们使用另一个技巧：设置stdin非阻塞。
    // 更干净的做法：单独一个信号或timerfd，这里简化：不将stdin加入epoll，而是单独用select? 其实也可以用epoll配合stdin。
    // 为了演示“多个事件源”，我们创建一个timerfd或eventfd。为保持简洁，不增加复杂度，通过信号? 但代码要可运行。
    // 决定：再增加一个标准输入的事件，设置stdin为非阻塞（仅作演示，生产环境不推荐这样对stdin使用epoll）。
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
    struct epoll_event ev_stdin = {.events = EPOLLIN, .data.fd = STDIN_FILENO};
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, STDIN_FILENO, &ev_stdin);

    printf("以逸待劳事件驱动服务器启动，监听端口 %d\n", PORT);
    printf("可尝试: telnet localhost %d，输入任意字符会回射\n", PORT);
    printf("本进程标准输入支持命令: stats / exit\n");

    struct epoll_event events[MAX_EVENTS];
    // 主事件循环 —— “以逸待劳”核心：无事时阻塞，有事立即响应
    while (1)
    {
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1); // -1 表示无限等待
        if (nfds == -1)
        {
            if (errno == EINTR)
                continue;
            perror("epoll_wait");
            break;
        }

        for (int i = 0; i < nfds; ++i)
        {
            int fd = events[i].data.fd;
            if (fd == listen_fd)
            {
                // 有新连接到来
                accept_new_connection(listen_fd, epoll_fd);
            }
            else if (fd == STDIN_FILENO)
            {
                // 标准输入有命令
                handle_stdin_command();
            }
            else
            {
                // 客户端socket有数据可读
                if (events[i].events & EPOLLIN)
                {
                    handle_client_data(fd);
                }
                else if (events[i].events & (EPOLLHUP | EPOLLERR))
                {
                    printf("客户端 %d 异常关闭\n", fd);
                    close(fd);
                }
            }
        }
    }

    close(listen_fd);
    close(epoll_fd);
    return EXIT_SUCCESS;
}
