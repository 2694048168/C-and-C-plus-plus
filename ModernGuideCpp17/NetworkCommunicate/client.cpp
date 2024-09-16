/**
 * @file server.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Linux OS 网络通信客户端
 * @version 0.1
 * @date 2024-01-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <arpa/inet.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ===================================
int main(int argc, const char **argv)
{
    // 1. 创建通信的套接字
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == -1)
    {
        perror("socket");
        exit(0);
    }

    // 2. 连接服务器
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(10000); // 大端端口
    inet_pton(AF_INET, "172.26.145.62", &addr.sin_addr.s_addr);

    int ret = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (ret == -1)
    {
        perror("connect");
        exit(0);
    }

    // 3. 和服务器端通信
    int number = 0;
    while (true)
    {
        // 发送数据
        char buf[1024];
        sprintf(buf, "你好, 服务器...%d\n", number++);
        write(fd, buf, strlen(buf) + 1);

        // 接收数据
        memset(buf, 0, sizeof(buf));
        int len = read(fd, buf, sizeof(buf));
        if (len > 0)
        {
            printf("服务器say: %s\n", buf);
        }
        else if (len == 0)
        {
            printf("服务器断开了连接...\n");
            break;
        }
        else
        {
            perror("read");
            break;
        }
        sleep(1); // 每隔1s发送一条数据
    }

    close(fd);

    return 0;
}
