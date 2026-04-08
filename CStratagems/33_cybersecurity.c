/**
 * @file 33_cybersecurity.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 苦肉计: 主动暴露与信息收集-自我牺牲引诱攻击者
 * @version 0.1
 * @date 2026-04-08
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <arpa/inet.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define PORT 8888
#define BUFFER_SIZE 64 // 看似很小的缓冲区（诱饵）
#define LOG_FILE "attacks.log"

// 记录攻击信息（苦肉计的核心：收集情报）
void log_attack(const char *client_ip, const char *payload)
{
    FILE *fp = fopen(LOG_FILE, "a");
    if (!fp)
        return;
    time_t now = time(NULL);
    fprintf(fp, "[%s] 攻击来源: %s | 载荷长度: %zu | 内容: %s\n", ctime(&now), client_ip, strlen(payload), payload);
    fclose(fp);
    printf("[苦肉计] 已记录来自 %s 的攻击尝试\n", client_ip);
}

// 脆弱的处理函数（看起来危险，实则安全）
void handle_request(const char *client_ip, const char *user_input)
{
    char small_buf[BUFFER_SIZE]; // 只有 64 字节（诱饵）

    // 苦肉计：主动检查长度，若超长则记录攻击并拒绝，而非真正溢出
    if (strlen(user_input) >= BUFFER_SIZE)
    {
        log_attack(client_ip, user_input);
        // 返回虚假的“崩溃”消息，迷惑攻击者
        send(client_socket, "Service crash (core dumped)\n", 28, 0);
        return;
    }

    // 此处看起来是不安全的 strcpy，但实际因上述检查而安全
    strcpy(small_buf, user_input);
    printf("[服务] 处理正常请求: %s\n", user_input);
    // 正常响应
    send(client_socket, "OK\n", 3, 0);
}

// 模拟客户端连接处理（省略完整socket代码以聚焦逻辑）
void process_client(int client_fd, const char *ip)
{
    char buffer[1024];
    ssize_t len = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (len > 0)
    {
        buffer[len] = '\0';
        handle_request(ip, buffer);
    }
    close(client_fd);
}

int main(void)
{
    // 忽略 SIGPIPE，防止写断开连接时崩溃
    signal(SIGPIPE, SIG_IGN);

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0)
    {
        perror("socket");
        exit(1);
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(PORT), .sin_addr.s_addr = INADDR_ANY};
    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("bind");
        close(listen_fd);
        exit(1);
    }
    if (listen(listen_fd, 10) < 0)
    {
        perror("listen");
        close(listen_fd);
        exit(1);
    }

    printf("[苦肉计] 诱饵服务启动，端口 %d，缓冲区大小 %d 字节\n", PORT, BUFFER_SIZE);
    printf("[欺骗] 攻击者会以为存在栈溢出漏洞，实际已内置防护和日志\n");

    while (1)
    {
        struct sockaddr_in client_addr;
        socklen_t clen = sizeof(client_addr);
        int client_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &clen);
        if (client_fd < 0)
        {
            perror("accept");
            continue;
        }

        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
        process_client(client_fd, ip);
    }
    close(listen_fd);
    return 0;
}
