/**
 * @file 28_disguise_data.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 树上开花: 数据加密的伪装和增强
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 28_disguise_data.exe 28_disguise_data.c
 * clang -o 28_disguise_data.exe 28_disguise_data.c
 *
 */

#include <Windows.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ========== Base64 编码/解码（树上开花的伪装层） ==========
static const char b64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void base64_encode(const uint8_t *in, size_t len, char *out)
{
    size_t i = 0, j = 0;
    while (i < len)
    {
        uint32_t octet_a = i < len ? in[i++] : 0;
        uint32_t octet_b = i < len ? in[i++] : 0;
        uint32_t octet_c = i < len ? in[i++] : 0;
        uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;
        out[j++] = b64_table[(triple >> 18) & 0x3F];
        out[j++] = b64_table[(triple >> 12) & 0x3F];
        out[j++] = b64_table[(triple >> 6) & 0x3F];
        out[j++] = b64_table[triple & 0x3F];
    }
    // 处理填充
    size_t mod = len % 3;
    if (mod == 1)
    {
        out[j - 2] = '=';
        out[j - 1] = '=';
    }
    else if (mod == 2)
    {
        out[j - 1] = '=';
    }
    out[j] = '\0';
}

int base64_decode(const char *in, uint8_t *out, size_t out_max)
{
    size_t len = strlen(in);
    if (len % 4 != 0)
        return -1;
    size_t i = 0, j = 0;
    while (i < len && j < out_max)
    {
        uint32_t sextet_a = in[i] == '=' ? 0 : strchr(b64_table, in[i]) - b64_table;
        uint32_t sextet_b = in[i + 1] == '=' ? 0 : strchr(b64_table, in[i + 1]) - b64_table;
        uint32_t sextet_c = in[i + 2] == '=' ? 0 : strchr(b64_table, in[i + 2]) - b64_table;
        uint32_t sextet_d = in[i + 3] == '=' ? 0 : strchr(b64_table, in[i + 3]) - b64_table;
        uint32_t triple = (sextet_a << 18) | (sextet_b << 12) | (sextet_c << 6) | sextet_d;
        if (j < out_max)
            out[j++] = (triple >> 16) & 0xFF;
        if (j < out_max && in[i + 2] != '=')
            out[j++] = (triple >> 8) & 0xFF;
        if (j < out_max && in[i + 3] != '=')
            out[j++] = triple & 0xFF;
        i += 4;
    }
    return (int)j;
}

// ========== 简单 XOR 加密（树上开花的“花”即加密） ==========
void xor_crypt(uint8_t *data, size_t len, const char *key)
{
    size_t key_len = strlen(key);
    for (size_t i = 0; i < len; ++i)
    {
        data[i] ^= key[i % key_len];
    }
}

// ========== 伪装与恢复（树上开花的主逻辑） ==========
// 伪装：原始消息 -> XOR加密 -> Base64编码 -> 添加伪装的图像头
char *disguise_message(const char *plain, const char *key)
{
    size_t plain_len = strlen(plain);
    uint8_t *encrypted = malloc(plain_len);
    if (!encrypted)
        return NULL;
    memcpy(encrypted, plain, plain_len);
    xor_crypt(encrypted, plain_len, key);

    // Base64 编码：长度约为 (plain_len+2)/3*4 + 1
    size_t b64_len = ((plain_len + 2) / 3) * 4 + 1;
    char *b64 = malloc(b64_len);
    if (!b64)
    {
        free(encrypted);
        return NULL;
    }
    base64_encode(encrypted, plain_len, b64);
    free(encrypted);

    // 添加伪装头（假装是图片数据）
    const char *fake_header = "data:image/png;base64,";
    size_t header_len = strlen(fake_header);
    char *disguised = malloc(header_len + b64_len);
    if (!disguised)
    {
        free(b64);
        return NULL;
    }
    strcpy(disguised, fake_header);
    strcat(disguised, b64);
    free(b64);
    return disguised;
}

// 恢复：从伪装数据中提取真实消息
char *recover_message(const char *disguised, const char *key)
{
    // 查找 Base64 数据起始位置（跳过伪装头）
    const char *b64_start = strstr(disguised, "base64,");
    if (!b64_start)
        return NULL;
    b64_start += 7; // 跳过 "base64,"

    // 解码 Base64
    size_t b64_len = strlen(b64_start);
    uint8_t *decoded = malloc(b64_len); // 足够大
    if (!decoded)
        return NULL;
    int decoded_len = base64_decode(b64_start, decoded, b64_len);
    if (decoded_len < 0)
    {
        free(decoded);
        return NULL;
    }

    // XOR 解密
    xor_crypt(decoded, decoded_len, key);

    // 构造字符串（假设原始数据为文本）
    char *plain = malloc(decoded_len + 1);
    if (!plain)
    {
        free(decoded);
        return NULL;
    }
    memcpy(plain, decoded, decoded_len);
    plain[decoded_len] = '\0';
    free(decoded);
    return plain;
}

// ========== 主程序演示 ==========
int main(void)
{
    // 设置控制台输出编码为UTF-8
    SetConsoleOutputCP(CP_UTF8);

    const char *original = "机密信息: 明天凌晨三点发动攻击";
    const char *key = "MySecretKey";

    printf("原始消息: %s\n", original);

    // 树上开花：伪装消息
    char *disguised = disguise_message(original, key);
    if (!disguised)
    {
        fprintf(stderr, "伪装失败\n");
        return EXIT_FAILURE;
    }
    printf("\n伪装后的数据（看起来像图片链接）:\n%.80s...\n", disguised);

    // 恢复消息（验证）
    char *recovered = recover_message(disguised, key);
    if (recovered)
    {
        printf("\n恢复的消息: %s\n", recovered);
        free(recovered);
    }
    else
    {
        printf("恢复失败\n");
    }

    free(disguised);
    return 0;
}
