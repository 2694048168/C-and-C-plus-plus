/**
 * @file 02_use_lib_system.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 借刀杀人: 善用库(标准库和第三方库)函数与系统调用
 * @version 0.1
 * @date 2026-04-05
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 02_use_lib_system.exe 02_use_lib_system.c
 * clang -o 02_use_lib_system.exe 02_use_lib_system.c
 *
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 单词及其出现次数
typedef struct
{
    char *word;
    size_t count;
} WordFreq;

// 比较函数：按频率降序，频率相同按字典序升序（用于 qsort）
int compare_freq_desc(const void *a, const void *b)
{
    const WordFreq *wa = (const WordFreq *)a;
    const WordFreq *wb = (const WordFreq *)b;
    if (wa->count != wb->count)
        return (wb->count > wa->count) ? 1 : -1; // 降序
    return strcmp(wa->word, wb->word);           // 升序
}

// 将单词转换为小写（借用 ctype.h 的 tolower）
void to_lowercase(char *str)
{
    for (; *str; ++str)
        *str = (char)tolower((unsigned char)*str);
}

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    FILE *fp = fopen(argv[1], "r"); // 借：fopen
    if (!fp)
    {
        perror("Open file failed"); // 借：perror
        return EXIT_FAILURE;
    }

    WordFreq *words = NULL;
    size_t capacity = 0; // 动态数组容量
    size_t total = 0;    // 已存储的不同单词数

    char line[4096];
    const char *delimiters = " \t\n\r,.;:!?\"'()[]{}<>";

    // 借：fgets 逐行读取，strtok 分割单词
    while (fgets(line, sizeof(line), fp))
    {
        char *token = strtok(line, delimiters);
        while (token)
        {
            // 借：strdup（POSIX）复制单词（非标准但广泛支持，也可用 malloc+strcpy）
            char *word = strdup(token);
            if (!word)
            {
                fprintf(stderr, "Out of Memory\n");
                fclose(fp);
                return EXIT_FAILURE;
            }
            to_lowercase(word); // 统一小写

            // 查找是否已有该单词（借：strcmp）
            size_t i;
            for (i = 0; i < total; ++i)
            {
                if (strcmp(words[i].word, word) == 0)
                {
                    free(word); // 重复单词，释放临时副本
                    words[i].count++;
                    break;
                }
            }
            if (i == total)
            { // 新单词
                if (total == capacity)
                {
                    capacity = capacity ? capacity * 2 : 16;
                    // 借：realloc 动态扩容
                    WordFreq *new_arr = realloc(words, capacity * sizeof(WordFreq));
                    if (!new_arr)
                    {
                        fprintf(stderr, "Out of Memory\n");
                        free(word);
                        fclose(fp);
                        return EXIT_FAILURE;
                    }
                    words = new_arr;
                }
                words[total].word = word; // 直接使用 strdup 的副本
                words[total].count = 1;
                total++;
            }
            token = strtok(NULL, delimiters);
        }
    }
    fclose(fp); // 借：fclose

    // 借：qsort 排序
    qsort(words, total, sizeof(WordFreq), compare_freq_desc);

    // 输出结果（借：printf）
    printf("Word frequency statistics(in descending order):\n");
    for (size_t i = 0; i < total && i < 30; ++i)
    { // 只显示前30个
        printf("%4zu  %s\n", words[i].count, words[i].word);
    }
    if (total > 30)
        printf("... total %zu different words\n", total);

    // 释放内存
    for (size_t i = 0; i < total; ++i)
        free(words[i].word);
    free(words);

    return EXIT_SUCCESS;
}
