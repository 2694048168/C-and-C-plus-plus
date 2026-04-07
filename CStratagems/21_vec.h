/**
 * @file 21_vec.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-04-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef VEC_H
#define VEC_H

#include <stdbool.h>
#include <stddef.h>

// 不透明类型：用户只能看到指针，无法解构内部
typedef struct Vector Vector;

// 创建和销毁
Vector *vec_create(void);
void    vec_destroy(Vector *v);

// 基本操作
bool   vec_push_back(Vector *v, int value);
bool   vec_pop_back(Vector *v, int *out);
bool   vec_get(const Vector *v, size_t index, int *out);
size_t vec_size(const Vector *v);
bool   vec_is_empty(const Vector *v);

// 遍历（回调函数，进一步解耦）
typedef void (*VecVisitFunc)(int value, void *ctx);
void vec_foreach(const Vector *v, VecVisitFunc func, void *ctx);

#endif