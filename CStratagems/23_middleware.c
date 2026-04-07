/**
 * @file 23_middleware.c
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 假道伐虎: 模块间的间接访问与解耦策略
 * @version 0.1
 * @date 2026-04-07
 *
 * @copyright Copyright (c) 2026
 *
 * gcc -o 23_middleware.exe 23_middleware.c
 * clang -o 23_middleware.exe 23_middleware.c
 *
 */

#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ========== 假道：存储接口（间接层） ==========
// 存储操作函数指针类型
typedef struct Storage Storage; // 不透明前向声明

// 存储虚函数表（C语言模拟接口）
typedef struct
{
    int (*open)(Storage *self, const char *path);
    int (*write)(Storage *self, const char *data);
    void (*close)(Storage *self);
    void (*destroy)(Storage *self);
} StorageVTable;

// 存储对象结构（包含虚表指针）
struct Storage
{
    StorageVTable *vtable;
    // 具体子类可扩展私有数据
};

// 辅助宏调用虚函数
#define STORAGE_OPEN(s, p) ((s)->vtable->open((s), (p)))
#define STORAGE_WRITE(s, d) ((s)->vtable->write((s), (d)))
#define STORAGE_CLOSE(s) ((s)->vtable->close((s)))
#define STORAGE_DESTROY(s) ((s)->vtable->destroy((s)))

// ========== 具体实现1：文件存储（真实的老虎） ==========
typedef struct
{
    Storage base; // 继承基类
    FILE *fp;
} FileStorage;

static int file_open(Storage *self, const char *path)
{
    FileStorage *fs = (FileStorage *)self;
    fs->fp = fopen(path, "a");
    return fs->fp ? 0 : -1;
}

static int file_write(Storage *self, const char *data)
{
    FileStorage *fs = (FileStorage *)self;
    if (!fs->fp)
        return -1;
    return fprintf(fs->fp, "%s\n", data) < 0 ? -1 : 0;
}

static void file_close(Storage *self)
{
    FileStorage *fs = (FileStorage *)self;
    if (fs->fp)
        fclose(fs->fp);
    fs->fp = NULL;
}

static void file_destroy(Storage *self)
{
    file_close(self);
    free(self);
}

static StorageVTable file_vtable = {
    .open = file_open, .write = file_write, .close = file_close, .destroy = file_destroy};

Storage *create_file_storage(void)
{
    FileStorage *fs = (FileStorage *)malloc(sizeof(FileStorage));
    fs->base.vtable = &file_vtable;
    fs->fp = NULL;
    return (Storage *)fs;
}

// ========== 具体实现2：内存存储（另一只老虎，模拟数据库） ==========
typedef struct
{
    Storage base;
    char **buffer;
    size_t count;
    size_t capacity;
} MemStorage;

static int mem_open(Storage *self, const char *path)
{
    (void)path; // 内存存储忽略路径
    MemStorage *ms = (MemStorage *)self;
    ms->buffer = NULL;
    ms->count = 0;
    ms->capacity = 0;
    return 0;
}

static int mem_write(Storage *self, const char *data)
{
    MemStorage *ms = (MemStorage *)self;
    if (ms->count >= ms->capacity)
    {
        size_t new_cap = ms->capacity ? ms->capacity * 2 : 4;
        char **new_buf = realloc(ms->buffer, new_cap * sizeof(char *));
        if (!new_buf)
            return -1;
        ms->buffer = new_buf;
        ms->capacity = new_cap;
    }
    ms->buffer[ms->count] = strdup(data);
    ms->count++;
    return 0;
}

static void mem_close(Storage *self)
{
    MemStorage *ms = (MemStorage *)self;
    for (size_t i = 0; i < ms->count; ++i)
        free(ms->buffer[i]);
    free(ms->buffer);
    ms->buffer = NULL;
    ms->count = ms->capacity = 0;
}

static void mem_destroy(Storage *self)
{
    mem_close(self);
    free(self);
}

static void mem_print(Storage *self)
{
    MemStorage *ms = (MemStorage *)self;
    printf("内存存储内容（%zu条）:\n", ms->count);
    for (size_t i = 0; i < ms->count; ++i)
    {
        printf("  %s\n", ms->buffer[i]);
    }
}

static StorageVTable mem_vtable = {.open = mem_open, .write = mem_write, .close = mem_close, .destroy = mem_destroy};

Storage *create_mem_storage(void)
{
    MemStorage *ms = (MemStorage *)malloc(sizeof(MemStorage));
    ms->base.vtable = &mem_vtable;
    ms->buffer = NULL;
    ms->count = ms->capacity = 0;
    return (Storage *)ms;
}

// ========== 数据采集模块（只依赖 Storage 接口，假道伐虎） ==========
typedef struct
{
    Storage *storage; // 通过接口间接访问具体存储
    char name[32];
} DataCollector;

void collector_init(DataCollector *dc, Storage *storage, const char *name)
{
    dc->storage = storage;
    strncpy(dc->name, name, sizeof(dc->name) - 1);
}

int collector_start(DataCollector *dc, const char *log_path)
{
    if (!dc->storage)
        return -1;
    if (STORAGE_OPEN(dc->storage, log_path) != 0)
        return -1;
    return 0;
}

void collect_data(DataCollector *dc, int value)
{
    char timestamp[32];
    time_t t = time(NULL);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&t));

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "[%s] 采集器 %s 获得数据: %d", timestamp, dc->name, value);
    STORAGE_WRITE(dc->storage, buffer);
}

void collector_stop(DataCollector *dc)
{
    if (dc->storage)
        STORAGE_CLOSE(dc->storage);
}

// ========== 主程序演示 ==========
int main(void)
{
    SetConsoleOutputCP(CP_UTF8);

    // 创建两种存储（两只老虎）
    Storage *file_storage = create_file_storage();
    Storage *mem_storage = create_mem_storage();

    // 创建数据采集器，分别使用不同存储（假道）
    DataCollector dc_file, dc_mem;
    collector_init(&dc_file, file_storage, "文件采集器");
    collector_init(&dc_mem, mem_storage, "内存采集器");

    // 启动采集（间接调用存储的 open）
    collector_start(&dc_file, "data.log");
    collector_start(&dc_mem, NULL); // 内存存储忽略路径

    // 采集数据
    for (int i = 1; i <= 5; ++i)
    {
        collect_data(&dc_file, i * 10);
        collect_data(&dc_mem, i * 10);
    }

    // 停止采集（关闭存储）
    collector_stop(&dc_file);
    collector_stop(&dc_mem);

    // 对于内存存储，可以查看内容（这是存储本身提供的额外操作，采集器未使用）
    printf("\n=== 假道伐虎效果 ===\n");
    mem_print(mem_storage);

    // 销毁存储
    STORAGE_DESTROY(file_storage);
    STORAGE_DESTROY(mem_storage);

    printf("\n假道伐虎核心思想：\n");
    printf("- 采集模块不直接依赖文件或数据库，只依赖 Storage 接口\n");
    printf("- 通过函数指针表（虚表）实现多态，运行时决定具体存储\n");
    printf("- 更换存储方式（从文件到内存）无需修改采集模块代码\n");
    printf("- 降低了模块间的耦合，提高了可扩展性\n");

    return 0;
}
