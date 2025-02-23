## Modern C++ Filesystem

```C++
#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem; // 推荐命名空间别名

// ------------跨平台策略
// 处理Windows长路径
#ifdef _WIN32
fs::path long_path = L"\\\\?\\C:\\VeryLongPath\\...";
#else
fs::path unix_path = "/mnt/data/long/unix/path";
#endif

// 统一处理路径分隔符
fs::path normalize_path(const fs::path& p) 
{
    return p.lexically_normal().make_preferred();
}

// ------------异常处理金字塔
try {
    auto space_info = fs::space("/data");
    std::cout << "Free space: " 
              << (space_info.available / (1024*1024)) 
              << " MB" << std::endl;
} catch (const fs::filesystem_error& excep) {
    std::cerr << "Filesystem error: " << excep.what() << "\n"
              << "Path1: " << excep.path1() << "\n"
              << "Path2: " << excep.path2() << std::endl;
} catch (const std::exception& excep) {
    std::cerr << "Standard exception: " << excep.what() << std::endl;
}

```

### 性能优化秘籍
|       |        |       |
| ----- | ------ | ----- |
|操作类型|时间复杂度|建议场景|
|exists() |O(1) |频繁状态检查|
|file_size() |O(1) |小文件快速获取大小|
|directory_scan |O(n) |结合缓存机制使用|
|recursive_scan |O(n^k) |避免深层目录的实时遍历|
|       |        |       |

```C++
// 缓存目录结构
class DirectoryCache {
    std::unordered_map<fs::path, fs::file_time_type> cache;
public:
    void update(const fs::path& dir) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            cache[entry.path()] = entry.last_write_time();
        }
    }
    bool is_modified(const fs::path& p) const {
        auto it = cache.find(p);
        return it == cache.end() || 
               fs::last_write_time(p) != it->second;
    }
};

// C++20为文件系统库带来重要增强
// 原子文件操作（C++20）
bool atomic_write(const fs::path& target, std::string_view content) {
    fs::path tmp = target;
    tmp += ".tmp";
    try {
        std::ofstream(tmp) << content;
        fs::rename(tmp, target); // 原子替换
        return true;
    } catch (...) {
        fs::remove(tmp);
        return false;
    }
}

```


### C++ Filesystem Reference
- [Filesystem since C++17](https://en.cppreference.com/w/cpp/filesystem)
- [Path class](https://en.cppreference.com/w/cpp/filesystem/path)
- [Non-member functions](https://en.cppreference.com/w/cpp/filesystem)
