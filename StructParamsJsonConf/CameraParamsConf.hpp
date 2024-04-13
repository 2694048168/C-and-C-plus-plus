#pragma once

/**
 * @file CameraParamsConf.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 将系统的参数配置序列化保存到json本地文件, 系统启动时候加载对应配置参数文件
 * @version 0.1
 * @date 2024-04-12
 * 
 * @copyright Copyright (c) 2024
 * 
 * step 1. 数据结构体
 * step 2. 从结构体数据到 json 数据
 * step 3. 从 json 数据到结构体数据
 * step 4. 保存到 json 本地文件
 * step 5. 从 json 本地文件中加载
 * step 6. 宏定义提供函数接口供外部使用
 * step 7. 将该结构体参数配置为一个class
 * 
 */

#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <string>

// 1. 数据结构体
struct CameraParams
{
    std::string  camera_name;
    std::string  camera_serial_number;
    bool         camera_status;
    unsigned int camera_width;
    unsigned int camera_height;
    float        camera_exposure;

    // 构造函数
    CameraParams()
    {
        camera_name          = "CCD";
        camera_serial_number = "AT9023873";
        camera_status        = true;
        camera_width         = 16384;
        camera_height        = 3000;
        camera_exposure      = 15;
    }
};

// 2. 从结构体数据到 json 数据
inline void to_json(nlohmann::json &j, const CameraParams &p)
{
    j = nlohmann::json{
        {         "camera_name",          p.camera_name},
        {"camera_serial_number", p.camera_serial_number},
        {       "camera_status",        p.camera_status},
        {        "camera_width",         p.camera_width},
        {       "camera_height",        p.camera_height},
        {     "camera_exposure",      p.camera_exposure}
    };
}

// 3. 从 json 数据到结构体数据
inline void from_json(const nlohmann::json &j, CameraParams &p)
{
    try
    {
        p.camera_name = j.at("camera_name").get<std::string>();
    }
    catch (const std::exception)
    {
    }

    try
    {
        p.camera_serial_number = j.at("camera_serial_number").get<std::string>();
    }
    catch (const std::exception)
    {
    }

    try
    {
        p.camera_status = j.at("camera_status").get<bool>();
    }
    catch (const std::exception)
    {
    }

    try
    {
        p.camera_width = j.at("camera_width").get<unsigned int>();
    }
    catch (const std::exception)
    {
    }

    try
    {
        p.camera_height = j.at("camera_height").get<unsigned int>();
    }
    catch (const std::exception)
    {
    }

    try
    {
        p.camera_exposure = j.at("camera_exposure").get<float>();
    }
    catch (const std::exception)
    {
    }
}

// 4. 保存到 json 本地文件
template<typename T>
inline void SaveParamToFile(std::shared_ptr<T> param, const std::string &dir, const std::string &filename,
                            const std::string &cn)
{
    std::filesystem::path path(dir);
    if (!std::filesystem::exists(path))
        std::filesystem::create_directories(path);

    nlohmann::json temp_json;
    temp_json.emplace(cn, *param);

    std::ofstream ofs(dir + filename, std::ios::out);
    if (ofs.is_open())
    {
        ofs << std::setw(4) << temp_json;
        ofs.close();
    }
}

template<typename T>
inline void SaveParamToFile(T *param, const std::string &dir, const std::string &filename, const std::string &cn)
{
    std::filesystem::path path(dir);
    if (!std::filesystem::exists(path))
        std::filesystem::create_directories(path);

    nlohmann::json temp_json;
    temp_json.emplace(cn, *param);

    std::ofstream ofs(dir + filename, std::ios::out);
    if (ofs.is_open())
    {
        ofs << std::setw(4) << temp_json;
        ofs.close();
    }
}

template<typename T>
inline void SaveParamToFile(T &param, const std::string &dir, const std::string &filename, const std::string &cn)
{
    std::filesystem::path path(dir);
    if (!std::filesystem::exists(path))
        std::filesystem::create_directories(path);

    nlohmann::json temp_json;
    temp_json.emplace(cn, param);

    std::ofstream ofs(dir + filename, std::ios::out);
    if (ofs.is_open())
    {
        ofs << std::setw(4) << temp_json;
        ofs.close();
    }
}

// 5. 从 json 本地文件中加载
template<typename T>
inline void LoadParamFromFile(std::shared_ptr<T> param, const std::string &dir, const std::string &filename,
                              const std::string &cn)
{
    try
    {
        std::ifstream ifs(dir + filename, std::ios::out);
        if (ifs.is_open())
        {
            nlohmann::json j;
            ifs >> j;
            *param = j[cn];
            ifs.close();
        }
    }
    catch (const std::exception &e)
    {
    }
}

template<typename T>
inline void LoadParamFromFile(T *param, const std::string &dir, const std::string &filename, const std::string &cn)
{
    try
    {
        std::ifstream ifs(dir + filename, std::ios::out);
        if (ifs.is_open())
        {
            nlohmann::json j;
            ifs >> j;
            *param = j[cn];
            ifs.close();
        }
    }
    catch (const std::exception &e)
    {
    }
}

template<typename T>
inline void LoadParamFromFile(T &param, const std::string &dir, const std::string &filename, const std::string &cn)
{
    try
    {
        std::ifstream ifs(dir + filename, std::ios::out);
        if (ifs.is_open())
        {
            nlohmann::json j;
            ifs >> j;
            param = j[cn];
            ifs.close();
        }
    }
    catch (const std::exception &e)
    {
    }
}

// 6. 宏定义提供函数接口供外部使用
// #的用法是负责将其后面的东西转化为字符串
#define SAVE_CAMERA_PARAMS(param, dir, filename) SaveParamToFile(param, dir, filename, #param)
#define LOAD_CAMERA_PARAMS(param, dir, filename) LoadParamFromFile(param, dir, filename, #param)

// 7. 将该结构体参数配置为一个class
class CameraParamsConf
{
public:
    CameraParamsConf(const CameraParamsConf &t)            = delete;
    CameraParamsConf &operator=(const CameraParamsConf &t) = delete;

    static CameraParamsConf *getInstance(std::string &path, std::string &filename)
    {
        m_path     = path;
        m_filename = filename;
        return m_pInstance;
    }

    void          SaveParams();
    void          LoadParams();
    CameraParams *getParams();
    void          setParams(CameraParams &params);

private:
    CameraParamsConf() = default;
    static CameraParamsConf *m_pInstance;

    void createDirectory(std::string &path);

    static std::string m_path;
    static std::string m_filename;
    CameraParams       m_cameraParams;
};
