/**
 * @file test_example.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-24
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ test_example.cpp -std=c++20
 * clang++ test_example.cpp -std=c++20
 * 
 */

#include "msgpack.hpp"
// https://github.com/mikeloomisgg/cppack

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Person
{
    std::string              name;
    uint16_t                 age;
    std::vector<std::string> aliases;

    template<class T>
    void msgpack(T &pack)
    {
        pack(name, age, aliases);
    }

    template<class T>
    void pack(T &pack)
    {
        pack(name, age, aliases);
    }
};

enum class CameraBrand : int
{
    LineScan_HIK = 0,
    LineScan_Dalsa,

    NUM_LineScan
};

enum class AcquisitionCardBrand : int
{
    AcquisiCare_HIK = 0,
    AcquisiCard_Dalsa,

    NUM_AcquisiCard
};

struct CameraParam
{
    CameraBrand          camera_brand;       // 线扫相机品牌
    AcquisitionCardBrand acquisi_card_brand; // 图像采集卡的品牌

    /**
     * @brief 相机名称
     * 视觉系统中对相机的命名，用于相机模块和硬件序列号一一对应
     * 如: 尺寸测量相机/CCD1
     *
     */
    std::string cameraName;          // 尺寸测量相机 16K
    std::string cameraSerialNum;     // 相机硬件的序列号
    std::string cardSerialNum;       // 采集卡硬件的序列号
    std::string cardConfFilepath;    // 线扫相机对应采集卡的配置文件
    std::string cameraConfFilepath;  // 相机的配置文件
    std::string cameraSimulatePath;  // 相机的仿真图像路径
    double      cameraSimulateSpeed; // 相机的加载图像的速度
    int         simulateStatusCode;  // 仿真状态码

    unsigned int rowHeight;  // 线扫相机的行高
    double       exposure;   // 相机的曝光值
    double       shiftGain;  // 线扫相机的数值增益
    int          preampGain; // 线扫相机的模拟增益

    // 构造函数进行初始化配置参数
    CameraParam()
    {
        camera_brand        = CameraBrand::LineScan_HIK;
        acquisi_card_brand  = AcquisitionCardBrand::AcquisiCare_HIK;
        cameraName          = "CCD0";
        cameraSerialNum     = "J76178727";
        cardSerialNum       = "";
        cameraSimulatePath  = "";
        cardConfFilepath    = "";
        cameraConfFilepath  = "";
        cameraSimulateSpeed = 0.;
        simulateStatusCode  = 3;
        rowHeight           = 3000;
        exposure            = 15;
        shiftGain           = 0;
        preampGain          = 1000;
    }

    bool operator==(const CameraParam b) const
    {
        if (this->camera_brand != b.camera_brand || this->acquisi_card_brand != b.acquisi_card_brand
            || this->cameraName != b.cameraName || this->cameraSerialNum != b.cameraSerialNum
            || this->cardSerialNum != b.cardSerialNum || this->cameraSimulatePath != b.cameraSimulatePath
            || this->cardConfFilepath != b.cardConfFilepath || this->cameraConfFilepath != b.cameraConfFilepath
            || this->cameraSimulateSpeed != b.cameraSimulateSpeed || this->simulateStatusCode != b.simulateStatusCode
            || this->rowHeight != b.rowHeight || this->exposure != b.exposure || this->shiftGain != b.shiftGain
            || this->preampGain != b.preampGain)
        {
            return false;
        }
        return true;
    }

    template<class T>
    void pack(T &pack)
    {
        pack((int &)camera_brand, (int &)acquisi_card_brand, cameraName, cameraSerialNum, cardSerialNum,
             cameraSimulatePath, cardConfFilepath, cameraConfFilepath, cameraSimulateSpeed, simulateStatusCode,
             rowHeight, exposure, shiftGain, preampGain);
    }
};

struct alignas(64) TestStruct
{
    CameraBrand          camera_brand;
    AcquisitionCardBrand acquisi_card_brand;
    std::string          cameraName;
    std::string          cameraSerialNum;
    std::string          cardSerialNum;
    std::string          cardConfFilepath;
    std::string          cameraConfFilepath;
    std::string          cameraSimulatePath;
    double               cameraSimulateSpeed;
    int                  simulateStatusCode;
    unsigned int         rowHeight;
    unsigned int         rowWidth;
    double               exposure;
    double               shiftGain;
};

struct alignas(64) TestStruct_
{
    std::string  cameraSimulatePath;  // 40-bytes
    double       cameraSimulateSpeed; // 8-bytes
    int          simulateStatusCode;  // 4-bytes
    unsigned int rowHeight;           // 4-bytes
    unsigned int rowWidth;            // 4-bytes
};

struct _TestStruct_
{
    std::string  cameraSimulatePath;  // 40-bytes
    double       cameraSimulateSpeed; // 8-bytes
    int          simulateStatusCode;  // 4-bytes
    unsigned int rowHeight;           // 4-bytes
    unsigned int rowWidth;            // 4-bytes
};

int main(int argc, const char **argv)
{
    auto person = Person{
        "John",
        22,
        {"Ripper", "Silverhand"}
    };
    auto data = msgpack::pack(person);         // Pack your object
    auto john = msgpack::unpack<Person>(data); // Unpack it

    std::ofstream outfile;
    std::string   filepath = "msgpack.bin";
    outfile.open(filepath, std::ios_base::out | std::ios::binary);
    outfile.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(uint8_t));
    outfile.close();

    long long int size_vec = 0;
    std::ifstream infile_(filepath, std::ios_base::in | std::ios::binary);
    infile_.seekg(0, infile_.end);
    size_vec = infile_.tellg();
    infile_.seekg(0, infile_.beg);
    infile_.close();

    std::ifstream infile;
    infile.open(filepath, std::ios_base::in | std::ios::binary);
    std::vector<uint8_t> pData(size_vec);
    infile.read(reinterpret_cast<char *>(pData.data()), size_vec * sizeof(uint8_t));
    infile.close();
    auto john_ = msgpack::unpack<Person>(pData); // Unpack it

    // -------=============================----------------
    CameraParam params{};
    auto        data_param = msgpack::pack(params);                    // Pack your object
    auto        cam_param  = msgpack::unpack<CameraParam>(data_param); // Unpack it

    filepath = "cam_param.bin";
    outfile.open(filepath, std::ios_base::out | std::ios::binary);
    outfile.write(reinterpret_cast<const char *>(data_param.data()), data_param.size() * sizeof(uint8_t));
    outfile.close();

    std::ifstream infile__(filepath, std::ios_base::in | std::ios::binary);
    infile__.seekg(0, infile__.end);
    size_vec = infile__.tellg();
    infile__.seekg(0, infile__.beg);
    infile__.close();

    infile.open(filepath, std::ios_base::in | std::ios::binary);
    std::vector<uint8_t> pDataParam(size_vec);
    infile.read(reinterpret_cast<char *>(pDataParam.data()), size_vec * sizeof(uint8_t));
    infile.close();
    auto cam_param_ = msgpack::unpack<CameraParam>(pDataParam); // Unpack it

    // -------=============================----------------
    TestStruct test;
    auto       bytes_bool         = sizeof(bool);
    auto       bytes_uint16_t     = sizeof(uint16_t);
    auto       bytes_string       = sizeof(std::string);
    auto       bytes_int          = sizeof(int);
    auto       bytes_unsigned_int = sizeof(unsigned int);
    auto       bytes_float        = sizeof(float);
    auto       bytes_double       = sizeof(double);
    auto       sizeBytes          = sizeof(test);
    auto       sizeBytes_         = sizeof(TestStruct_);
    auto       _sizeBytes_        = sizeof(_TestStruct_);

    return 0;
}
