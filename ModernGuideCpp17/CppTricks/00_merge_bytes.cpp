/**
 * @file 00_merge_bytes.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-08-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <bitset>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

uint32_t bytesToUInt32(uint8_t byte1, uint8_t byte2, uint8_t byte3, uint8_t byte4)
{
    return (static_cast<uint32_t>(byte1) << 24) | (static_cast<uint32_t>(byte2) << 16)
         | (static_cast<uint32_t>(byte3) << 8) | static_cast<uint32_t>(byte4);
}

float binaryToFloat(const std::string &binaryStr)
{
    // Ensure the binary string is 32 bits long
    if (binaryStr.length() != 32)
    {
        throw std::invalid_argument("Binary string must be 32 bits long.");
    }

    // Interpret the binary string as a 32-bit unsigned integer
    uint32_t intRepresentation = std::bitset<32>(binaryStr).to_ulong();

    // Extract the sign bit, exponent bits, and mantissa bits
    uint32_t signBit      = (intRepresentation >> 31) & 0x1;
    uint32_t exponentBits = (intRepresentation >> 23) & 0xFF;
    uint32_t mantissaBits = intRepresentation & 0x7FFFFF;

    // Calculate the exponent
    int   exponent = exponentBits - 127;
    // Bias for single precision is 127
    // Calculate the mantissa (with an implicit leading 1 for normalized numbers)
    float mantissa = 1.0f;
    for (int i = 0; i < 23; ++i)
    {
        if (mantissaBits & (1 << (22 - i)))
        {
            mantissa += std::pow(2.0f, -(i + 1));
        }
    }

    // Calculate the final float value
    float value = std::pow(2.0f, exponent) * mantissa;

    // Apply the sign
    if (signBit == 1)
    {
        value = -value;
    }
    return value;
}

float bytesToFloat(uint8_t byte1, uint8_t byte2, uint8_t byte3, uint8_t byte4)
{
    // Interpret the binary string as a 32-bit unsigned integer
    uint32_t intRepresentation = (static_cast<uint32_t>(byte1) << 24) | (static_cast<uint32_t>(byte2) << 16)
                               | (static_cast<uint32_t>(byte3) << 8) | static_cast<uint32_t>(byte4);

    // Extract the sign bit, exponent bits, and mantissa bits
    uint32_t signBit      = (intRepresentation >> 31) & 0x1;
    uint32_t exponentBits = (intRepresentation >> 23) & 0xFF;
    uint32_t mantissaBits = intRepresentation & 0x7FFFFF;

    // Calculate the exponent
    int   exponent = exponentBits - 127;
    // Bias for single precision is 127
    // Calculate the mantissa (with an implicit leading 1 for normalized numbers)
    float mantissa = 1.0f;
    for (int i = 0; i < 23; ++i)
    {
        if (mantissaBits & (1 << (22 - i)))
        {
            mantissa += std::pow(2.0f, -(i + 1));
        }
    }

    // Calculate the final float value
    float value = std::pow(2.0f, exponent) * mantissa;

    // Apply the sign
    if (signBit == 1)
    {
        value = -value;
    }
    return value;
}

// ---------------------------------------
int main(int argc, const char **argv)
{
    // 示例字节
    uint8_t byte1 = 0x12;
    uint8_t byte2 = 0x34;
    uint8_t byte3 = 0x56;
    uint8_t byte4 = 0x78;

    // 合并字节
    uint32_t result = bytesToUInt32(byte1, byte2, byte3, byte4);

    // 打印结果
    std::cout << "Combined uint32_t value: 0x" << std::hex << result << std::endl << std::endl;

    std::string binaryStr = "01000000101000000000000000000000";
    // Example binary representation of 5.0

    try
    {
        float result = binaryToFloat(binaryStr);
        std::cout << "Decimal value: " << std::fixed << std::setprecision(7) << result << std::endl;
    }
    catch (const std::invalid_argument &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // 40 49 0F DA  在线进制转换
    // http://www.speedfly.cn/tools/hexconvert/
    unsigned char bytes[4] = {0x40, 0x49, 0x0F, 0xDA};

    uint8_t high_byte1 = bytes[0];
    uint8_t high_byte2 = bytes[1];
    uint8_t low_byte1  = bytes[2];
    uint8_t low_byte2  = bytes[3];
    std::cout << bytesToFloat(high_byte1, high_byte2, low_byte1, low_byte2);

    return 0;
}

// g++ 00_merge_bytes.cpp -std=c++17
