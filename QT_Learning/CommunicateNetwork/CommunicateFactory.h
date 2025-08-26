/**
 * @file CommunicateFactory.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2025-08-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include "ICommunication.h"

#include <QMap>
#include <QString>
#include <QVariantMap>

enum class ProtocolType
{
    Serial,
    ModbusRTU,
    CanBus,
    NetworkTCP
};

class CommunicationFactory
{
public:
    static CommunicationFactory &instance()
    {
        static CommunicationFactory factory;
        return factory;
    }

    void registerProtocol(ProtocolType type, std::function<ICommunication *()> creator)
    {
        m_creators[type] = creator;
    }

    ICommunication *create(ProtocolType type, const QVariantMap &config)
    {
        if (m_creators.contains(type))
        {
            ICommunication *device = m_creators[type]();
            if (device->connect(config))
            {
                return device;
            }
            else
            {
                delete device;
                device = nullptr;
                return nullptr;
            }
        }
        qWarning() << "未支持的协议类型：" << type;
        return nullptr;
    }

private:
    CommunicationFactory()
    {
        // 注册内置协议（可扩展）
        registerProtocol(Serial, []()
                         { returnnewSerialPort(); });
        registerProtocol(ModbusRTU, []()
                         { returnnewModbusRTU(); });
        registerProtocol(CanBus, []()
                         { returnnewCanBus(); });
        registerProtocol(NetworkTCP, []()
                         { returnnewNetworkSocket(); });
    }

    QMap<ProtocolType, std::function<ICommunication *()>> m_creators;
};

// TODO 全局通信管理器（单例模式）
// 注册宏（简化扩展）
#define REGISTER_PROTOCOL(type, cls)                                                                                       \
    namespace                                                                                                              \
    {                                                                                                                      \
        struct cls##Registrar                                                                                              \
        {                                                                                                                  \
            cls##Registrar()                                                                                               \
            {                                                                                                              \
                CommunicationFactory::instance().registerProtocol(CommunicationFactory::type, []() { return new cls(); }); \
            }                                                                                                              \
        } cls##RegistrarInstance;                                                                                          \
    }
