/**
 * @file TCPAssistantServer.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#pragma once

#include "ui_TCPAssistantServer.h"

#include <QTcpServer>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

namespace Ui {
class WidgetTCPAssistantServer;
};

QT_END_NAMESPACE

class TCPAssistantServer : public QWidget
{
    Q_OBJECT

public:
    TCPAssistantServer(QWidget *parent = nullptr);
    ~TCPAssistantServer();

protected slots:
    void sl_newClientConnect();
    void sl_listenHandler();
    void sl_stopListenHandler();
    void sl_disconnectHandler();
    void sl_socketReadyReadHandler();
    void sl_clientDisconnectedHandler();
    void sl_serverSendHandler();
    void sl_clientsRefreshHandler();
    void on_comboBox_clients_activated(int index); // void QComboBox::activated(int index)

private:
    Ui::WidgetTCPAssistantServer *ui;

    QTcpServer *m_pServer;
    int         m_currentClientIndex;
};
