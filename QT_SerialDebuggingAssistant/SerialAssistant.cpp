#include "SerialAssistant.h"

#include <qglobal.h>
#include <qmessagebox.h>
#include <qpushbutton.h>
#include <qtextcursor.h>

#include <QByteArray>
#include <QCheckBox>
#include <QDateTime>
#include <QFileDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QSerialPortInfo>
#include <cctype>
#include <cstring>

SerialAssistant::SerialAssistant(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WidgetSerialAssistant())
    , m_pSerialPort{new QSerialPort(this)}
    , m_writeBytesCounterTotal{0}
    , m_readBytesCounterTotal{0}
    , m_recordMsgBack{""}
    , m_isOpen{false}
    , m_pTimerInSend{new QTimer(this)}
    , m_pTimerSysDateTime{new QTimer(this)}
    , m_currentDateTimeStr{""}
    , m_pTimerCycleSend{new QTimer(this)}
    , m_btnIndex{0}
{
    ui->setupUi(this);
    initSerialPorts();
    initConnects();
    initWidgetStatus();

    for (int idx{1}; idx <= 9; ++idx)
    {
        QString      btnName = QString("Table_btn_%1").arg(idx);
        QPushButton *btn     = findChild<QPushButton *>(btnName);
        if (btn)
        {
            btn->setProperty("btn_ID", idx);
            m_pButtons.append(btn);
            connect(btn, &QPushButton::clicked, this, &SerialAssistant::sl_command);
        }

        QString    lineEditName = QString("Table_lineEdit_%1").arg(idx);
        QLineEdit *lineEdit     = findChild<QLineEdit *>(lineEditName);
        if (lineEdit)
        {
            lineEdit->setProperty("lineEdit_ID", idx);
            m_pLineEdits.append(lineEdit);
        }

        QString    checkBoxName = QString("Table_checkBox_%1").arg(idx);
        QCheckBox *checkBox     = findChild<QCheckBox *>(checkBoxName);
        if (checkBox)
        {
            checkBox->setProperty("checkBox_ID", idx);
            m_pCheckBoxes.append(checkBox);
        }
    }
}

SerialAssistant::~SerialAssistant()
{
    if (ui)
    {
        delete ui;
        ui = nullptr;
    }
}

void SerialAssistant::initSerialPorts()
{
    ui->comboBox_serialPorts->clear();
    const auto serialPortInfos = QSerialPortInfo::availablePorts();
    for (const QSerialPortInfo &portInfo : serialPortInfos)
    {
        //     qDebug() << "\n"
        //              << "Port:" << portInfo.portName() << "\n"
        //              << "Location:" << portInfo.systemLocation() << "\n"
        //              << "Description:" << portInfo.description() << "\n"
        //              << "Manufacturer:" << portInfo.manufacturer() << "\n"
        //              << "Serial number:" << portInfo.serialNumber() << "\n"
        //              << "Vendor Identifier:"
        //              << (portInfo.hasVendorIdentifier() ? QByteArray::number(portInfo.vendorIdentifier(), 16)
        //                                                 : QByteArray())
        //              << "\n"
        //              << "Product Identifier:"
        //              << (portInfo.hasProductIdentifier() ? QByteArray::number(portInfo.productIdentifier(), 16)
        //                                                  : QByteArray());

        ui->comboBox_serialPorts->addItem(portInfo.portName());
    }
    ui->label_sendOK->setText("COM Ready!");
}

void SerialAssistant::initConnects()
{
    connect(ui->btn_openCloseSerial, &QPushButton::clicked, this, &SerialAssistant::sl_OpenCloseSerial);
    connect(ui->btn_sendData, &QPushButton::clicked, this, &SerialAssistant::sl_SendContext);
    connect(m_pSerialPort, &QSerialPort::readyRead, this, &SerialAssistant::sl_SerialDataReadyToRead);
    connect(m_pTimerInSend, &QTimer::timeout, [=]() { sl_SendContext(); });
    connect(ui->btn_clearRecv, &QPushButton::clicked, this, &SerialAssistant::sl_ClearRecvMessage);
    connect(ui->btn_saveRecvMsg, &QPushButton::clicked, this, &SerialAssistant::sl_SaveRecvMessage);
    connect(m_pTimerSysDateTime, &QTimer::timeout,
            [=]()
            {
                QDateTime time       = QDateTime::currentDateTime();
                m_currentDateTimeStr = time.toString("yyyy-MM-dd hh:mm:ss");
                ui->label_sysDateTime->setText(m_currentDateTimeStr);
            });

    // connect(ui->btn_hideTable, &QPushButton::clicked, this, &SerialAssistant::sl_hideTable);
    // connect(ui->btn_hideRecord, &QPushButton::clicked, this, &SerialAssistant::sl_hideRecord);

    connect(ui->comboBox_serialPorts, &SerialRefresh::refresh, this, &SerialAssistant::initSerialPorts);

    // connect(ui->Table_btn_1, &QPushButton::clicked, this, &SerialAssistant::sl_command1);
    // connect(ui->Table_btn_2, &QPushButton::clicked, this, &SerialAssistant::sl_command2);
    // connect(ui->Table_btn_3, &QPushButton::clicked, this, &SerialAssistant::sl_command3);
    // connect(ui->Table_btn_4, &QPushButton::clicked, this, &SerialAssistant::sl_command4);
    // connect(ui->Table_btn_5, &QPushButton::clicked, this, &SerialAssistant::sl_command5);
    // connect(ui->Table_btn_6, &QPushButton::clicked, this, &SerialAssistant::sl_command6);
    // connect(ui->Table_btn_7, &QPushButton::clicked, this, &SerialAssistant::sl_command7);
    // connect(ui->Table_btn_8, &QPushButton::clicked, this, &SerialAssistant::sl_command8);
    // connect(ui->Table_btn_9, &QPushButton::clicked, this, &SerialAssistant::sl_command9);

    connect(m_pTimerCycleSend, &QTimer::timeout, this, &SerialAssistant::sl_handleCycleSend);
    connect(ui->Table_btn_reset, &QPushButton::clicked, this, &SerialAssistant::sl_handleTableReset);
    connect(ui->Table_btn_save, &QPushButton::clicked, this, &SerialAssistant::sl_handleTableSave);
    connect(ui->Table_btn_load, &QPushButton::clicked, this, &SerialAssistant::sl_handleTableLoad);
}

void SerialAssistant::initWidgetStatus()
{
    ui->btn_sendData->setEnabled(false);
    ui->checkBox_sendInTimer->setEnabled(false);
    ui->checkBox_format->setEnabled(false);
    ui->checkBox_HexSend->setEnabled(false);
    ui->checkBox_newLineSend->setEnabled(false);

    // set checkable for signal and slot
    ui->btn_openCloseSerial->setCheckable(true);
    ui->btn_hideTable->setCheckable(true);
    ui->btn_hideRecord->setCheckable(true);
    ui->checkBox_cycleSend->setCheckable(true);

    m_pTimerSysDateTime->start(1000); // 1s
    QDateTime time       = QDateTime::currentDateTime();
    m_currentDateTimeStr = time.toString("yyyy-MM-dd hh:mm:ss");
}

void SerialAssistant::sl_OpenCloseSerial()
{
    if (!m_isOpen)
    {
        // Step 1. 选择串口
        m_pSerialPort->setPortName(ui->comboBox_serialPorts->currentText());
        // Step 2. 设置波特率
        m_pSerialPort->setBaudRate(ui->comboBox_BaudRate->currentText().toInt());
        // Step 3. 设置数据位
        m_pSerialPort->setDataBits(QSerialPort::DataBits(ui->comboBox_dataBits->currentText().toInt()));
        // Step 4. 设置校验位
        switch (ui->comboBox_Parity->currentIndex())
        {
        case 0: //"None"
            m_pSerialPort->setParity(QSerialPort::NoParity);
            break;
        case 1: //"Even"
            m_pSerialPort->setParity(QSerialPort::EvenParity);
            break;
        case 2: //"Odd"
            m_pSerialPort->setParity(QSerialPort::OddParity);
            break;
        case 3: //"Space"
            m_pSerialPort->setParity(QSerialPort::SpaceParity);
            break;
        case 4: //"Mark"
            m_pSerialPort->setParity(QSerialPort::MarkParity);
            break;
        default:
            m_pSerialPort->setParity(QSerialPort::NoParity);
            break;
        }
        // Step 5. 设置停止位
        m_pSerialPort->setStopBits(QSerialPort::StopBits(ui->comboBox_stopBits->currentText().toInt()));
        // Step 6. 设置流控
        switch (ui->comboBox_FlowControl->currentIndex())
        {
        case 0: //"None"
            m_pSerialPort->setFlowControl(QSerialPort::NoFlowControl);
            break;
        case 1: //"Hardware"
            m_pSerialPort->setFlowControl(QSerialPort::HardwareControl);
            break;
        case 2: //"Software"
            m_pSerialPort->setFlowControl(QSerialPort::SoftwareControl);
            break;
        default:
            m_pSerialPort->setFlowControl(QSerialPort::NoFlowControl);
            break;
        }

        // Step 7. open the serial port
        if (m_pSerialPort->open(QIODeviceBase::ReadWrite))
        {
            ui->label_status->setStyleSheet(m_green_SheetStyle);

            ui->comboBox_serialPorts->setEnabled(false);
            ui->comboBox_BaudRate->setEnabled(false);
            ui->comboBox_dataBits->setEnabled(false);
            ui->comboBox_Parity->setEnabled(false);
            ui->comboBox_stopBits->setEnabled(false);
            ui->comboBox_FlowControl->setEnabled(false);
            ui->btn_sendData->setEnabled(true);
            ui->checkBox_sendInTimer->setEnabled(true);
            ui->checkBox_HexSend->setEnabled(true);
            ui->checkBox_newLineSend->setEnabled(true);

            ui->btn_openCloseSerial->setText(u8"关闭串口");
            m_isOpen = true;
        }
        else
        {
            ui->label_status->setStyleSheet(m_red_SheetStyle);
            QMessageBox msg_box;
            msg_box.setWindowTitle(u8"打开串口错误");
            msg_box.setText(u8"打开串口失败, 串口可能被占用或者拔出, 请检查");
            msg_box.exec();
        }
    }
    else
    {
        m_pSerialPort->close();

        ui->comboBox_serialPorts->setEnabled(true);
        ui->comboBox_BaudRate->setEnabled(true);
        ui->comboBox_dataBits->setEnabled(true);
        ui->comboBox_Parity->setEnabled(true);
        ui->comboBox_stopBits->setEnabled(true);
        ui->comboBox_FlowControl->setEnabled(true);
        ui->btn_sendData->setEnabled(false);
        ui->checkBox_sendInTimer->setEnabled(false);
        ui->checkBox_HexSend->setEnabled(false);
        ui->checkBox_newLineSend->setEnabled(false);

        ui->btn_openCloseSerial->setText(u8"打开串口");
        m_isOpen = false;

        ui->checkBox_sendInTimer->setCheckState(Qt::Unchecked);
        m_pTimerInSend->stop();
        ui->lineEdit_InTimer->setEnabled(true);
        ui->lineEdit_SendContext->setEnabled(true);
    }
}

void SerialAssistant::sl_SendContext()
{
    // const char *message = ui->lineEdit_SendContext->text().toStdString().c_str();
    // auto message = ui->lineEdit_SendContext->text().toStdString();
    auto message = ui->lineEdit_SendContext->text().toLocal8Bit().constData();

    int writeBytesCounter;
    // 串口发送数据
    if (ui->checkBox_HexSend->isChecked())
    {
        QString    temp = ui->lineEdit_SendContext->text();
        // 参数防呆: 是否是偶数
        QByteArray tempArray = temp.toLocal8Bit();
        if (tempArray.size() % 2 != 0)
        {
            ui->label_sendOK->setText("Error input!");
            return;
        }
        // 参数防呆: 是否符合十六进制
        for (const auto &c : tempArray)
        {
            if (!std::isdigit(c))
            {
                ui->label_sendOK->setText("Error input!");
                return;
            }
        }

        if (ui->checkBox_newLineSend->isChecked())
            tempArray.append("\r\n");

        // 进制转换后发送, 注意避免变成ASCII的字符了！
        QByteArray arrayByteSend = QByteArray::fromHex(tempArray);
        writeBytesCounter        = m_pSerialPort->write(arrayByteSend);
    }
    else
    {
        if (ui->checkBox_newLineSend->isChecked())
        {
            QByteArray arraySendData(message, strlen(message));
            arraySendData.append("\r\n");
            writeBytesCounter = m_pSerialPort->write(arraySendData);
        }
        else
        {
            writeBytesCounter = m_pSerialPort->write(message);
        }
    }

    // auto writeBytesCounter = m_pSerialPort->write(message.c_str());
    if (-1 == writeBytesCounter)
    {
        ui->label_sendOK->setText("Send Error!");
    }
    else
    {
        // ui->textEdit_recordMsg->append(message);
        // if (0 != std::strcmp(m_recordMsgBack.c_str(), message.c_str()))
        if (0 != std::strcmp(m_recordMsgBack.toStdString().c_str(), message))
        {
            // ui->textEdit_recordMsg->append(message.c_str());
            ui->textEdit_recordMsg->append(message);
            // m_recordMsgBack = message;
            m_recordMsgBack = QString::fromUtf8(message);
        }

        ui->label_sendOK->setText("Send OK!");
        m_writeBytesCounterTotal += writeBytesCounter;
        // update
        ui->label_sendBytesTotal->setNum(m_writeBytesCounterTotal);
    }
}

void SerialAssistant::sl_SerialDataReadyToRead()
{
    // auto receivedData = m_pSerialPort->readAll();
    QByteArray receivedData = m_pSerialPort->readAll();
    // 解析指令和数据并进行相应操作

    if (nullptr != receivedData)
    {
        if (ui->checkBox_newLineShow->isChecked())
        {
            receivedData.append("\r\n"); // Windows 操作系统
        }

        if (ui->checkBox_HexShow->isChecked())
        {
            QByteArray tempHexString = receivedData.toHex().toUpper();
            // 因为勾选了, 原来控件中的内容是 hex
            QString    temStringHex = ui->textEdit_recvMsg->toPlainText();
            tempHexString           = temStringHex.toUtf8() + tempHexString; // 新接受的和原始的拼接hex
            ui->textEdit_recvMsg->setText(QString::fromUtf8(tempHexString));
        }
        else
        {
            if (Qt::Checked == ui->checkBox__recvDateTime->checkState())
            {
                QString receivedDataStr = m_currentDateTimeStr + QString(receivedData);
                // ui->textEdit_recvMsg->append(receivedDataStr); // 多了一个 \n
                ui->textEdit_recvMsg->insertPlainText(receivedDataStr);
            }
            else
            {
                // ui->textEdit_recvMsg->append(receivedData); // 多了一个 \n
                ui->textEdit_recvMsg->insertPlainText(receivedData);
            }
        }

        m_readBytesCounterTotal += receivedData.size();
        // update
        ui->label_recvBytesTotal->setNum(m_readBytesCounterTotal);

        // 下拉滚动条位置
        ui->textEdit_recvMsg->moveCursor(QTextCursor::End);
        ui->textEdit_recvMsg->ensureCursorVisible();
        // ui->textEdit_recvMsg->setFocus();
    }
}

void SerialAssistant::on_checkBox_sendInTimer_clicked(bool checked)
{
    if (checked)
    {
        ui->lineEdit_InTimer->setEnabled(false);
        ui->lineEdit_SendContext->setEnabled(false);
        // sl_SendContext();
        // m_pTimerInSend->start(500);
        m_pTimerInSend->start(ui->lineEdit_InTimer->text().toInt());
    }
    else
    {
        m_pTimerInSend->stop();
        ui->lineEdit_InTimer->setEnabled(true);
        ui->lineEdit_SendContext->setEnabled(true);
    }
}

void SerialAssistant::sl_ClearRecvMessage()
{
    ui->textEdit_recvMsg->clear();
}

void SerialAssistant::sl_SaveRecvMessage()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "D:/serialData.txt", tr("Text (*.txt)"));

    if (filename != nullptr)
    {
        QFile file(filename);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;

        QTextStream out(&file);
        out << ui->textEdit_recvMsg->toPlainText();
        file.close();
    }
}

void SerialAssistant::on_checkBox_HexShow_clicked(bool checked)
{
    if (checked)
    {
        // Step 1. 读取
        QString temp_str = ui->textEdit_recvMsg->toPlainText();

        // Step 2. 转换 hex
        QByteArray temp_byte = temp_str.toUtf8();
        temp_byte            = temp_byte.toHex();

        // Step 3. 显示,大写范式
        QString lastShow;
        temp_str = QString::fromUtf8(temp_byte);
        for (int idx{0}; idx < temp_str.size(); idx += 2)
        {
            lastShow += temp_str.mid(idx, 2) + " ";
        }
        ui->textEdit_recvMsg->setText(lastShow.toUpper());
    }
    else
    {
        // Step 1. 读取
        QString temp_HEXstr = ui->textEdit_recvMsg->toPlainText();

        // Step 2. 转换 hex
        QByteArray temp_HEXbyte = temp_HEXstr.toUtf8();
        QByteArray temp_HexStr  = QByteArray::fromHex(temp_HEXbyte);

        // Step 3. 显示
        ui->textEdit_recvMsg->setText(QString::fromUtf8(temp_HexStr));
    }
}

void SerialAssistant::on_btn_hideTable_clicked(bool checked)
{
    if (checked)
    {
        ui->btn_hideTable->setText(tr(u8"拓展面板"));
        ui->groupBox_Table->hide();
    }
    else
    {
        ui->btn_hideTable->setText(tr(u8"隐藏面板"));
        ui->groupBox_Table->show();
    }
}

void SerialAssistant::on_btn_hideRecord_clicked(bool checked)
{
    if (checked)
    {
        ui->btn_hideRecord->setText(tr(u8"拓展历史"));
        ui->groupBox_Record->hide();
    }
    else
    {
        ui->btn_hideRecord->setText(tr(u8"隐藏历史"));
        ui->groupBox_Record->show();
    }
}

void SerialAssistant::sl_command1()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_1->text());
    // if (ui->Table_checkBox_1->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_1->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command2()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_2->text());
    // if (ui->Table_checkBox_2->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_2->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command3()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_3->text());
    // if (ui->Table_checkBox_3->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_3->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command4()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_4->text());
    // if (ui->Table_checkBox_4->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_4->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command5()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_5->text());
    // if (ui->Table_checkBox_5->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_5->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command6()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_6->text());
    // if (ui->Table_checkBox_6->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_6->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command7()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_7->text());
    // if (ui->Table_checkBox_7->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_7->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command8()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_8->text());
    // if (ui->Table_checkBox_8->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_8->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command9()
{
    ui->lineEdit_SendContext->setText(ui->Table_lineEdit_9->text());
    // if (ui->Table_checkBox_9->isChecked())
    // ui->checkBox_HexSend->setChecked(true);
    ui->checkBox_HexSend->setChecked(ui->Table_checkBox_9->isChecked());

    sl_SendContext();
}

void SerialAssistant::sl_command()
{
    QPushButton *btn = qobject_cast<QPushButton *>(sender());
    if (btn)
    {
        int idx = btn->property("btn_ID").toInt();

        QString    lineEditName = QString("Table_lineEdit_%1").arg(idx);
        QLineEdit *lineEdit     = findChild<QLineEdit *>(lineEditName);
        if (lineEdit)
        {
            if (lineEdit->text().size() <= 0) // 空白
                return;
            ui->lineEdit_SendContext->setText(lineEdit->text());
        }

        QString    checkBoxName = QString("Table_checkBox_%1").arg(idx);
        QCheckBox *checkBox     = findChild<QCheckBox *>(checkBoxName);
        if (checkBox)
            ui->checkBox_HexSend->setChecked(checkBox->isChecked());
    }

    sl_SendContext();
}

void SerialAssistant::on_checkBox_cycleSend_clicked(bool checked)
{
    if (checked)
    {
        m_pTimerCycleSend->start(ui->spinBox_cycleTime->value());
        ui->spinBox_cycleTime->setEnabled(false);
    }
    else
    {
        m_pTimerCycleSend->stop();
        ui->spinBox_cycleTime->setEnabled(true);
    }
}

void SerialAssistant::sl_handleCycleSend()
{
    if (m_btnIndex < m_pButtons.size())
    {
        emit m_pButtons[m_btnIndex]->clicked();
        ++m_btnIndex;
    }
    else
    {
        m_btnIndex = 0;
    }
}

void SerialAssistant::sl_handleTableReset()
{
    QMessageBox msg_box;
    msg_box.setWindowTitle(tr(u8"提示信息"));
    msg_box.setIcon(QMessageBox::Question);
    msg_box.setText(tr(u8"重置列表不可逆, 请确认是否重置? "));
    // msg_box.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
    QPushButton *yesBtn = msg_box.addButton(u8"确定", QMessageBox::YesRole);
    QPushButton *noBtn  = msg_box.addButton(u8"取消", QMessageBox::NoRole);
    msg_box.exec();
    if (msg_box.clickedButton() == yesBtn)
    {
        // 遍历 lineEdit, 清空对应内容
        // 遍历 checkBox, 取消对应勾选
        for (int idx{0}; idx < m_pLineEdits.size(); ++idx)
        {
            m_pLineEdits[idx]->clear();
            m_pCheckBoxes[idx]->setChecked(false);
        }
    }
    else if (msg_box.clickedButton() == noBtn)
    {
    }
}

void SerialAssistant::sl_handleTableSave()
{
    QDateTime time     = QDateTime::currentDateTime();
    auto      nameStr  = "serialData_" + time.toString("yyyy-MM-dd_hh_mm_ss") + ".txt";
    QString   filename = QFileDialog::getSaveFileName(this, tr("Save File"), "D:/" + nameStr, tr("Text (*.txt)"));

    if (filename != nullptr)
    {
        QFile file(filename);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
            return;

        QTextStream out(&file);
        for (int idx{0}; idx < m_pLineEdits.size(); ++idx)
        {
            out << m_pCheckBoxes[idx]->isChecked() << " ---> " << m_pLineEdits[idx]->text() << "\n";
        }

        file.close();
    }
}

void SerialAssistant::sl_handleTableLoad()
{
    QDateTime time     = QDateTime::currentDateTime();
    auto      nameStr  = "serialData_" + time.toString("yyyy-MM-dd_hh_mm_ss") + ".txt";
    QString   filename = QFileDialog::getOpenFileName(this, tr("Open File"), "D:/" + nameStr, tr("Text (*.txt)"));

    if (filename != nullptr)
    {
        QFile file(filename);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
            return;

        int         idx = 0;
        QTextStream in(&file);
        while (!in.atEnd() && idx < 9)
        {
            QString line_info   = in.readLine();
            auto    split_parts = line_info.split(" ---> ");
            if (2 == split_parts.count())
            {
                m_pCheckBoxes[idx]->setChecked(split_parts[0].toInt());
                m_pLineEdits[idx]->setText(split_parts[1]);
            }
            ++idx;
        }

        file.close();
    }
}
