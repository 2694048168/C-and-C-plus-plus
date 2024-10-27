#pragma once

#include "spdlog/spdlog.h"

#include <QMainWindow>
#include <QTimer>
#include <memory>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

public slots:
    void sl_logMessage();

private:
    QTimer *pTimer;

    std::shared_ptr<spdlog::logger> m_pLogger;
};
