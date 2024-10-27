#include "20_mainwidget.hpp"

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/qt_sinks.h"

#include <QTextEdit>
#include <vector>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , pTimer{new QTimer}
{
    // 设置主窗口的最小尺寸
    setMinimumSize(640, 480);

    // 设置font14px，微软雅黑，浅绿色显示
    QFont font = QFont("Microsoft YaHei", 12, 2);

    // 创建 QTextEdit 控件用于显示日志
    auto log_widget = new QTextEdit(this);
    log_widget->setStyleSheet("background-color: grey;");
    log_widget->setReadOnly(true); // 设置为只读
    log_widget->setFont(font);

    // 将 QTextEdit 设置为主窗口的中央控件
    setCentralWidget(log_widget);

    // 设置最大行数限制，超过时删除旧行
    int max_lines = 50;

    // 创建一个彩色日志记录器，输出到 QTextEdit 控件
    //<1.创建多个sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/qt_log.txt", true);
    auto qt_sink   = std::make_shared<spdlog::sinks::qt_color_sink_mt>(log_widget, max_lines, false, true);
    std::vector<spdlog::sink_ptr> sinks = {file_sink, qt_sink};

    //<2.创建使用多个sink的单个logger，logger会把内容输出到不同位置，此处是控制台以及RotatingFileLog.txt
    m_pLogger = std::make_shared<spdlog::logger>("LoggerQT", sinks.begin(), sinks.end());

    pTimer->start(125); // ms
    connect(pTimer, &QTimer::timeout, this, &MainWindow::sl_logMessage);

    // // 记录一些日志消息
    // const unsigned int msg_size = max_lines + 200;
    // for (size_t idx{0}; idx < msg_size; ++idx)
    // {
    //     logger->info("Some info message");
    //     logger->warn("This is a warning message");
    //     logger->error("This is an error message");
    //     logger->critical("This is a critical message");
    // }

    /* 代码解释
    * 1. 创建 QTextEdit 控件:
    * ----QTextEdit 控件用于显示日志消息, 将其设置为只读模式, 以避免用户修改显示的日志内容.
    * ----setCentralWidget(log_widget); 将 QTextEdit 设置为主窗口的中央控件.
    * 2. 设置日志记录器:
    * ----使用 spdlog::qt_color_logger_mt 创建一个彩色日志记录器, 将日志消息输出到 QTextEdit 控件中;
    * ----max_lines 参数设置了 QTextEdit 中的最大行数, 
    *   如果日志行数超过这个限制, 旧的日志行将被删除, 以保持内容的更新.
    * 3. 记录日志消息:
    * ----定时器模拟不断向控件输出日志信息;
    * 日志消息将在 QTextEdit 控件中显示, 并且颜色会根据日志级别自动调整
    * （例如，info 为绿色，warn 为黄色，error 为红色等）
    */
}

void MainWindow::sl_logMessage()
{
    // 不断刷新记录一些日志消息
    m_pLogger->info("Some info message");
    m_pLogger->warn("This is a warning message");
    m_pLogger->error("This is an error message");
    m_pLogger->critical("This is a critical message");
}
