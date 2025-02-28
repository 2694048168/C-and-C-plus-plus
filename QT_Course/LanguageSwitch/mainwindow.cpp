#include "mainwindow.h"

#include "ui_mainwindow.h"

#include <QDebug>
#include <QDir>
#include <QProcess>
#include <QTextCodec>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // 1、初始化语言菜单
    initLanguageMenu();

    // 2、处理配置文件
    // Qt中使用QSettings类读写ini文件，第一个参数是ini文件路径，第二个参数表示是ini文件格式，第三个参数可省略
    settingsFile = new QSettings(QCoreApplication::applicationName() + ".ini", QSettings::IniFormat, this);
    // 解决QSettings保存中文值，ini自动保存为ANSI、GB2312格式后，启动崩溃问题
    settingsFile->setIniCodec(QTextCodec::codecForName("GBK"));

    // 从配置文件中获取用户选择语言
    currentLanguageQm = settingsFile->value("Main/strCurrentLanguageQm", "").toString();

    // 3、切换到保存的用户选择语言
    changeLanguage(currentLanguageQm);
}

MainWindow::~MainWindow()
{
    delete ui;
    // 释放 new 分配的内存
    if (translator)
    {
        delete translator;
    }
}

// 初始化语言菜单：编译 .ts 文件 .qm 文件完成后，再调用
// onCompileTsFilesFinished() 创建语言菜单项、刷新界面语言
void MainWindow::initLanguageMenu()
{
    // 1、编译 .ts 文件 .qm 文件
    QDir        languageDir("res/language");
    QStringList tsFileList = languageDir.entryList(QStringList("*.ts"), QDir::Files);
    QStringList tsFilePathList;
    for (const QString &tsFile : tsFileList)
    {
        tsFilePathList.append(languageDir.filePath(tsFile));
    }

    // 调用 lrelease.exe 编译 .ts 文件
    QProcess *process = new QProcess(this);

    process->start("res/lrelease/lrelease.exe", tsFilePathList);

    // 连接finished信号到槽函数。等待编译 .ts 文件 .qm 文件完成后，再调用
    // onCompileTsFilesFinished() 创建语言菜单项、刷新界面语言
    connect(process, static_cast<void (QProcess::*)(int, QProcess::ExitStatus)>(&QProcess::finished), this,
            [this, process]()
            {
                qDebug() << "Process finished:" << process->exitCode();
                if (process->exitStatus() == QProcess::NormalExit && process->exitCode() == 0)
                {
                    qDebug() << "Process exited normally.";
                    this->onCompileTsFilesFinished();
                }
                else
                {
                    qDebug() << "Process failed to exit normally.";
                    this->onCompileTsFilesFinished();
                }

                // 清理 QProcess 对象
                process->deleteLater();

                // 在 Qt 中，deleteLater()
                // 是一个用于延迟删除对象的方法。当你调用 process->deleteLater()
                // 时，Qt
                // 会在事件循环的下一个迭代中删除该对象。因此，你不需要再手动调用
                // delete process。
            });
}

// 等待编译 .ts 文件为 .qm 文件完成后再调用，用于创建语言菜单项、刷新界面语言
void MainWindow::onCompileTsFilesFinished()
{
    // 获取菜单栏
    QMenuBar *menuBar = this->menuBar();

    // 查找或创建“语言”主菜单
    QMenu           *languageMenu = nullptr;
    QList<QAction *> actions      = menuBar->actions();
    for (auto it = actions.begin(); it != actions.end(); ++it)
    {
        QAction *action = *it;
        if (action->text() == tr("语言"))
        {
            languageMenu = action->menu();
            break;
        }
    }

    if (!languageMenu)
    {
        languageMenu = menuBar->addMenu(tr("语言"));
    }

    // 扫描 res/language/ 目录下的 .qm 文件
    QDir        languageDir("res/language");
    QStringList qmFiles = languageDir.entryList(QStringList("*.qm"), QDir::Files);

    // 动态添加语言菜单项
    for (const QString &qmFile : qmFiles)
    {
        // 提取语言名称（例如：LangTest_en_US.qm -> 英文）
        QString languageName = extractLanguageName(qmFile);

        // 创建菜单项
        QAction *action = new QAction(languageName, this);
        action->setCheckable(true);
        languageMenu->addAction(action);

        // 存储菜单项和对应的 .qm 文件
        languageActions[action] = qmFile;

        // 连接信号与槽
        connect(action, &QAction::triggered, this, [this, qmFile]() { changeLanguage(qmFile); });
    }

    // 更新界面语言
    // retranslateUi();
    changeLanguage(currentLanguageQm);
}

// 提取语言名称
QString MainWindow::extractLanguageName(const QString &qmFile)
{
    if (qmFile.isEmpty())
    {
        return tr("简体中文");
    }

    // 不能定义为成员变量，否则语言切换时无法翻译语言菜单中的语言名称
    QMap<QString, QString> mapLanguage = {
        {   "_ar",     tr("阿拉伯语")},
        {   "_bg",   tr("保加利亚语")},
        {   "_ca", tr("加泰罗尼亚语")},
        {   "_cs",       tr("捷克语")},
        {   "_da",       tr("丹麦语")},
        {   "_de",         tr("德语")},
        {   "_el",       tr("希腊语")},
        {   "_en",         tr("英文")},
        {   "_es",     tr("西班牙语")},
        {   "_et",   tr("爱沙尼亚语")},
        {   "_eu",     tr("巴斯克语")},
        {   "_fa",       tr("波斯语")},
        {   "_fi",       tr("芬兰语")},
        {   "_fr",         tr("法语")},
        {   "_gl",   tr("加利西亚语")},
        {   "_he",     tr("希伯来语")},
        {   "_hi",       tr("印地语")},
        {   "_hr",   tr("克罗地亚语")},
        {   "_hu",     tr("匈牙利语")},
        {   "_id", tr("印度尼西亚语")},
        {   "_it",     tr("意大利语")},
        {   "_ja",         tr("日语")},
        {   "_ka",   tr("格鲁吉亚语")},
        {   "_ko",         tr("韩语")},
        {   "_lt",     tr("立陶宛语")},
        {   "_lv",   tr("拉脱维亚语")},
        {   "_ms",       tr("马来语")},
        {   "_nl",       tr("荷兰语")},
        {   "_no",       tr("挪威语")},
        {   "_pl",       tr("波兰语")},
        {   "_pt",     tr("葡萄牙语")},
        {"_pt_BR", tr("巴西葡萄牙语")},
        {   "_ro",   tr("罗马尼亚语")},
        {   "_ru",         tr("俄语")},
        {   "_sk",   tr("斯洛伐克语")},
        {   "_sl", tr("斯洛文尼亚语")},
        {   "_sr",   tr("塞尔维亚语")},
        {   "_sv",       tr("瑞典语")},
        {   "_th",         tr("泰语")},
        {   "_tr",     tr("土耳其语")},
        {   "_uk",     tr("乌克兰语")},
        {   "_vi",       tr("越南语")},
        {"_zh_CN",     tr("简体中文")},
        {"_zh_TW",     tr("繁体中文")}
    };

    // mapLanguage.size() = 44
    qDebug() << "mapLanguage.size() =" << mapLanguage.size();
    for (auto it = mapLanguage.begin(); it != mapLanguage.end(); ++it)
    {
        // qDebug() << it.key() << ": " << it.value();
        if (qmFile.contains(it.key()))
        {
            return it.value();
        }
    }

    return qmFile; // 默认返回文件名
}

// 切换语言
void MainWindow::changeLanguage(const QString &qmFile)
{
    // 移除旧的翻译器
    if (translator)
    {
        qApp->removeTranslator(translator);
        delete translator;
    }

    currentLanguageQm = qmFile; // 保存当前语言对应的qm文件名称
    qDebug() << "currentLanguage =" << currentLanguageQm;

    // 加载新的翻译器
    translator = new QTranslator(this);
    if (translator->load("res/language/" + qmFile))
    {
        qApp->installTranslator(translator);
        qDebug() << "Language changed to:" << qmFile;
    }
    else
    {
        qDebug() << "Failed to load language file:" << qmFile;
    }

    // 更新界面语言
    retranslateUi();
}

// 更新界面语言
void MainWindow::retranslateUi()
{
    // 更新主窗口的 UI
    ui->retranslateUi(this);

    // 更新语言菜单项的文本
    QAction *currentAction = nullptr;
    for (auto it = languageActions.begin(); it != languageActions.end(); ++it)
    {
        QAction       *action = it.key();
        const QString &qmFile = it.value();
        action->setText(extractLanguageName(qmFile));
        action->setChecked(false);
        if (currentLanguageQm.isEmpty()
            && 0 == defaultLanguage.compare(extractLanguageName(qmFile), Qt::CaseInsensitive))
        {
            currentAction = action;
        }
        else if (0 == currentLanguageQm.compare(qmFile, Qt::CaseInsensitive))
        {
            currentAction = action;
        }
    }
    if (nullptr != currentAction)
    {
        currentAction->setChecked(true);
    }

    // 注意：在.h头文件中定义字符串在QT语言切换时翻译失败，必须在赋值时直接使用tr("")才行
    // QString strLang = tr("QT多语言演示程序");currentLanguage
    QString strLang = tr("QT多语言演示程序。当前语言是");
    ui->label_3->setText(strLang + extractLanguageName(currentLanguageQm));

    // 保存当前语言 qm 文件名称
    settingsFile->setValue("Main/strCurrentLanguageQm", currentLanguageQm);

    qDebug() << "UI retranslated.";
}
