/**
 * @file mainwindow.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-02-28
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <QMainWindow>
#include <QMap>
#include <QProcess>
#include <QSettings>
#include <QTranslator>

QT_BEGIN_NAMESPACE

namespace Ui {
class MainWindow;
}

QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow          *ui;
    QTranslator             *translator = nullptr;         // 翻译器对象
    QMap<QAction *, QString> languageActions;              // 存储菜单项和对应的 .qm 文件名称
    QString                  currentLanguageQm;            // 当前语言对应的 .qm 文件名称
    const QString            defaultLanguage = "简体中文"; // 缺省语言名称

    // 初始化语言菜单：编译 .ts 文件 .qm 文件完成后，再调用
    // onCompileTsFilesFinished() 创建语言菜单项、刷新界面语言
    void       initLanguageMenu();
    QString    extractLanguageName(const QString &qmFile); // 提取语言名称
    void       changeLanguage(const QString &qmFile);      // 切换语言
    void       retranslateUi();                            // 更新界面语言
    QSettings *settingsFile;                               // 配置文件对象

    // 不能定义为成员变量，否则语言切换时无法翻译语言菜单中的语言名称
    // QMap<QString, QString> mapLanguage = {
    //     {"_ar", tr("阿拉伯语")},     {"_bg", tr("保加利亚语")},
    //     {"_ca", tr("加泰罗尼亚语")}, {"_cs", tr("捷克语")},
    //     {"_da", tr("丹麦语")},       {"_de", tr("德语")},
    //     {"_el", tr("希腊语")},       {"_en", tr("英文")},
    //     {"_es", tr("西班牙语")},     {"_et", tr("爱沙尼亚语")},
    //     {"_eu", tr("巴斯克语")},     {"_fa", tr("波斯语")},
    //     {"_fi", tr("芬兰语")},       {"_fr", tr("法语")},
    //     {"_gl", tr("加利西亚语")},   {"_he", tr("希伯来语")},
    //     {"_hi", tr("印地语")},       {"_hr", tr("克罗地亚语")},
    //     {"_hu", tr("匈牙利语")},     {"_id", tr("印度尼西亚语")},
    //     {"_it", tr("意大利语")},     {"_ja", tr("日语")},
    //     {"_ka", tr("格鲁吉亚语")},   {"_ko", tr("韩语")},
    //     {"_lt", tr("立陶宛语")},     {"_lv", tr("拉脱维亚语")},
    //     {"_ms", tr("马来语")},       {"_nl", tr("荷兰语")},
    //     {"_no", tr("挪威语")},       {"_pl", tr("波兰语")},
    //     {"_pt", tr("葡萄牙语")},     {"_pt_BR", tr("巴西葡萄牙语")},
    //     {"_ro", tr("罗马尼亚语")},   {"_ru", tr("俄语")},
    //     {"_sk", tr("斯洛伐克语")},   {"_sl", tr("斯洛文尼亚语")},
    //     {"_sr", tr("塞尔维亚语")},   {"_sv", tr("瑞典语")},
    //     {"_th", tr("泰语")},         {"_tr", tr("土耳其语")},
    //     {"_uk", tr("乌克兰语")},     {"_vi", tr("越南语")},
    //     {"_zh_CN", tr("简体中文")},  {"_zh_TW", tr("繁体中文")}};

private slots:
    void onCompileTsFilesFinished();
};
