#ifndef __MAINWINDOWS_H__
#define __MAINWINDOWS_H__

#include <QMainWindow>
#include <QMenuBar>
#include <QToolBar>
#include <QAction>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>
#include <QLabel>
#include <QGraphicsPixmapItem>

#include "erode_plugin.h"


class MainWindow : public QMainWindow
{
private:
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow();

private:
	void initUI();
	void createActions();
	void showImage(QString path);
	void setupShortcuts();

	// plugin
	void loadPlugins();

private slots:
	void openImage();
	void zoomIn();
	void zoomOut();
	void prevImage();
	void nextImage();
	void saveAs();

	// for editting
	void blurImage();
	void pluginPerform();

private:
	QMenu* fileMenu;
	QMenu* viewMenu;
	QMenu* editMenu;

	QToolBar* fileToolBar;
	QToolBar* viewToolBar;
	QToolBar* editToolBar;

	QGraphicsScene* imageScene;
	QGraphicsView* imageView;

	QStatusBar* mainStatusBar;
	QLabel* mainStatusLabel;

	QAction* openAction;
	QAction* saveAsAction;
	QAction* exitAction;
	QAction* zoomInAction;
	QAction* zoomOutAction;
	QAction* prevAction;
	QAction* nextAction;

	QAction* blurAction;

	QGraphicsPixmapItem* currentImage;
	QString currentImagePath;

	// the QMap type to register all the loaded plugins :
	QMap<QString, EditorPluginInterface*> editPlugins;

};

#endif // !__MAINWINDOWS_H__
