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

private slots:
	void openImage();
	void zoomIn();
	void zoomOut();
	void prevImage();
	void nextImage();
	void saveAs();

private:
	QMenu* fileMenu;
	QMenu* viewMenu;

	QToolBar* fileToolBar;
	QToolBar* viewToolBar;

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

	QGraphicsPixmapItem* currentImage;
	QString currentImagePath;
};

#endif // !__MAINWINDOWS_H__
