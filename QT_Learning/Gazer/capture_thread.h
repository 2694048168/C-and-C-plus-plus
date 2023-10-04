#ifndef __CAPTURE_THREAD_H__
#define __CAPTURE_THREAD_H__

#include <QString>
#include <QThread>
#include <QMutex>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/background_segm.hpp>


class CaptureThread :public QThread
{
private:
	Q_OBJECT

public:
	CaptureThread(int camera, QMutex* lock);
	CaptureThread(QString videoPath, QMutex* lock);

	~CaptureThread();

	void setRunning(bool run) { running = run; };
	void startCalcFPS() { fps_calculating = true; };

	enum VideoSavingStatus
	{
		STARTING,
		STARTED,
		STOPPING,
		STOPPED
	};

	void setVideoSavingStatus(VideoSavingStatus status) { video_saving_status = status; };
	void setMotionDetectingStatus(bool status)
	{
		motion_detecting_status = status;
		motion_detected = false;
		if (video_saving_status != STOPPED) video_saving_status = STOPPING;
	};

protected:
	// The run method of QThread is the starting point for a thread.
	// When we call the start method of a thread, 
	// its run method will be called after the new thread is created.
	void run() override;

signals:
	void frameCaptured(cv::Mat* data);
	void fpsChanged(float fps);
	void videoSaved(QString name);

private:
	void calculateFPS(cv::VideoCapture& cap);
	void startSavingVideo(cv::Mat& firstFrame);
	void stopSavingVideo();
	void motionDetect(cv::Mat& frame);

private:
	bool running;
	int cameraID;
	QString videoPath;
	QMutex* data_lock;
	cv::Mat frame;

	// FPS calculating
	bool fps_calculating;
	float fps;

	// video saving
	int frame_width, frame_height;
	VideoSavingStatus video_saving_status;
	QString saved_video_name;
	cv::VideoWriter* video_writer;

	// motion analysis
	bool motion_detecting_status;
	bool motion_detected;
	cv::Ptr<cv::BackgroundSubtractorMOG2> segmentor;
};

#endif // !__CAPTURE_THREAD_H__
