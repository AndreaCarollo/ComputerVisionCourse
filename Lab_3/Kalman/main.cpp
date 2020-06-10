#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// struct with x y coordinates
struct mouse_info_struct { int x,y; };
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;

vector<Point> mousev,kalmanv;

// get x y
void on_mouse(int event, int x, int y, int flags, void* param) {
	{
		last_mouse = mouse_info;
		mouse_info.x = x;
		mouse_info.y = y;

	}
}

// plot points: cross in the position
#define drawCross( center, color, d )                                 \
line( img, Point( center.x - d, center.y - d ),                \
Point( center.x + d, center.y + d ), color, 2, CV_8U, 0); \
line( img, Point( center.x + d, center.y - d ),                \
Point( center.x - d, center.y + d ), color, 2, CV_8U, 0 )

int main () {
    Mat img(1000,1000, CV_8UC3); // image of int with 3 channel

	KalmanFilter KF(6, 2, 0); // the size of the states, measurement of the states (positions)

	Mat_<float> state(6,1); // x, y, v_x, v_y
	Mat processNoise(6,1, CV_32F); // x, y, v_x, v_y
	// different way to initialize

	Mat_<float> measurement(2,1); // x and y measurements

	measurement.setTo(0); // initial value for the filter


	// create a window and allow interaction with the mouse
	namedWindow("mouse kalman");
	setMouseCallback("mouse kalman",on_mouse,0);

	KF.statePre.at<float>(0) = mouse_info.x;
	KF.statePre.at<float>(1) = mouse_info.y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;

	KF.transitionMatrix = (Mat_<float>(6,6)<<
	                        1,0,1,0,0.5, 0,
							0,1,0,1, 0, 0.5,
							0,0,1,0, 1,  0,
							0,0,0,1, 0,  1,
							0,0,0,0, 1,  0,
							0,0,0,0, 0,  1);

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov,Scalar::all(0.1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	// vector for visualization
	mousev.clear();
	kalmanv.clear();

	// run it and predict the position
	for (;;){
		// prediction of the next stage -> update internal states (statePrev)
		Mat prediction = KF.predict(); // 2,1 matrix: x and y
		Point predictPt(prediction.at<float>(0),
		                   prediction.at<float>(1));

		// measurement
		measurement(0) = mouse_info.x;
		measurement(1) = mouse_info.y;

		// plot
		Point measPt(measurement(0),measurement(1));
		mousev.push_back(measPt); // trace of the previous position

		// update
		Mat estimated = KF.correct(measurement);
		Point statePt(estimated.at<float>(0), estimated.at<float>(1));

		kalmanv.push_back(statePt);

		// plot everything
		img = Scalar::all(0); //  set image to black at each iteration

		drawCross(statePt, Scalar(255,255,255),5);
		drawCross(measPt, Scalar(255,0,0),5);
		drawCross(predictPt, Scalar(0,255,0),10);

		for (int i=0; i<mousev.size()-1; i++){
			line(img, mousev[i],mousev[i+1],Scalar(255,255,0),1);
		}

		for (int i=0; i<kalmanv.size()-1; i++){
			line(img, kalmanv[i],kalmanv[i+1],Scalar(0,255,0),1);
		}

		imshow("mouse kalman",img);
        waitKey(100);

	}


    return 0;
}

