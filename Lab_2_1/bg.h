#ifndef BG_H_

#define BG_H_

#include <opencv2/opencv.hpp>  //openCV library

using namespace cv;

void bg_train(Mat frame, Mat* bg);

void bg_update(Mat frame, Mat* bg, float alpha);

#endif