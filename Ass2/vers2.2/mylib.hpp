#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <assert.h>

using namespace cv;
using namespace std;


void exact_points(const char * path_truth_data, vector<Rect> *exact_points, int i);