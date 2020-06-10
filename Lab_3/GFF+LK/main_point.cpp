#include <opencv2/opencv.hpp>  // openCV library
#include <opencv2/highgui.hpp> // user interface library
#include <opencv2/imgproc.hpp>  // image processing

using namespace cv;
using namespace std;

int main(){

    // declaration of frames
    Mat frame, prev_frame, frame_gray, copy;
    // video capture to stream the video
    VideoCapture cap(0); // webcam

    // parameters for the algorithm
    // to store all the ...
    vector<Point2f> corners;
    vector<uchar> status; // store the state of the computation 
    vector<float> err; // each point we are computing (?)

    // how many corners
    int maxCorners = 100;
    // quality level: minimum directions with high contrast
    double qualityLevel = 0.01; // 1 = perfect quality
    // for the strenght -> minimum distance between 2 points
    double minDistance = 10;
    // group in blocks of pixels -> square of blockSize x blockSize
    int blockSize = 3;
    // Harris detector (not useful)
    bool useHarrisDetector = false;
    // weight (not useful)
    double k = 0.04;

     if (!cap.isOpened()){
        return 0;
    }

    for (int i=0; i<1000; i++) {
        // get the video and take each frame
        cap >> frame;
        // visualization: store in grayscale
        copy = frame.clone();
        cvtColor(frame,frame_gray,COLOR_BGR2GRAY); // BGR?
        // select GFF feature
        goodFeaturesToTrack(frame_gray, corners, maxCorners,
                            qualityLevel, minDistance, Mat(),
                            blockSize, useHarrisDetector, k); // the output can be of a size < maxCorners

        // show the output
        int r=4; // how big the circle
        for (int j=0; j < corners.size(); j++) { // up to the number of corners
            // circle for plotting on the copy of the frame
            circle(copy,corners[j],r,Scalar(5*j,2*j,255-j),-1,8,0);
        }

        imshow("feature",copy);
        waitKey(1);





    }




    return 0;
}