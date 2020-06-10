#include <opencv2/opencv.hpp>  // openCV library
#include <opencv2/highgui.hpp> // user interface library
#include <opencv2/imgproc.hpp>  // image processing

using namespace cv;
using namespace std;

int main(){

    // declaration of frames
    Mat frame, prev_frame, frame_gray, copy, copy_update, copy_dritto;
    // video capture to stream the video
    VideoCapture cap(0); // webcam

    // parameters for the algorithm
    // to store all the ...
    vector<Point2f> corners;
    vector<Point2f> prev_corners;
    vector<uchar> status; // store the state of the computation 
    vector<float> err; // each point we are computing (?)

    // how many corners
    int maxCorners = 800;
    // quality level: minimum directions with high contrast
    double qualityLevel = 0.005; // 1 = perfect quality
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

    // we want to minimize the distance between good feature to track
    // by using the LK optical flow

    for (int i=0; i<1000; i++) {
        // get the video and take each frame
        cap >> frame;
        // visualization: store in grayscale
        copy = frame.clone();
        copy_update = frame.clone();
        cvtColor(frame,frame_gray,COLOR_BGR2GRAY); // BGR?
       


        if (i<5 | i%150 == 0){

             // select GFF feature
            goodFeaturesToTrack(frame_gray, corners, maxCorners,
                            qualityLevel, minDistance, Mat(),
                            blockSize, useHarrisDetector, k); // the output can be of a size < maxCorners

        }
        else{
            calcOpticalFlowPyrLK(prev_frame, frame, prev_corners, corners, status, err);
        }

                // show the output
        int r = 3; // how big the circle
        for (int j=0; j < corners.size(); j++) { // up to the number of corners
            // circle for plotting on the copy of the frame
            circle(copy,corners[j],r,Scalar(5*j,2*j,255-j),-1,8,0);
        }
        

        prev_frame = frame.clone();
        prev_corners = corners;
        flip(copy,copy_dritto,+1);

        imshow("feature",copy_dritto);
        waitKey(1);





    }




    return 0;
}