#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library

using namespace cv;

int main(){
    // how to display a video
    //video capture 
    VideoCapture cap(0);
    // VideoCapture cap("Video.mp4"); // to read the video   
    // VideoCapture cap(0); // to read the web cam

    Mat frame;

    // if the video is not opened
    if (!cap.isOpened()){
        return 0;
    }

    for (int i=0; i < 1000; i++){ //1000 frames
        cap >> frame; // put the capture into the frame

        imshow("frame",frame); // display the frame
        waitKey(1);// how much to wait between two frame

    }

    return 0;
}


