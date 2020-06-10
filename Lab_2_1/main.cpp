#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library

#include "bg.h" //include the bg library

using namespace cv;

int main(){
    
    Mat frame, frame_gray;
    Mat bg, motion_mask, motion_mask_t;
    float alpha = 0.05;
    
    //VideoCapture cap("Video.mp4");
    VideoCapture cap(0); // to read the web cam
    // for background -> use an if i>0 to discard the very first frame (light adjustment)

    // if the video is not open
    if (!cap.isOpened()){
        return 0;
    }

    for (int i = 0; i < 1000; i++){
        
            cap >> frame;
        if (i>3){
            cvtColor(frame, frame_gray, COLOR_RGB2GRAY);
            // store the first frame as background
            bg_train(frame_gray, &bg);

            // backgroudn update
            bg_update(frame_gray, &bg, alpha);

            // bg subtraction
            absdiff(bg, frame_gray, motion_mask);
            threshold(motion_mask,motion_mask_t,50,255,THRESH_BINARY);

            // display the image
            imshow("original",frame);
            imshow("background",bg);
            imshow("motion_mask",motion_mask_t);
            waitKey(1);
        }
    }

    return 0;
}


