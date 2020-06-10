#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library

using namespace cv;

int main(){
    // out to display a video
    //video capture 
    VideoCapture cap(0);
    // VideoCapture cap(0); // to read the web cam

    Mat frame;
    Mat frame_gray;
    Mat bg_prev, bg_update, motion_mask;
    float alpha = 0.1;

    // if the video is not open
    if (!cap.isOpened()){
        return 0;
    }

    for (int i=0; i < 1000; i++){ //1000 frames

        cap >> frame; // put the capture into the frame

        cvtColor(frame,frame_gray, COLOR_RGB2GRAY); //convert in gray scale image, 3^ integer: which conversion

        if (i > 5){ // we cannot do it for the first frame

            bg_update = bg_prev*(1.0-alpha) + alpha*frame_gray;

            // frame_gray = frame_gray - prev_frame;
            absdiff(frame_gray,bg_update,motion_mask); // asbsolute difference between frames, 3^ where to store
            imshow("motion_mask",motion_mask); // display the motion_mask
            imshow("bg",bg_update); // display the motion_mask
            bg_update.copyTo(bg_prev);
        } else if(i==5){
            frame_gray.copyTo(bg_prev);
        }

        waitKey(1);// how much to wait between two frame

    }

    return 0;
}


