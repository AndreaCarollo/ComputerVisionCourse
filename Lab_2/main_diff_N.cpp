#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library

using namespace cv;

int main(){
    // out to display a video
    //video capture 
    VideoCapture cap("Video.mp4");
    // VideoCapture cap(0); // to read the web cam

    Mat frame;
    Mat frame_gray;
    Mat* frames = new Mat[1000];
    Mat motion_mask;
    int N = 15;

    // if the video is not open
    if (!cap.isOpened()){
        return 0;
    }

    for (int i=0; i < 1000; i++){ //1000 frames

        cap >> frame; // put the capture into the frame

        cvtColor(frame,frame_gray, COLOR_RGB2GRAY); //convert in gray scale image, 3^ integer: which conversion

        if (i > N){ // we cannot do it for the first frame

        // frame_gray = frame_gray - prev_frame;
        absdiff(frame_gray,frames[i-N],motion_mask); // asbsolute difference between frames, 3^ where to store
        
        imshow("motion_mask",motion_mask); // display the motion_mask
        }

        frame_gray.copyTo(frames[i]); //store inside the previous frame

        imshow("frame",frame);
        waitKey(1);// how much to wait between two frame

    }

    delete[]frames;

    return 0;
}


