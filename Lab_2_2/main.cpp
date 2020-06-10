#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library
#include <opencv2/bgsegm.hpp>  // library for the bg segmentation

using namespace cv;

int main(){

    Mat frame;
    Mat frame_gray;
    Mat motion_mask;
    Mat bg;
    
    // pointer to MOG
    //Ptr<bgsegm::BackgroundSubtractorMOG> pMOG; // mixed o gaussian

    // pointer to MOG2
    Ptr<BackgroundSubtractorMOG2> pMOG2; // mixed o gaussian

    int history = 300;  // how many frames are used to compute the weights for the gaussians
    int nmixture = 5; // how many gaussian to model bg and foreground
    double backgroundRatio = 0.7; // threshold bg fg

    double learningRate = 0.1; // how fast the weights are updated

    double noiseSignal = 10; // to be more robust to light noise (tackle brightness changes)

    //pMOG = bgsegm::createBackgroundSubtractorMOG(history,nmixture,backgroundRatio,noiseSignal);

    // mog2 is able to detect the shadows -> remove them
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0),false); //3^ false or true detection of the shadows

    VideoCapture cap(0);

    if (!cap.isOpened()){
        return 0;
    }

    for (int i=0; i<1000; i++){
        cap >> frame; // store each frame in the the 'frame' variable

        // we don't have to use gray scale -> it works also with colored images

        // apply MOG to current frame
        //pMOG->apply(frame,motion_mask,learningRate);
        pMOG2 -> apply(frame,motion_mask,learningRate);
        pMOG2 -> getBackgroundImage(bg);

        imshow("frame",frame);
        imshow("background",bg);
        imshow("MOG",motion_mask);
        waitKey(1);
    }

    return 0;
}


