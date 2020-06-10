#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

// GLOBAL ENVIROMENTS
// path to the sequence of images
string path_to_img = "./Video/img1/%06d.jpg";
// path to the classifier for full body pedestrian
String pedestrian_cascade_name = "/home/mmlab/opencv/data/haarcascades/haarcascade_fullbody.xml";
// creation of the cascade classifier
CascadeClassifier pedestrian_cascade;

// main function
int main(){
    // create a matrix for the frame rgb and gray
    Mat frame, frame_gray;

    // load the sequence of images
    VideoCapture sequence(path_to_img);

    // load the classifier
    if(!pedestrian_cascade.load(pedestrian_cascade_name))
        return -1;

    // Start for cicle
    for (int i = 0; i<795; i++){

        // push current frame into the matrix
        sequence >> frame;

        // apply classifier to each frame
        vector<Rect> pedestrian;
        cvtColor(frame,frame_gray, COLOR_BGR2GRAY);

        // image equalization to increse the performances
        equalizeHist(frame_gray,frame_gray);        

        // pedestrian detection [method in the class] + scale handling
        pedestrian_cascade.detectMultiScale(frame_gray,pedestrian,1.1,2,0|CASCADE_SCALE_IMAGE,Size(20,20),Size(150,150));

        // display results of detection
        for(int j=0; j<pedestrian.size(); j++){
            Point center(pedestrian[j].x + pedestrian[j].width/2.0, pedestrian[j].y + pedestrian[j].height/2.0);
            rectangle(frame, pedestrian[i],Scalar(0,255,0),4,8,0);
        }
    
        imshow("Video", frame);
        waitKey(1);
    }



    return 0;
}
