#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

String face_cascade_name = "/home/mmlab/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String face_cascade_profile = "/home/mmlab/opencv/data/haarcascades/haarcascade_profileface.xml";
CascadeClassifier face_cascade_1, face_cascade_2; // Class for the cascade classifier


int main()
{

    // VideoCapture cap("vtest.avi");
    VideoCapture cap(0);
    Mat frame, frame_gray;

    // load the classifier [method in CascadeClassifier]
    if(!face_cascade_1.load(face_cascade_name) | !face_cascade_2.load(face_cascade_profile)){
        return -1;
    }

    for(int i=0; i<1000; i++){
        cap >> frame;

        // apply calssifier to each frame
        vector<Rect> faces_1, faces_2;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // image equalization to increse the performances
        equalizeHist(frame_gray,frame_gray);

        // face detection [method in the class] + scale handling
        face_cascade_1.detectMultiScale(frame_gray,faces_1,1.1,2,0|CASCADE_SCALE_IMAGE,Size(30,30));

        // display the results
        for(int j=0; j<faces_1.size(); j++){
            Point center(faces_1[j].x + faces_1[j].width/2.0, faces_1[j].y + faces_1[j].height/2.0);
            ellipse(frame, center, Size(faces_1[j].width*0.5,faces_1[j].height*0.5), 0,0,360, Scalar(255,0,0),4,8,0);
        }

        // face detection [method in the class] + profile
        face_cascade_2.detectMultiScale(frame_gray,faces_2,1.1,2,0|CASCADE_SCALE_IMAGE,Size(30,30));

                // display the results
        for(int j=0; j<faces_2.size(); j++){
            Point center(faces_2[j].x + faces_2[j].width/2.0, faces_2[j].y + faces_2[j].height/2.0);
            ellipse(frame, center, Size(faces_2[j].width*0.5,faces_2[j].height*0.5), 0,0,360, Scalar(0,255,0),3,8,0);
        }

        imshow("Video",frame);
        if(waitKey(1)==27){
            return 0;
        }
    }
    return 0;
}