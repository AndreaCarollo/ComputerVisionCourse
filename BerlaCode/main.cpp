#include <opencv2/opencv.hpp>  //openCV library
#include <opencv2/highgui.hpp> //user interface library
#include <opencv2/bgsegm.hpp>  // library for the bg segmentation
#include "opencv2/imgproc.hpp"
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

String face_cascade_name = "/home/mmlab/opencv/data/haarcascades/haarcascade_fullbody.xml";
String face_cascade_profile = "/home/mmlab/opencv/data/haarcascades/haarcascade_fullbody.xml";
CascadeClassifier face_cascade_1, face_cascade_2; // Class for the cascade classifier

int erosion_elem = 0;
int erosion_size = 2;
int dilation_elem = 0;
int dilation_size = 1;
int const max_elem = 2;
int const max_kernel_size = 21;

void Erosion( Mat * src, Mat * erosion_dst);
void Dilation( Mat * src, Mat * dilation_dst);

int main(){

    Mat frame, frame_people, frame_gray, frame_dil, frame_final;
    Mat motion_mask;
    Mat bg;

    // load the classifier [method in CascadeClassifier]
    if(!face_cascade_1.load(face_cascade_name) | !face_cascade_2.load(face_cascade_profile)){
        return -1;
    }
    
    // pointer to MOG
    //Ptr<bgsegm::BackgroundSubtractorMOG> pMOG; // mixed o gaussian

    // pointer to MOG2
    Ptr<BackgroundSubtractorMOG2> pMOG2; // mixed o gaussian

    int history = 700;  // how many frames are used to compute the weights for the gaussians
    int nmixture = 5; // how many gaussian to model bg and foreground
    double backgroundRatio = 0.7; // threshold bg fg

    double learningRate = 0.01; // how fast the weights are updated

    double noiseSignal = 10; // to be more robust to light noise (tackle brightness changes)

    //pMOG = bgsegm::createBackgroundSubtractorMOG(history,nmixture,backgroundRatio,noiseSignal);

    // mog2 is able to detect the shadows -> remove them
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0), true); //3^ false or true detection of the shadows

    int absorb = 1; //round(1.0 / learningRate);
    printf("absorb = %i", absorb);

    for (int i=1; i< (795 + absorb); i++){

        // apply calssifier to each frame
        vector<Rect> person_1, person_2;

        char path [23];

        sprintf(path, "Video/img1/%06i.jpg",i - absorb);

        Mat bg_basic = imread("background.jpg",1);

        if (i <= absorb){
          frame = bg_basic;
          frame_people = frame.clone();
        }else{
          frame = imread(path,1); //load the image (0 = gray scale)
          //fprintf(stdout,"%s\n",path); check path name
          frame_people = frame.clone();
        }

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // image equalization to increse the performances
        equalizeHist(frame_gray,frame_gray);

        // face detection [method in the class] + scale handling
        face_cascade_1.detectMultiScale(frame_gray,person_1,1.1,2,0|CASCADE_SCALE_IMAGE,Size(30,30));

        // display the results
        for(int j=0; j<person_1.size(); j++){
          Point center(person_1[j].x + person_1[j].width/2.0, person_1[j].y + person_1[j].height/2.0);
          ellipse(frame_people, center, Size(person_1[j].width*0.5,person_1[j].height*0.5), 0,0,360, Scalar(255,0,0),4,8,0);
        }

        // face detection [method in the class] + profile
        face_cascade_2.detectMultiScale(frame_gray,person_2,1.1,2,0|CASCADE_SCALE_IMAGE,Size(30,30));

        // display the results
        for(int j=0; j<person_2.size(); j++){
          Point center(person_2[j].x + person_2[j].width/2.0, person_2[j].y + person_2[j].height/2.0);
          ellipse(frame_people, center, Size(person_2[j].width*0.5,person_2[j].height*0.5), 0,0,360, Scalar(0,255,0),3,8,0);
        }

        // we don't have to use gray scale -> it works also with colored images

        // apply MOG to current frame
        //pMOG->apply(frame,motion_mask,learningRate);
        pMOG2 -> apply(frame,motion_mask,learningRate);
        pMOG2 -> getBackgroundImage(bg);
        Mat sub_easy = frame - bg_basic;
        
        Dilation( &motion_mask, &frame_dil);
        Erosion( &frame_dil, &frame_final);
        threshold(frame_final,frame_final,230,255,THRESH_BINARY);

        // imshow("frame",frame);
        // imshow("background",bg);
        imshow("people",frame_people);
        imshow("bg_easy",sub_easy);
        // imshow("MOG",motion_mask);
        imshow("MOG dilated and eroded",frame_final);        
        waitKey(1);
    }

    return 0;
}

void Erosion( Mat * src, Mat * erosion_dst)
{
  int erosion_type = 0;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( erosion_type,
                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                       Point( erosion_size, erosion_size ) );
  erode( *src, *erosion_dst, element );
}

void Dilation( Mat * src, Mat * dilation_dst)
{
  int dilation_type = 0;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( dilation_type,
                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       Point( dilation_size, dilation_size ) );
  dilate( *src, *dilation_dst, element );
}