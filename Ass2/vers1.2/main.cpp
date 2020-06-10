#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <climits>
#include <cstring>
#include <ctime>
// #include "./mylib.hpp"

using namespace cv;
using namespace std;


// GLOBAL ENVIROMENTS

// structure to give a label to each rectangle of detection
struct obj_det{
    Rect rect;
    int64 ID;
    int f_lost = -1;
    bool mod = 0;
    Mat hist;
    Mat histimg = Mat::zeros(400, 640, CV_8UC3);
    Mat backproj;

    Rect rectC;
    // TODO: add MASK field -> taken from the BG subtraction
    //       add HIST fieldw
};

// random color generator
// Fill the vector with random colors
void getRandomColors(vector<Scalar> &colors, int numColors)
{
    RNG rng(0);
    for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255))); 
}

//static Ptr<TrackerBoosting> cv::TrackerBoosting::create( const TrackerBoosting::Params &  parameters );

// main function
int main(){
    // create a matrix for the frame rgb and gray
    Mat frame;
    // path to the sequence of images
    string path_to_img = "./Video/img1/%06d.jpg";
    // load the sequence of images
    VideoCapture sequence(path_to_img);

    /* ***** Parameters for the Detector ***** */
    // set parameters for the HOG descriptor
    HOGDescriptor hog(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    // load Daimler People Detector pre trained classifier with previous parameters
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector()); 
    vector<Rect>    pedestrian_prev;
    vector<Rect>    list_of_pedestrian;
    vector<obj_det> lost_pedestrian;
    
    int number_of_ID, number_of_ID_prev;
    vector<obj_det> lists_finded;
    vector<obj_det> lists_finded_prev;
    int max_ID = -1;
    int max_ID_prev = -1;

    /* ***** variable for MOG ***** */
    Mat frame_gray;
    Mat motion_mask;
    Mat bg;
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    int history = 25; 
    int nmixture = 5;
    double backgroundRatio = 0.7;
    double learningRate = 0.1;
    double noiseSignal = 10;
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0),false); 

    /* ***** Parameters for Erosion & Dilation ***** */
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    int erosion_elem = 1;
    int erosion_size = 1;
    int dilation_elem = 4;
    int dilation_size = 5;
    int const max_elem = 4;
    int const max_kernel_size = 21;
    /* ***** Parameters for Calculation of Histogram ***** */
    // int vmin = 10, vmax = 256, smin = 30;
    // Mat mask_hist,frame_hist, hsv, hue;
    Mat mask_hist;

    /* ***** Parameters for Tracker ***** */
    // set the default tracking algorithm
    // std::string trackingAlg = "KCF";
    // create the tracker
    Ptr<MultiTracker> multiTracker = TrackerBoosting::create();   

    vector<Rect> ROIs, bboxes;
    

    //Ptr<Tracker> multiTracker = TrackerBoosting::create();
    //string trackerType = "CSRT";

    // vector<Rect> bboxes;


    /* **** Start main for cicle **** */
    for (int frame_idx = 0; frame_idx<9999; frame_idx++){

        // push current frame into the matrix
        sequence >> frame;

        // control if the frames are finished
        if (frame.empty()){
            cout << "Finished reading: empty frame" << endl;
            break;
        }

        // geberation of the motion mask
        pMOG2 -> apply(frame,motion_mask,learningRate);
        pMOG2 -> getBackgroundImage(bg);

        // Erosion & Dilation of the mask
        motion_mask.copyTo(mask_hist);
        Mat element = getStructuringElement( MORPH_ELLIPSE,
                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                    Point( erosion_size, erosion_size ) );
        erode( mask_hist, mask_hist, element );
        Mat element2 = getStructuringElement( MORPH_RECT,
                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        Point( dilation_size, dilation_size ) );
        dilate( mask_hist, mask_hist, element2 );



        // apply HOG classifier to each frame
        vector<Rect> pedestrian;
        hog.detectMultiScale(frame,pedestrian, 0, Size(8,8), Size(), 1.05, 2, true);


        // vectors of colors for the tracking algorithm
        vector<Scalar> colors;
        bboxes =  pedestrian;
        getRandomColors(colors, bboxes.size());
        
        // // initialize multitracker
        for(int i=0; i < bboxes.size(); i++)
            cv::MultiTracker::add(multiTracker, frame, Rect2d(bboxes[i]));  
  

        // compute
        vector<float> descriptorValues;
        hog.compute(frame,descriptorValues);

        multiTracker->update(frame);

        number_of_ID = pedestrian.size();
        max_ID = -1;
        if(frame_idx==0){
            // generating the lists of finded object & assign an univocal ID
            for( int k = 0; k < number_of_ID; k++){
                
                obj_det temp;
                temp.rect   = pedestrian[k];
                temp.ID     = k;
                lists_finded.push_back(temp);
                max_ID = k;
           }
        }else{
            // find max ID assigned to the previous list
            for (int j = 0; j < number_of_ID_prev; j++){
                if( lists_finded_prev[j].ID > max_ID ){
                    max_ID = lists_finded_prev[j].ID;
                }
            }

            // generate new list of finded pedestrian
            for( int k = 0; k < number_of_ID; k++){
                
                obj_det temp;
                temp.rect   = pedestrian[k];
                temp.ID     = -1;
                lists_finded.push_back(temp);
            }


            /*
                DONE: Calcolare background subtraction per ogni frame o MOG (meglio MOG2) -> motion_mask
                
                Utilizzare ROI ritornato da DaimlerPeopleDetector per ritagliare baground mask
                Calcolare istogramma con maschera


                Mat roi(hue, selection), maskroi(mask, selection);  //Selection sarà pedestrian[k], mask la maschera del BGSUB e hue l'immagine
                imshow("Maschera",maskroi);
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges); //Cambiare per usare 2 canali, non solo HUE?
                normalize(hist, hist, 0, 255, NORM_MINMAX);

            */

            
            
        }
        
        /*** Plot Rectangles to the Frame + stamp the ID number ***/
        for( int j=0; j< lists_finded.size(); j++){
            // print Rectangles
            rectangle(frame,lists_finded[j].rect.tl(),lists_finded[j].rect.br(),Scalar(0,0,255),1);
            rectangle(frame,lists_finded[j].rectC.tl(),lists_finded[j].rectC.br(),Scalar(0,255,0),1);
            // print Label of ID
            String tmp_label = "ID " + to_string(lists_finded[j].ID) ;
            Point tmp_point = (lists_finded[j].rect.tl(),lists_finded[j].rect.br());
            putText (frame, tmp_label, tmp_point, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
        }

        /*** Show Output of Detection ***/
        namedWindow("Video", WINDOW_NORMAL);
        cv::resizeWindow("Video",1920,1080);
        // show frame with rectangles
        cv::imshow("Video",frame);
        cv::imshow("Video mask",mask_hist);
        // show BG
        cv::imshow("background",bg);
        // show motion mask
        cv::imshow("MOG",motion_mask);
        cv::waitKey(1);

        /* *** Save current vectors to prev *** */
        // pedestrian_prev = pedestrian;   //TODO: check if remove
        number_of_ID_prev = number_of_ID;
        lists_finded_prev = lists_finded;
        /* *** Clear current vectors *** */
        lists_finded.clear();
        number_of_ID = 0;
        max_ID_prev = max_ID;
    }

    return 0;
}
