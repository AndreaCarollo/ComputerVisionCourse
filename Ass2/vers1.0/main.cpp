#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>

#include <iostream>
#include <stdio.h>
#include <vector>
#include <climits>
#include "./mylib.hpp"

using namespace cv;
using namespace std;


// GLOBAL ENVIROMENTS
// path to the sequence of images
string path_to_img = "./Video/img1/%06d.jpg";

// structure to give a label to each rectangle of detection
struct obj_det{
    Rect rect;
    int ID;
    int f_lost = -1;
    bool mod = 0;
    Mat hist;
    Mat histimg = Mat::zeros(400, 640, CV_8UC3);
    // TODO: add MASK field -> taken from the BG subtraction
    //       add HIST fieldw
};

// main function
int main(){
    // create a matrix for the frame rgb and gray
    Mat frame;

    // load the sequence of images
    VideoCapture sequence(path_to_img);

    // set parameters for the HOG descriptor
    HOGDescriptor hog(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    // load Daimler People Detector pre trained classifier with previous parameters
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector()); 
    vector<Rect> pedestrian_prev;
    vector<Rect> list_of_pedestrian;
    vector<obj_det> lost_pedestrian;
    // vector<int> x_prev;
    // vector<int> y_prev;
    // vector<CvHistogram> lost_hist;
    // vector<CvHistogram> tmp_occlusion;
    int number_of_ID, number_of_ID_prev;
    vector<obj_det> lists_finded;
    vector<obj_det> lists_finded_prev;

    /* ***** variable for MOG ***** */
    Mat frame_gray;
    Mat motion_mask;
    Mat bg;
    // pointer to MOG2
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    int history = 25; 
    int nmixture = 5;
    double backgroundRatio = 0.7;
    double learningRate = 0.1;
    double noiseSignal = 10;
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0),false); 

    /* ***** Parameters for Erosion & Dilation ***** */
    int erosion_elem = 1;
    int erosion_size = 1;
    int dilation_elem = 3;
    int dilation_size = 5;
    int const max_elem = 4;
    int const max_kernel_size = 21;
    Mat mask_hist, hsv, hue;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

    // // open log file
    // FILE * log_file = fopen("./log.txt","w");

    /* **** Start main for cicle **** */
    for (int frame_idx = 0; frame_idx<1000; frame_idx++){

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
        Mat element2 = getStructuringElement( MORPH_ELLIPSE,
                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                        Point( dilation_size, dilation_size ) );
        dilate( mask_hist, mask_hist, element2 );


        // apply HOG classifier to each frame
        vector<Rect> pedestrian;
        hog.detectMultiScale(frame,pedestrian, 0, Size(8,8), Size(), 1.05, 2, true);

        // save pedestrian to pedestrian_prev
        pedestrian_prev = pedestrian;   //TODO: check if remove
        number_of_ID_prev = number_of_ID;
        lists_finded_prev = lists_finded;
        // compute
        vector<float> descriptorValues;
        hog.compute(frame,descriptorValues);

        number_of_ID = pedestrian.size();
        if(frame_idx==0){
            // generating the lists of finded object & assign an univocal ID
            for( int k = 0; k < number_of_ID; k++){
                
                obj_det temp;
                temp.rect  = pedestrian[k];
                temp.ID = k;

                lists_finded.push_back(temp);
           }
        }else{
            // find max ID assigned to the previous list
            int max_ID = -1;
            for (int j = 0; j < number_of_ID_prev; j++){
                if( lists_finded_prev[j].ID > max_ID ){
                    max_ID = lists_finded_prev[j].ID;
                }
            }

            // generate new list of finded pedestrian
            for( int k = 0; k < number_of_ID; k++){
                
                obj_det temp;
                temp.rect  = pedestrian[k];
                temp.ID = -1;

                lists_finded.push_back(temp);
            }

            // calculate hist of each rectangle
            // for(int j = 0; j < pedestrian.size(); j++ ){

            //     // ??? Histograms
            //     // ...

            // }

            // compute the euclidian distances & assign the corrispondent ID by nearness
            for( int l = 0; l < number_of_ID; l++ ){
                int min_dist = INT_MAX;
                int min_ped_id = -1;
                // set a theshold for near position, based on the dimension of the rectangles
                int threshold_detect = sqrt(lists_finded[l].rect.width^2 + lists_finded[l].rect.height^2)/2;

                for( int j = 0; j < number_of_ID_prev; j++){

                    Point centre      = (lists_finded[l].rect.br() + lists_finded[l].rect.tl())/2;
                    Point centre_prev = (lists_finded_prev[j].rect.br() + lists_finded_prev[j].rect.tl())/2;
                    int distance      = sqrt((centre.x-centre_prev.x)^2 + (centre.y-centre_prev.y)^2 );

                    if((distance < min_dist) && (distance < threshold_detect)){

                        min_dist = distance;
                        min_ped_id = j;
                        lists_finded[min_ped_id].ID = l;
                        lists_finded[min_ped_id].mod = 1;

                    }
                }
                // TODO:  gestire il fatto se cambia il numero di detection!
            }
            for( int k = 0; k<lists_finded.size(); k++){
                if(lists_finded[k].ID=-1){
                    max_ID++;
                    lists_finded[k].ID = max_ID;
                    lists_finded[k].mod = 1;
                }
            }
            // collect the finded index
            vector<int> index_find;
            for(int k = 0; k < lists_finded.size(); k++){
                if(lists_finded[k].mod=1){
                    index_find.push_back(lists_finded[k].ID);
                }
            }
            // sort the index in ascendend mode
            sort(index_find.begin(), index_find.end());

            // populating the lost vector with unknown detection (not in index_find)
            for (int k = 0; k < lists_finded.size(); k++){
                if (std::find(index_find.begin(), index_find.end(),lists_finded[k].ID)==index_find.end()){
                    lost_pedestrian.push_back(lists_finded_prev[k]);
                }
            }

            /*
                Calcolare background subtraction per ogni frame o MOG (meglio MOG2)
                Utilizzare ROI ritornato da DaimlerPeopleDetector per ritagliare baground mask
                Calcolare istogramma con maschera


                Mat roi(hue, selection), maskroi(mask, selection);  //Selection sarà pedestrian[k], mask la maschera del BGSUB e hue l'immagine
                imshow("Maschera",maskroi);
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges); //Cambiare per usare 2 canali, non solo HUE?
                normalize(hist, hist, 0, 255, NORM_MINMAX);



            */
        //    for( int k = 0; k < lists_finded.size(); k++ ){
        //         Mat roi(hue, lists_finded[k].rect), maskroi(mask_hist, lists_finded[k].rect);  //Selection sarà pedestrian[k], mask la maschera del BGSUB e hue l'immagine
        //         imshow("Maschera",maskroi);
        //         calcHist(&roi, 1, 0, maskroi, lists_finded[k].hist, 1, &hsize, &phranges); //Cambiare per usare 2 canali, non solo HUE?
        //         normalize(lists_finded[k].hist, lists_finded[k].hist, 0, 255, NORM_MINMAX);
        //    }
            
        }

        // Draw rectangles
        for(int j=0; j<pedestrian.size(); j++){            
            //adjustRect(pedestrian[frame_idx]);
            rectangle(frame,pedestrian[j].tl(),pedestrian[j].br(),Scalar(0,0,255),2);
        }
        // show frame with rectangles
        imshow("Video",frame);
        imshow("Video mask",mask_hist);
        // show BG
        imshow("background",bg);
        // show motion mask
        imshow("MOG",motion_mask);
        waitKey(1);
    }

    return 0;
}
