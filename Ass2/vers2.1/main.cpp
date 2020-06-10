#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

// GLOBAL ENVIROMENTS

// structure to give a label to each rectangle of detection
struct obj_det
{
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
    for (int i = 0; i < numColors; i++)
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

int main()
{
    /*** Generic Parameters ***/
    // threshold of overlapping area of rectangles
    float lmb_overlap = 0.75;
    float lmb_non_overlap = 0.1;
    int refresh = 10;

    // set the default tracking algorithm
    // std::string trackingAlg = "KCF";

    // create the tracker
    MultiTracker trackers;
    std::vector<Ptr<Tracker>> algorithms;

    // container of the tracked objects
    vector<Rect2d> objects;

    // set input video
    string path_to_img = "./Video/img1/%06d.jpg";
    VideoCapture sequence(path_to_img);

    Mat frame;
    Mat frame_det;

    // get bounding box
    sequence >> frame;
    vector<Rect> ROIs;
    // selectROIs("tracker", frame, ROIs);

    /* ***** Parameters for the Detector ***** */
    // set parameters for the HOG descriptor
    HOGDescriptor hog(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);

    // load Daimler People Detector pre trained classifier with previous parameters
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    vector<Rect> pedestrian;
    vector<Rect> pedestrian_prev;
    vector<Rect> list_of_pedestrian;
    vector<obj_det> lost_pedestrian;

    int number_of_ID, number_of_ID_prev;
    vector<obj_det> lists_finded;
    vector<obj_det> lists_finded_prev;
    int max_ID = -1;
    int max_ID_prev = -1;

    // /* ***** variable for MOG ***** */
    // Mat frame_gray;
    // Mat motion_mask;
    // Mat bg;
    // Ptr<BackgroundSubtractorMOG2> pMOG2;
    // int history = 25;
    // int nmixture = 5;
    // double backgroundRatio = 0.7;
    // double learningRate = 0.1;
    // double noiseSignal = 10;
    // pMOG2 = createBackgroundSubtractorMOG2(history, (16.0),false);

    // /* ***** Parameters for Erosion & Dilation ***** */
    // int hsize = 16;
    // float hranges[] = {0,180};
    // const float* phranges = hranges;
    // int erosion_elem = 1;
    // int erosion_size = 1;
    // int dilation_elem = 4;
    // int dilation_size = 5;
    // int const max_elem = 4;
    // int const max_kernel_size = 21;

    int ROIs_prev_size;

    // apply HOG classifier to first frame
    hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(), 1.05, 2, true);
    // compute
    vector<float> descriptorValues;
    hog.compute(frame, descriptorValues);

    // clone pedestrian in ROIs
    ROIs = pedestrian;

    //  initialize the tracker
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        algorithms.push_back(TrackerCSRT::create());
        objects.push_back(ROIs[i]);
    }

    trackers.add(algorithms, frame, objects);
    // pedestrian_prev = pedestrian;

    // do the tracking
    printf("Start the tracking process, press ESC to quit.\n");
    for (int i; i < 795; i++)
    {
        // load frame
        sequence >> frame;
        frame.copyTo(frame_det);

        // stop the program if no more images
        if (frame.empty())
        {
            cout << "end of the video" << endl;
            return 0;
        }

        hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(), 1.05, 2, true);
        // compute
        vector<float> descriptorValues;
        hog.compute(frame, descriptorValues);

        if (i % refresh == 0)
        {
            ROIs = pedestrian;
            algorithms.clear();
            objects.clear();

            trackers.clear();
            MultiTracker newtrackers;
            
            // reinitialize the tracker
            for (size_t i = 0; i < ROIs.size(); i++)
            {

                algorithms.push_back(TrackerCSRT::create());
                objects.push_back(ROIs[i]);
            }
            newtrackers.add(algorithms, frame, objects);
            trackers = newtrackers;
        }

        trackers.update(frame);

        /*************************** GRAPHICAL STUFF *******************************/
        // draw the tracked object
        for (int j = 0; j < trackers.getObjects().size(); j++)
        {
            rectangle(frame, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
            putText(frame, to_string(trackers.getDefaultName()[j]),
                    Point(trackers.getObjects()[j].x, trackers.getObjects()[j].y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
        }
        // plot detection
        for (int j = 0; j < pedestrian.size(); j++)
        {
            rectangle(frame_det, pedestrian[j], Scalar(0, 255, 0), 2, 1);
        }
        // show image with the tracked object
        cv::imshow("tracker", frame);
        cv::imshow("detection", frame_det);

        /*********** STORE VARIABLES FOR NEXT ITERATION ****************************/
        // store previous variables
        ROIs_prev_size = ROIs.size();
        pedestrian_prev.clear();
        pedestrian_prev = pedestrian;

        //quit on ESC button
        if (waitKey(1) == 27)
            return 0;
    }
}