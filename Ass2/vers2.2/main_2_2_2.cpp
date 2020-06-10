#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv4/opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

// GLOBAL ENVIROMENTS

// random color generator
// Fill the vector with random colors
void getRandomColors(vector<Scalar> &colors, int numColors)
{
    RNG rng(0);
    for (int i = 0; i < numColors; i++)
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

// return the count white, count black, sum of white & black
void countwhite(Mat mask, Rect src, vector<float> *output)
{
    int count_white = countNonZero(mask(src));
    int total = src.width * src.height;
    int count_black = total - count_white;

    output->push_back(count_white);
    output->push_back(count_black);
    output->push_back(total);
}

int main()
{
    // structure to give a label to each rectangle of detection
    // struct obj_det
    // {
    //     Rect rect;
    //     int64 ID;
    //     int f_lost = -1;
    //     bool mod = 0;
    //     Mat hist;
    //     Mat histimg = Mat::zeros(400, 640, CV_8UC3);
    //     Mat backproj;

    //     Rect rectC;
    //     // TODO: add MASK field -> taken from the BG subtraction
    //     //       add HIST fieldw
    // };

    /*** Generic Parameters ***/
    // threshold of overlapping area of rectangles
    float lmb_overlap = 0.75;
    float lmb_non_overlap = 0.1;
    float lmb_end_trck = 0.1;

    int refresh = 10;
    // set input video
    string path_to_img = "../Video/img1/%06d.jpg";
    VideoCapture sequence(path_to_img);

    // create matrix for frames
    Mat frame;
    Mat frame_det; // just used to plot rectangles of detection

    // read first frame
    sequence >> frame;

    /* ***** Parameters for the Detector ***** */
    // set parameters for the HOG descriptor
    HOGDescriptor hog(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    // load Daimler People Detector pre trained classifier with previous parameters
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    vector<Rect> pedestrian;
    vector<Rect> pedestrian_prev;
    vector<Rect> list_of_pedestrian;
    // vector<obj_det> lost_pedestrian;

    int number_of_ID, number_of_ID_prev;
    // vector<obj_det> lists_finded;
    // vector<obj_det> lists_finded_prev;
    int max_ID = -1;
    int max_ID_prev = -1;

    /* ***** variable for MOG ***** */
    Mat frame_gray;
    Mat motion_mask;
    Mat motion_mask_ed;
    Mat bg;
    Ptr<BackgroundSubtractorMOG2> pMOG2;
    int history = 50;
    int nmixture = 5;
    double backgroundRatio = 0.7;
    double learningRate = 0.1;
    double noiseSignal = 10;
    pMOG2 = createBackgroundSubtractorMOG2(history, (16.0), false);

    /* ***** Parameters for Erosion & Dilation ***** */
    int hsize = 16;
    float hranges[] = {0, 180};
    const float *phranges = hranges;
    int erosion_elem = 1;
    int erosion_size = 1;
    int dilation_elem = 4;
    int dilation_size = 4;
    int const max_elem = 4;
    int const max_kernel_size = 21;

    int ROIs_prev_size;

    // apply HOG classifier to first frame
    hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(), 1.05, 2, true);
    // compute
    vector<float> descriptorValues;
    hog.compute(frame, descriptorValues);

    /***** Tracker Initialization *****/
    // create the multitracker
    MultiTracker trackers;
    std::vector<Ptr<Tracker>> algorithms;
    // container of the tracked objects
    vector<Rect2d> objects;
    // tracker's Region of Interest vector
    vector<Rect> ROIs;
    // clone pedestrian in ROIs
    ROIs = pedestrian;
    //  initialize the tracker
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        algorithms.push_back(TrackerCSRT::create());
        objects.push_back(ROIs[i]);
    }
    // add object to track
    trackers.add(algorithms, frame, objects);

    /***** Start loop processing *****/
    printf("Start the tracking process, press ESC to quit.\n");
    for (int i = 0; i < 795; i++)
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

        /***** Compute Motion Mask *****/
        pMOG2->apply(frame, motion_mask, learningRate);
        pMOG2->getBackgroundImage(bg);

        /***** Erosion & Dilation of the mask ****/
        // clone motion_mask to compute erosion and dilation
        motion_mask.copyTo(motion_mask_ed);
        Mat element = getStructuringElement(MORPH_ELLIPSE,
                                            Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            Point(erosion_size, erosion_size));
        erode(motion_mask_ed, motion_mask_ed, element);
        Mat element2 = getStructuringElement(MORPH_RECT,
                                             Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                             Point(dilation_size, dilation_size));
        dilate(motion_mask_ed, motion_mask_ed, element2);

        /***** Compute Detector *****/
        hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(), 1.05, 2, true);
        vector<float> descriptorValues;
        hog.compute(frame, descriptorValues);

        /***** Refresh of the MultiTracker *****/

        // if track smthing not on the mog2 -> kill it
        // use  >> countwhite(Mat mask, Rect src) <<

        //vector<float> output;
        vector<int> to_remove;
        for (int k = 0; k < ROIs.size(); k++)
        {
            // taken the rectangle of the tracker and calculate number of white pixels in the motion mask
            // the calculate the ratio of white over the overall area of the crop
            Mat cropedImage = motion_mask_ed(Rect(trackers.getObjects()[k].tl(), trackers.getObjects()[k].br()));
            // Mat croppedImage;
            //motion_mask_ed_copy = motion_mask_ed_copy(crop);
            int white = countNonZero(cropedImage);
            int area_crop = cropedImage.rows * cropedImage.cols ;
            int black = area_crop - white;

            printf("f %03d - %06d - %06d - %06d \n", i, white, black, area_crop);

            // int white = output[1];
            // int black = output[2];
            // int sum_wb = output[3];

            if (white / area_crop < lmb_end_trck)
            {
                to_remove.push_back(k);
                //printf("remove\n");
                //waitKey();
            }
            else
            {
                //printf("\n");
            }
            //output.clear();
        }

        if (i % (refresh) == 0)
        {
            // FOR THE MOMENT IS REINITIALIZED !!
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
            rectangle(motion_mask_ed, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
            putText(frame, to_string(trackers.getDefaultName()[j]),
                    Point(trackers.getObjects()[j].x, trackers.getObjects()[j].y),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
        }
        Mat frame_trk = frame;
        // plot detection
        for (int j = 0; j < pedestrian.size(); j++)
        {
            rectangle(frame, pedestrian[j], Scalar(0, 255, 0), 2, 1);
        }

        // show detection results
        // cv::imshow("detection", frame_det);

        // show tracker results
        // cv::imshow("tracker", frame_trk);

        // show image with the tracked object
        cv::imshow("tracker + detector", frame);

        // export image
        char nomefile_w[30];
        sprintf(nomefile_w, "./Videoout/%06d.jpg", i);
        cv::imwrite(nomefile_w, frame);

        // show BG
        // imshow("background", bg);

        // show motion mask
        // imshow("MOG", motion_mask);

        // show motion mask after erosion & dilation
        imshow("MOG ED", motion_mask_ed);

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