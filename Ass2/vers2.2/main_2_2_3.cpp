#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

// GLOBAL ENVIROMENTS
// ....
// ....
//////////////////////

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
    // output log
    FILE *pFile;
    pFile = fopen("./Output/log.txt", "w");

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
    // MultiTracker_Alt Long_term_tracker;
    // Long_term_tracker.trackers. ;
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
    for (int i = 0; i < 1000; i++)
    {
        // load frame
        sequence >> frame;
        frame.copyTo(frame_det);
        Mat test;
        frame.copyTo(test);

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
        // first save all statement of the tracker
        for (int k = 0; k < ROIs.size(); k++)
        {
            /* code */
            std::string stato_trakers;

            int tmp_id = trackers.getDefaultName()[k];
            int tmp_x = trackers.getObjects()[k].x;
            int tmp_y = trackers.getObjects()[k].y;
            int tmp_w = trackers.getObjects()[k].width;
            int tmp_h = trackers.getObjects()[k].height;

            fprintf(pFile, "%03d,%04d,%03d,%03d,%03d,%03d\n", i, tmp_id, tmp_x, tmp_y, tmp_w, tmp_h);
        }

        // if track smthing not on the mog2 -> kill it
        // use  >> countwhite(Mat mask, Rect src) <<

        //vector<float> output;
        vector<int> tkr_to_remove;
        for (int k = 0; k < trackers.getObjects().size(); k++)
        {
            // taken the rectangle of the tracker and calculate number of white pixels in the motion mask
            // the calculate the ratio of white over the overall area of the crop
            Mat motion_mask_ed_copy = motion_mask_ed;
            Mat cropedImage;
            Point2i tmp_tl;
            tmp_tl.x = (int)trackers.getObjects()[k].tl().x;
            tmp_tl.y = (int)trackers.getObjects()[k].tl().y;
            Point2i tmp_br;
            tmp_br.x = (int)trackers.getObjects()[k].br().x;
            tmp_br.y = (int)trackers.getObjects()[k].br().y;
            Rect crop = Rect(tmp_tl, tmp_br);
            //printf("f %03d, tl(%03d, %03d) - br(%03d, %03d) - ", i, tmp_tl.x, tmp_tl.y, tmp_br.x, tmp_br.y);
            // cropedImage = motion_mask_ed_copy( crop );

            Mat *ROI;
            try
            {
                ROI = new Mat(motion_mask_ed_copy, crop);
            }
            catch (cv::Exception exc)
            {
                continue;
            }
            if (ROI == NULL)
                continue;
            ROI->copyTo(cropedImage);
            delete ROI;
            // if(i>140){
            //     imshow("maskcrop", cropedImage);
            //     waitKey();
            // }

            // Mat croppedImage;
            //motion_mask_ed_copy = motion_mask_ed_copy(crop);
            int white = countNonZero(cropedImage);
            int area_crop = cropedImage.rows * cropedImage.cols;
            int black = area_crop - white;

            //printf("%06d - %06d - %06d \n", white, black, area_crop);
            // tkr_to_remove.push_back(k);
            if ((float)white / (float)area_crop < (float)lmb_end_trck)
            {
                tkr_to_remove.push_back(k);
                //printf("remove\n");
                //waitKey();
            }
        }

        vector<Rect> new_ROIs;
        std::vector<Ptr<Tracker>> new_algorithms;
        vector<Rect2d> new_objects;

        vector<int> to_jump;
        int pippo = trackers.getObjects().size();
        int ppp = -1;
        // find new pedestrian
        trackers.clear();
        for (int k = 0; k < pedestrian.size(); k++)
        {
            bool add_flag = false;
            for (int j = 0; j < pippo; j++)
            {
                Point2i tmp_tl;
                tmp_tl.x = (int)trackers.getObjects()[j].tl().x;
                tmp_tl.y = (int)trackers.getObjects()[j].tl().y;
                Point2i tmp_br;
                tmp_br.x = (int)trackers.getObjects()[j].br().x;
                tmp_br.y = (int)trackers.getObjects()[j].br().y;
                Rect rettangolo = Rect(tmp_tl, tmp_br);
                Rect intersection = pedestrian[k] & rettangolo;
                if (intersection.area() > 0)
                {
                    //to_jump.push_back(k);
                    // cout << "to jump" << endl;
                    add_flag = false;
                    // j = pippo +10;
                    break;
                }
                else
                {
                    add_flag = true;
                }
                //else // not overlapped rectangle
                //{

                // j = pippo + 1;

                // trackers.add(TrackerCSRT::create(), frame, pedestrian[k]);

                //trackers.add(TrackerCSRT::create(), frame, pedestrian[k]);
                // bool jump_flag = false;
                // for (int n = 0; n < to_jump.size(); n++)
                // {
                //     if (k == to_jump[n])
                //     {
                //         jump_flag = true;

                //         // n = to_jump.size();
                //     }
                // }
                // if (!jump_flag)
                // {
                //     trackers.add(TrackerCSRT::create(), frame, pedestrian[k]);
                // }
                //}
            }
            if (add_flag = true)
            {
                new_ROIs.push_back(pedestrian[k]);
                // algorithms.push_back(TrackerCSRT::create());
            }
        }
        cout << i << "-" << ppp << endl;

        for (int p = 0; p < new_ROIs.size(); p++)
        {
            trackers.add(algorithms[0], frame, new_ROIs[p]);
        }

        // trackers.add(new_algorithms, frame, new_objects);
        // if (i % (refresh) == 0 && !pedestrian.empty())
        // {
        //     // FOR THE MOMENT IS REINITIALIZED !!
        //     algorithms.clear();
        //     objects.clear();

        //     trackers.clear();
        //     MultiTracker newtrackers;

        //     // reinitialize the tracker
        //     for (size_t i = 0; i < ROIs.size(); i++)
        //     {
        //         algorithms.push_back(TrackerCSRT::create());
        //         objects.push_back(ROIs[i]);
        //     }
        //     newtrackers.add(algorithms, frame, objects);
        //     trackers = newtrackers;
        // }

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
        sprintf(nomefile_w, "./Output/Videoout/%06d.jpg", i);
        cv::imwrite(nomefile_w, frame);

        // show BG
        // imshow("background", bg);

        // show motion mask
        // imshow("MOG", motion_mask);

        // show motion mask after erosion & dilation
        // imshow("MOG ED", motion_mask_ed);

        /*********** STORE VARIABLES FOR NEXT ITERATION ****************************/
        // store previous variables
        //ROIs_prev_size = ROIs.size();
        pedestrian_prev = pedestrian;

        //quit on ESC button
        if (waitKey(5) == 27)
            return 0;
    }
}