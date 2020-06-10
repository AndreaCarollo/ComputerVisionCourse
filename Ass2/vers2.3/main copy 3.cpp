#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath> 

#include "./Hungarian.h"

// #include "./mylib.hpp"

using namespace cv;
using namespace std;

void ss(const char *path_truth_data, vector<Rect> *exact_points, int i);

int main()
{

    /*** Generic Parameters ***/

    // threshold of overlapping area of rectangles
    // float lmb_overlap = 0.75;
    // float lmb_non_overlap = 0.1;
    float threshold_kill_BG = 0.01;
    float threshold_intersection_ratio = 0.90;

    // Output log
    FILE *pFile;
    pFile = fopen("./Output/log.txt", "w");

    // Stuff for metric
    //FILE *GTFile;
    // char * path_gt = "../Video/gt/gt.csv";
    //GTFile = fopen("../Video/gt/gt.csv", "r");

    // TMP refresh tracker
    int refresh_tracker = 5;

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
    int dilation_elem = 3;
    int dilation_size = 3;
    int const max_elem = 4;
    int const max_kernel_size = 21;

    // apply HOG classifier to first frame
    hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(), 1.05, 2, true);
    // compute
    vector<float> descriptorValues;
    // hog.compute(frame, descriptorValues);

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
    trackers.update(frame);

    //  list of continuative tracking positions
    vector<vector<Rect>> trajectories;
    vector<Rect> old_rectangles;

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
            break;
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
        hog.detectMultiScale(frame, pedestrian, 0, Size(8, 8), Size(16, 16), 1.05, 2, true);
        vector<float> descriptorValues;
        hog.compute(frame, descriptorValues);

        // parse pedestrian for delete too big rectangles
        vector<Rect> tmp_pedestrian;
        for (int k = 0; k < pedestrian.size(); k++)
        {
            if (!(pedestrian[k].width > 70) || !(pedestrian[k].height > 140))
            {
                tmp_pedestrian.push_back(pedestrian[k]);
            }
        }
        pedestrian = tmp_pedestrian;
        // tmp_pedestrian.clear();

        /***** Refresh of the MultiTracker *****/
        // first save all statement of the tracker

        // if track smthing not on the mog2 -> kill it
        // use  >> countwhite(Mat mask, Rect src) <<
        trackers.update(frame);
        bool flag_refresh = false;

        for (int k = 0; k < pedestrian.size(); k++)
        {
            for (int p = 0; p < trackers.getObjects().size(); p++)
            {
                if (!(((Rect)pedestrian[k] & (Rect)trackers.getObjects()[p]).area() > 0))
                {
                    flag_refresh = true;
                    break;
                }
            }
        }

        if (i % refresh_tracker == 0 || flag_refresh == true)
        {

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

                // Mat croppedImage;
                //motion_mask_ed_copy = motion_mask_ed_copy(crop);
                int white = countNonZero(cropedImage);
                int area_crop = cropedImage.rows * cropedImage.cols;
                int black = area_crop - white;

                //printf("%06d - %06d - %06d \n", white, black, area_crop);
                // tkr_to_remove.push_back(k);
                if ((float)white / (float)area_crop < (float)threshold_kill_BG)
                {
                    tkr_to_remove.push_back(k);
                    // cout << "to remove " << k << " ID " << (int)trackers.getDefaultName()[k] << endl;
                    //waitKey();
                }
            }

            std::vector<Ptr<Tracker>> new_algorithms;
            vector<Rect> new_ROIs;
            vector<Rect> old_ROIs;
            vector<Rect2d> new_objects;

            int nobj = trackers.getObjects().size();
            // cout << nobj << endl;
            // int ppp = -1;
            // find new pedestrian
            new_ROIs.clear();
            old_ROIs.clear();

            for (int k = 0; k < trackers.getObjects().size(); k++)
            {
                // jump the value of the to_remove_tkr (that are drifted somewhere)
                if (!tkr_to_remove.empty())
                {
                    /* v is non-empty */
                    // control to jump
                    if (std::find(tkr_to_remove.begin(), tkr_to_remove.end(), k) == tkr_to_remove.end())
                    {
                        // collect "old" ROIs from the tracker, by rebuilding rectangles
                        int tmp_x = trackers.getObjects()[k].x;
                        int tmp_y = trackers.getObjects()[k].y;
                        int tmp_w = trackers.getObjects()[k].width;
                        int tmp_h = trackers.getObjects()[k].height;
                        Rect tmp_rect = Rect(tmp_x, tmp_y, tmp_w, tmp_h);
                        old_ROIs.push_back(tmp_rect);
                    }
                }
                else
                {
                    // collect "old" ROIs from the tracker, by rebuilding rectangles
                    int tmp_x = trackers.getObjects()[k].x;
                    int tmp_y = trackers.getObjects()[k].y;
                    int tmp_w = trackers.getObjects()[k].width;
                    int tmp_h = trackers.getObjects()[k].height;
                    Rect tmp_rect = Rect(tmp_x, tmp_y, tmp_w, tmp_h);
                    old_ROIs.push_back(tmp_rect);
                }
            }

            // parse old ROIs in function of dimensions
            // vector<Rect> tmp_old_ROIs;
            // for (int k = 0; k < old_ROIs.size(); k++)
            // {
            //     if ( !(old_ROIs[k].width > 70) || !(old_ROIs[k].height > 140) )
            //     {
            //         tmp_old_ROIs.push_back(old_ROIs[k]);
            //     }
            // }
            // old_ROIs = tmp_old_ROIs;
            // tmp_old_ROIs.clear();
            vector<Rect> tmp_tmp_new_ROIs;
            new_ROIs = old_ROIs;
            for (int k = 0; k < new_ROIs.size(); k++)
            {
                bool add_flag = true;
                Rect r1;
                Rect r2;
                Rect r3;
                int r4;
                int r5;
                for (int l = k + 1; l < new_ROIs.size(); l++)
                {
                    r1 = new_ROIs[k];
                    r2 = new_ROIs[l];
                    r3 = r2 & r1;
                    r4 = r1.area() + r2.area() - 2 * r3.area();
                    r5 = (r1 | r2).area();

                    //  && (r4 / (float)r3.area()) > 0.9
                    if (r3.area() > 0)
                    {
                        if (r3.area() == r2.area())
                        {
                            //cout << "r2 is inside r1" << endl;
                            // tmp_tmp_new_ROIs.push_back(r2);
                            add_flag = false;
                            break;
                        }
                        else if (r3.area() == r1.area())
                        {
                            //cout << "r1 is inside r2" << endl;
                            // tmp_tmp_new_ROIs.push_back(r1);
                            add_flag = true;
                            // break;
                        }
                        else if (r5 == r1.area())
                        {
                            add_flag = false;
                            break;
                        }
                        else if (r5 == r2.area())
                        {
                            add_flag = true;
                        }
                        else if (r4 / (float)r3.area() > 0.5)
                        {
                            //cout << "Overlapping Rectangles" << endl;
                            float r1_su_r3 = (float)r1.area() / (float)r3.area();
                            float r2_su_r3 = (float)r2.area() / (float)r3.area();
                            if (r1.area() > r2.area())
                            {
                                add_flag = false;
                                break;
                                //tmp_tmp_new_ROIs.push_back(r1);
                            }
                            else
                            {
                                add_flag = true;
                            }
                        }
                        else
                        {
                            if ((float)r1.area() < (float)r2.area())
                            {
                                add_flag = false;
                                break;
                            }
                            else
                            {
                                add_flag = true;
                            }
                        }
                    }
                    else
                    {
                        //cout << "Non-overlapping Rectangles" << endl;
                        add_flag = true;
                    }
                }

                if (add_flag == true)
                {
                    tmp_tmp_new_ROIs.push_back(new_ROIs[k]);
                }
            }
            old_ROIs = tmp_tmp_new_ROIs;
            tmp_tmp_new_ROIs.clear();

            new_ROIs = old_ROIs;
            trackers.clear();

            // for loop for comparison of new pedestrian and old finded
            if (old_ROIs.size() != 0 & pedestrian.size() != 0)
            {
                bool add_flag = false;
                for (int k = 0; k < pedestrian.size(); k++)
                {
                    for (int j = 0; j < old_ROIs.size(); j++)
                    {
                        Point old_ROI_centre = (old_ROIs[j].br() + old_ROIs[j].tl()) / 2.0;
                        Point pedestrian_centre = (pedestrian[k].br() + pedestrian[k].tl()) / 2;
                        float distance = sqrt((old_ROI_centre.x - pedestrian_centre.x) ^ 2 + (old_ROI_centre.y - pedestrian_centre.y) ^ 2);
                        if (distance < 15)
                        {
                            add_flag = false;
                            break;
                        }
                        else
                        {
                            add_flag = true;
                        }
                        // calculation of the intersaction area
                        Rect intersection = pedestrian[k] & old_ROIs[j];
                        float intersection_ratio = (float)intersection.area() / (float)pedestrian[k].area();

                        if (intersection.area() > 1)
                        {
                            // exist an intersection so control all the cases:
                            if (intersection.area() == pedestrian[k].area())
                            {
                                // prev ROI is inside pedestrain[k] -> tracker better than the detector
                                add_flag = false;
                                break;
                                // it is already condidered, break if.
                            }
                            else if (intersection.area() == old_ROIs[j].area())
                            {
                                // pedestrian[k] is inside previous ROI
                                // subs the old ROI with the new one -> detection better than the tracker
                                new_ROIs[j] = pedestrian[k];
                                add_flag = false;
                                break;
                                // it is already condidered, break if.
                            }
                            else if (old_ROIs[j].area() == pedestrian[k].area())
                            {
                                add_flag = false;
                                break;
                            }
                            else
                            {
                                // Overlapping Rectangles -> consider area of intersection_ratio // threshold_intersection
                                if (intersection_ratio > threshold_intersection_ratio)
                                {
                                    // overlap is greater, means that rectangle is already consider by the tracker
                                    // new_ROIs[j] = pedestrian[k];
                                    if (old_ROIs[j].area() < pedestrian[k].area())
                                    {
                                        add_flag = false;
                                        break;
                                    }
                                    else
                                    {
                                        add_flag = true;
                                    }
                                    // it is already condidered, break if.
                                }
                                else
                                {
                                    // olverlap is lower, means that rectangle is not already consider -> new person!
                                    add_flag = true;
                                }
                            }
                        }
                        // there is no intersection so add the new rectangle
                        else
                        {
                            add_flag = true;
                        }
                    }
                    if (add_flag == true)
                    {
                        // cout << "add" << k << endl;
                        new_ROIs.push_back(pedestrian[k]);
                        // algorithms.push_back(TrackerCSRT::create());
                    }
                }
            }
            else
            {
                if (old_ROIs.size() == 0)
                {
                    new_ROIs = pedestrian;
                }
                else
                {
                    new_ROIs = pedestrian;
                }
            }

            //cout << i << "-" << ppp << endl;

            algorithms.clear();
            objects.clear();
            // ROIs.clear();
            ROIs = new_ROIs;
            MultiTracker newtrackers;

            for (int p = 0; p < ROIs.size(); p++)
            {
                algorithms.push_back(TrackerCSRT::create());
                objects.push_back(ROIs[p]);
            }

            newtrackers.add(algorithms, frame, objects);
            trackers = newtrackers;
        }

        // Metric & Print in log file
        // read the metric true file
        vector<Rect> exact_points;
        ss("./Video/gt/gt.txt", &exact_points, i+1);
        cout << "size of exact points: " << exact_points.size() << endl;
        cout << "size of ROIs        : " << ROIs.size() << endl;
        // DO HUNGARIAN STUFF
        // Build Cost Matrix, the elements are the square distances 
        // of the center of ROIs & the Exact_Point from the ground truth
        vector<vector<double>> costMatrix;
        for(int k = 0; k < exact_points.size(); k++){
            vector<double> tmp_row;
            Point exact_k = ( exact_points[k].tl() + exact_points[k].br() )/2.0;
            for(int j = 0; j < ROIs.size(); j++){
                Point ROIs_j = ( ROIs[j].tl() + ROIs[j].br() )/2.0;
                double tmp_distance =  abs((ROIs_j.x - exact_k.x)^2 + (ROIs_j.y - exact_k.y)^2);
                if(tmp_distance > 250){
                    tmp_distance = 250;
                }
                tmp_row.push_back(tmp_distance);
                cout << " " << tmp_distance;
            }
            cout << " " << endl;
            costMatrix.push_back(tmp_row);
        }        
        // cout << "After cost matrix creation, dimension is:  " << costMatrix.size() <<endl;

        // Hungarian Algorithm find best combination to minimize the costs
        // return a vector of assignments element by elements
        // if the element is "-1" means that there is no association
        HungarianAlgorithm HungAlgo;
        vector<int> assignment;
        // cout << " sono qui " << endl;
        double cost = HungAlgo.Solve(costMatrix, assignment);
        // cout << " sono qui vivo " << endl;

        for (unsigned int x = 0; x < costMatrix.size(); x++)
            std::cout << x << "," << assignment[x] << "\t";

        std::cout << "\ncost: " << cost << std::endl;


        for (int k = 0; k < ROIs.size(); k++)
        {
            // Rect tmp_rect = Rect(trackers.getObjects()[k].tl(),trackers.getObjects()[k].br());
            // new_Rectangles.push_back( tmp_rect );
            // float tmp_object;
            // int tmp_id = trackers.getDefaultName()[k];
            // int tmp_x = trackers.getObjects()[k].x;
            // int tmp_y = trackers.getObjects()[k].y;
            // int tmp_w = trackers.getObjects()[k].width;
            // int tmp_h = trackers.getObjects()[k].height;
            // Point tmp_center = (trackers.getObjects()[k].br() + trackers.getObjects()[k].tl()) / 2.0;
            // tmp_mat.push_back(tmp_center);
            // fprintf(pFile, "%d,%04d,%03d,%03d,%03d,%03d,%03d,%03d\n", i + 1, tmp_id, tmp_x, tmp_y, tmp_w, tmp_h, tmp_center.x, tmp_center.y);
        }

        // fprintf(pFile, "%d,%04d,%03d,%03d,%03d,%03d,%03d,%03d\n", i + 1, tmp_id, tmp_x, tmp_y, tmp_w, tmp_h, tmp_center.x, tmp_center.y);

        /*************************** GRAPHICAL STUFF *******************************/
        // draw the tracked object
        for (int j = 0; j < trackers.getObjects().size(); j++)
        {
            rectangle(frame, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
            rectangle(motion_mask_ed, trackers.getObjects()[j], Scalar(255, 0, 0), 2, 1);
            putText(frame, to_string(trackers.getDefaultName()[j]),
                    Point(trackers.getObjects()[j].x, trackers.getObjects()[j].y),
                    FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0), 1, 8);
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
        // namedWindow("tracker + detector", WINDOW_NORMAL);
        // cv::resizeWindow("tracker + detector", 1920, 1080);
        cv::imshow("tracker + detector", frame);
        cout << i << endl;

        // export image
        char nomefile_w[30];
        std::sprintf(nomefile_w, "./Output/Videoout/%06d.jpg", i);
        cv::imwrite(nomefile_w, frame);

        // show BG
        // imshow("background", bg);

        // show motion mask
        // imshow("MOG", motion_mask);

        // show motion mask after erosion & dilation
        // cv::imshow("MOG ED", motion_mask_ed);

        //quit on ESC button
        if (waitKey(5) == 27)
            return 0;
    }

    cout << "Start evaluation" << endl;

    // evaluation of the algorithm
    // ...
}

void ss(const char *path_truth_data, vector<Rect> *exact_points, int i)
{
    int line_len;
    size_t n;
    char *line = NULL;
    char *token = NULL;
    bool flag = true;
    vector<vector<int>> truth_people;
    int index = 0;
    int count = 0;
    FILE *truth = fopen(path_truth_data, "r"); // "../Video/gt/gt.csv"
    assert(truth != NULL);
    while (flag)
    {
        vector<int> truth_person;
        count = 0;
        line_len = getline(&line, &n, truth);
        for (int k = 0; k <= 5; k++)
        {
            token = strsep(&line, ",");
            if (count == 0)
            {
                index = atoi(token);
            }
            if (index == i) // if the index is related to the current frame
            {
                truth_person.push_back(atoi(token));
            }
            else if (atoi(token) > i & count == 0) // if the index is greater than the current frame
            {
                flag = false;
            }
            count++;
        }
        if (flag & index == i)
        {
            truth_people.push_back(truth_person); // save the truth person
        }
        while ((token = strsep(&line, ",")) != NULL)
        {
        } // finish the line to reset the token
    }
    fclose(truth);
    for (int k = 0; k < truth_people.size(); k++)
    {
        int tl = truth_people[k][2];
        int tp = truth_people[k][3];
        int w = truth_people[k][4];
        int l = truth_people[k][5];
        Rect ROI(tl, tp, w, l);
        //Point center = ( ROI.br() + ROI.tl() )/2.0 ;
        exact_points->push_back(ROI);
        // printf("person:%i -> ID:%2i lf:%i tp:%i w:%i l:%i\n",i,truth_people[k][1],truth_people[k][2],truth_people[k][3],truth_people[k][4],truth_people[k][5]);
    }
}