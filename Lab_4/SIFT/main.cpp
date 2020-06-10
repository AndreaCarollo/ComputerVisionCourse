#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(){

    // load
    Mat img_object = imread("box.png",0); // 0 grayscale
    Mat img_scene  = imread("box_in_scene.png",0); // 0 grayscale

    // detect keypoints using SIFT + compute the descriptor
    Ptr<SIFT> detector = SIFT::create(400); // pointer to a SIFT deterctor
    // 400 minimum quality level

    // vector to sote keypoints and descriptors
    vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

    // compute sift
    detector->detectAndCompute(img_object, noArray(), keypoints_object, descriptors_object);
    // 1 image
    // 2 no mask -> no array
    // 3&4 where to store
    detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

    // display
    Mat keyPlot1, keyPlot2;

    drawKeypoints(img_object, keypoints_object, keyPlot1, Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // input array -> image
    // keypoints
    // where to plot
    // color
    // show directions
    imshow("img1",keyPlot1);

    drawKeypoints(img_scene, keypoints_scene, keyPlot2, Scalar(0,255,0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("img2",keyPlot2);

    waitKey(0);

    // Match descriptor between the images
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;

    matcher.match(descriptors_object,descriptors_scene, matches);

    Mat img_matches;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, matches,
                img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


    // image
    imshow("Match",img_matches);
    waitKey(0);

    // get only the good matches
    vector<DMatch> good_matches;

    // filtering
    for(int i=0;i<descriptors_object.rows;i++){

        //threshold on the distance (difference between keypoints)
        if(matches[i].distance < 150){
            good_matches.push_back(matches[i]);

        }
    }

    Mat img_matches_good;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches,
                img_matches_good, Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // image
    imshow("Match_good",img_matches_good);
    waitKey(0);

    // stitching
    Mat result;

    // localize the object
    vector<Point2f> obj, scene;

    for(int i=0; i<good_matches.size(); i++){

        // get keypoints from the goof matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        // puch back -> concatenate
        // query -> obtsin the index
        // convert to pt

        // the same for the scene
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    // homografy matrix that compute the transformation
    Mat H = findHomography(obj, scene, RANSAC); // RANSAC -> understand the most parallel points

    // apply transformation
    warpPerspective(img_object, result, H, Size(img_scene.cols,img_scene.rows), INTER_CUBIC);
    // source
    // store in result
    // type of transf -> H
    // size of the output
    // interpolation to decide which pixel to replace (color) -> INTER_CUBIC

    // so far the bacground is black

    Mat result_mask = Mat::zeros(result.size(),CV_8UC1);

    result_mask.setTo(255, result=0); // put white where result is not zero

    imshow("Transformed",result_mask);
    result.copyTo(img_scene,result_mask); // copy only where the mask is black
    waitKey(0);

    imshow("Stitched",img_scene);
    waitKey(0);

    return 0;
}