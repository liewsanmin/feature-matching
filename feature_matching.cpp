#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "helper.hpp"

int main(int argc, char** argv)
{
    Timer timer;
    // Read image
    cv::Mat img_1, img_2;

    img_1 = cv::imread(argv[1]);
    if(img_1.empty() == true)
    {
        std::cout << "Image 1 could not be loaded!" << std::endl;
        return -1;
    }

    img_2 = cv::imread(argv[2]);
    if(img_2.empty() == true)
    {
        std::cout << "Image 2 could not be loaded!" << std::endl;
        return -1;
    }


// FOR FREAK
/*
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptor_1, descriptor_2;

    // Create Detector
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    detector->setHessianThreshold(minHessian); // uncomment this for SURF

    cv::Ptr<cv::xfeatures2d::FREAK> freak = cv::xfeatures2d::FREAK::create();
    //freak->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptor_1);
    //freak->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptor_2);

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    freak->compute(img_1, keypoints_1, descriptor_1);
    freak->compute(img_2, keypoints_2, descriptor_2);
*/

// FOR BRISK
/*
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptor_1, descriptor_2;

    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
    brisk->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptor_1);
    brisk->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptor_2);
*/

// FOR ORB
/*
     std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
     cv::Mat descriptor_1, descriptor_2;

     cv::Ptr<cv::ORB> orb = cv::ORB::create();
     orb->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptor_1);
     orb->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptor_2);
*/

// FOR AKAZE
/*
     std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
     cv::Mat descriptor_1, descriptor_2;

     cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
     akaze->detectAndCompute(img_1, cv::noArray(), keypoints_1, descriptor_1);
     akaze->detectAndCompute(img_2, cv::noArray(), keypoints_2, descriptor_2);
*/

// FOR MSER
/*
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptor_1, descriptor_2;

    // Create Detector
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    detector->setHessianThreshold(minHessian); // uncomment this for SURF

    cv::Ptr<cv::MSER> mser = cv::MSER::create();

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    mser->compute(img_1, keypoints_1, descriptor_1);
    mser->compute(img_2, keypoints_2, descriptor_2);
*/


// FOR STAR
/*
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptor_1, descriptor_2;

    // Create Detector
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::STAR> star = cv::xfeatures2d::STAR::create();
    star->setHessianThreshold(minHessian); // uncomment this for SURF

    //cv::Ptr<cv::STAR> star = cv::STAR::create();

    star->detect(img_1, keypoints_1);
    star->detect(img_2, keypoints_2);

    star->compute(img_1, keypoints_1, descriptor_1);
    star->compute(img_2, keypoints_2, descriptor_2);
*/


    // Create Detector
    int minHessian = 400;
    //cv::Ptr<cv::xfeatures2d::SIFT> detector = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(); // FOR SURF
    detector->setHessianThreshold(minHessian); // uncomment this for SURF

    // Detect keypoints using FLANN descriptor
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptor_1, descriptor_2;

    detector->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptor_1);
    detector->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptor_2);


    //std::cout << "Number of keypoints = " << keypoints_1.size() << std::endl;
    //std::cout << "Size of Descriptor = " << descriptor_1.size() << std::endl;

    /*
        for(int i=0; i<keypoints_1.size(); i++)
        {
            std::cout << keypoints_1[i].pt << std::endl;
        }
    */

    startTimer(&timer);

    // Match descriptors using FLANN matcher
    cv::FlannBasedMatcher matcher; // for sift, surf

    //cv::BFMatcher matcher; // for akaze, orb, brisk, freak
    std::vector<cv::DMatch> matches;
    matcher.match(descriptor_1, descriptor_2, matches);

    // Calculate max and min distance between keypoints
    double max_dist = 0, min_dist = 100;
    for(int i = 0; i < descriptor_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }
    //std::cout << "Max dist = " << max_dist << std::endl;
    //std::cout << "Min dist = " << min_dist << std::endl;

    // Get good matches
    std::vector<cv::DMatch> good_matches;

    for(int i = 0; i < descriptor_1.rows; i++)
    {
        if(matches[i].distance <= std::max(2*min_dist, 0.2))
            good_matches.push_back(matches[i]);
    }

    stopTimer(&timer);
    std::cout << "Matching time = " << elapsedTime(timer) << " seconds"<<std::endl;

    // Draw good matches
    cv::Mat img_matches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches,
                    img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), 2);

    cv::imshow("Matches", img_matches);
    cv::waitKey(0);

    return 0;
}
