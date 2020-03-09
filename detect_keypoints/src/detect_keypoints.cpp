#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// [ktg]
//using namespace std;

void detKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Shi-Tomasi detector
    int blockSize = 6;       //  size of a block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max (1.0, minDistance); // max. num. of keypoints
    double qualityLevel = 0.01;                                   // minimal accepted quality of image corners
    double k = 0.04;
    bool useHarris = false;

    std::vector<cv::KeyPoint> kptsShiTomasi;
    std::vector<cv::Point2f> corners;
    double t = (double)cv::getTickCount();
    cv::goodFeaturesToTrack(imgGray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Shi-Tomasi with n= " << corners.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    for (auto it = corners.begin(); it != corners.end(); ++it)
    { // add corners to result vector

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        kptsShiTomasi.push_back(newKeyPoint);
    }

    // visualize results
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string windowName = "Shi-Tomasi Results";
    cv::namedWindow(windowName, 1);
    imshow (windowName, visImage);
    cv::waitKey (0);
    
    // TODO: use the OpenCV library to add the FAST detector
    // in addition to the already implemented Shi-Tomasi 
    // detector and compare both algorithms with regard to 
    // (a) number of keypoints, (b) distribution of 
    // keypoints over the image and (c) processing speed.

    int threshold = 30; // intensity discriminator between target pixel and neighbors
    bool doNonMaxSuppress = true;
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create (threshold, doNonMaxSuppress, type);

    std::vector<cv::KeyPoint> vKeyPtsFAST;
    t = (double)cv::getTickCount();    
    detector->detect (imgGray, vKeyPtsFAST);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "FAST with n= " <<  vKeyPtsFAST.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

    cv::Mat vizImage = img.clone ();
    cv::drawKeypoints (img, vKeyPtsFAST, vizImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow ("Fast window", 2);
    cv::imshow ("Fast window", vizImage);
    cv::waitKey (0);

    return;
}

int main()
{
    std::cout << "Callng detKeypoints1()\n";
    detKeypoints1();
}
