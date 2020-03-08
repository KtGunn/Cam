#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

// [KTG]
//using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat imgIn;
    imgIn = cv::imread("../images/img1.png");

    cv::Mat img;
    cv::cvtColor(imgIn, img, cv::COLOR_BGR2GRAY); // convert to grayscale
    
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)

    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    
    // 'img'=input, 'dst'=output
    cv::cornerHarris (img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);

    // Scale-normalize the harris resonse matrix ; otuput remains CV_32F1
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // Convert to CV_8U format
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    
    // visualize results
    std::string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);
    
    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    std::cout << " Type of dst_norm = " << dst_norm.type() << std::endl;
    std::cout << " Figuring out local maxima:\n";
    
    double maxOverlap = 1.0; // 0->1kp; 1->299kp
    
    int cMin = 0;
    std::vector<cv::KeyPoint> vKeyPts;
    for (size_t r=0; r < dst_norm.rows; r++)
    {
	for (size_t c=0; c < dst_norm.cols; c++)
	{
	    // Test the harris response
	    int harrisValue = (int)dst_norm.at<float>(r,c);
	    if (harrisValue > minResponse)
	    {
		++cMin;

		// Create a keypoint
		cv::KeyPoint newKp;
		newKp.pt = cv::Point2f(c,r); //NOTE: Point2f(x,y)!
		newKp.size = 2*apertureSize;
		newKp.response = harrisValue;

		// Test for overlap
		bool doesOverlap = false;
		if (true) {
		    for (auto it=vKeyPts.begin(); it != vKeyPts.end() ;it++)
		    {
			// If there's overlap
			double overlap = cv::KeyPoint::overlap (newKp, *it);
			if (overlap >= maxOverlap)
			{
			    // There is overlap
			    doesOverlap = true;
			    if (newKp.response > (*it).response)
			    {
				*it = newKp;
				break;
			    }
			}
		    }
		}
		// If there was no overlap, we add this point
		// Note, if there was overlap, we may have replaced an
		// exisiting key point
		if ( doesOverlap == false ) {
		    vKeyPts.push_back (newKp);
		}
	    }
	}
    }

    std::cout << " Count of local maxima = " <<  vKeyPts.size() << std::endl;
    std::cout << " Count of lover min = " <<  cMin << std::endl;

    // Now visualize the keypoints
    std::string keyPointWindow ("Harris KeyPts");
    cv::namedWindow (keyPointWindow, 5);
    cv::Mat vizImage = dst_norm_scaled.clone();
    //cv::drawKeypoints (dst_norm_scaled, vKeyPts, vizImage, cv::Scalar::all(-1),
    //		       cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints (dst_norm_scaled, vKeyPts, vizImage, cv::Scalar::all(-1),
    		       cv::DrawMatchesFlags::DEFAULT);
    cv::imshow (keyPointWindow, vizImage);
    cv::waitKey (0);
}

int main()
{
    std::cout << " Harris corner detector:\n";
    cornernessHarris();
}
