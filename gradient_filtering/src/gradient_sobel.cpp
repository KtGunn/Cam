#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//using namespace std;

// GAUSSIAN smoothing
// copied from file 'guassian_smoothing.cpp'
//
void gaussianSmoothing1(cv::Mat& rfImage, cv::Mat& rfSmoothed)
{
    // create filter kernel
    float gauss_data[25] = {1, 4, 7, 4, 1,
                            4, 16, 26, 16, 4,
                            7, 26, 41, 26, 7,
                            4, 16, 26, 16, 4,
                            1, 4, 7, 4, 1};

    // Must normalize the kernel
    float sum = 0;
    for (short n=0; n<25; n++)
	sum += gauss_data[n];
    
    cv::Mat kernel = cv::Mat (5, 5, CV_32F, gauss_data)/sum;

    // apply filter
    int ddepth = -1; // '-1' means same depth as source
    cv::filter2D (rfImage, rfSmoothed, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    return;
}


void ocv_gradientSobel (cv::Mat& rfImage)
{
    // Compute gradient in x-directin
    float sobel_x[9] = {-1,0,1 -2,0,2, -1,0,1}; // This is normalized
    cv::Mat kernel_x = cv::Mat (3,3, CV_32F, sobel_x);
    cv::Mat result_x;
    cv::filter2D (rfImage, result_x, -1, kernel_x, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);


    // Compute combined xy-direction gradient
    float sobel_y[9] = {-1,-2,-1, 0,0,0, 1,2,1}; // This is normalized
    cv::Mat kernel_y = cv::Mat (3,3, CV_32F, sobel_y);
    cv::Mat result_xy;
    cv::filter2D (result_x, result_xy, -1, kernel_y, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
    
    std::string windowName = "OpenCV xy-Sobel";
    cv::namedWindow (windowName, 1); // create window
    cv::imshow (windowName, result_xy);
    cv::waitKey (0); // wait for keyboard input before continuing
    cv::destroyWindow (windowName);

    // Now compute gradient in y-direction
    cv::Mat result_y;
    cv::filter2D (rfImage, result_y, -1, kernel_y, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);
    cv::Mat mMagn (rfImage.size(), CV_8UC1, cv::Scalar(255));

    for (ushort row=0; row<mMagn.rows; row++) {
	for (ushort col=0; col<mMagn.cols; col++) {

	    // Since I'm using ref/point I must stick with pointer
	    // access arithmetic
	    int index = rfImage.step[0]*row + rfImage.step[1]*col;
	    uchar rx = *(result_x.data + index);
	    uchar ry = *(result_y.data + index);

	    mMagn.at<uchar>(row,col) = sqrt (rx*rx + ry*ry);
	}
    }

    windowName = "Magnitude";
    cv::namedWindow (windowName, 1); // create window
    cv::imshow (windowName, mMagn);
    cv::waitKey (0); // wait for keyboard input before continuing
    cv::destroyWindow (windowName);


    return;
}

void directional_gradient (cv::Mat& mIn, cv::Mat& mOut, float* pKernel)
{
    int cols = mIn.cols;
    int rows = mIn.rows;
    uchar* pData = mIn.data;

    std::cout << " mIn.step[1] = " << mIn.step[1] << std::endl;
    
    float norm;
    for (short n=0;n<9;n++)
	norm += abs(*(pKernel+n));

    cv::Mat mDer (mIn.size(), CV_32FC1, cv::Scalar(255));
    for (short r=1; r<(rows-0); r++) {
	
	for (short c=1; c<(cols-0); c++) {

	    float sum=0;
	    for (short row=0; row<3; row++) {
		for (short col=0; col<3; col++) {
		    int base = mIn.step[0]*(r-1+row) + mIn.step[1]*(c-1+col);

		    float imgValue = (float)(*(pData+base));
		    float kernValue = *(pKernel+row*3+col);
		    //sum += (*(pData+base)) * (*(pKernel+row*3+col));
		    sum += imgValue*kernValue;
		}
	    }
	    //mOut.at<uchar>(r,c) = (uchar)sum;
	    //sum /= norm;
	    //sum = sum > 255 ? 255 : sum;
	    //sum = sum < 0 ? 0 : sum;
	    //mOut.at<uchar>(r,c) = sum;
	    mDer.at<float>(r,c) = sum;
	}
    }
    mDer.convertTo (mOut, CV_8UC1);
}

void mygradientSobel (cv::Mat& rfImage)
{
    cv::Mat mSmoothed;
    gaussianSmoothing1 (rfImage, mSmoothed);
    //mSmoothed = rfImage;
    
    cv::Mat mDerX (rfImage.size(), rfImage.type(), cv::Scalar(255));
    float der_x[9] = {-1,0,-1,  -2,0,2,  -1,0,-1};

    directional_gradient (mSmoothed, mDerX, &der_x[0]);

    // show x-derative
    std::string windowName = "xDer";
    cv::namedWindow (windowName, 1); // create window
    cv::imshow (windowName, mDerX);
    cv::waitKey (0); // wait for keyboard input before continuing
    cv::destroyWindow (windowName);

    return;
}

void gradientSobel(cv::Mat& rfImage)
{
    // TODO: Based on the image gradients in both x and y, compute an image 
    // which contains the gradient magnitude according to the equation at the 
    // beginning of this section for every pixel position. Also, apply different 
    // levels of Gaussian blurring before applying the Sobel operator and compare the results.

    cv::Mat mSmoothed;
    gaussianSmoothing1 (rfImage, mSmoothed);

    // show original
    std::string windowName = "Original image";
    cv::namedWindow (windowName, 1); // create window
    cv::imshow (windowName, rfImage);
    cv::waitKey (0); // wait for keyboard input before continuing
    cv::destroyWindow (windowName);

    // show result
    windowName = "Gaussian Blurring";
    cv::namedWindow (windowName, 1); // create window
    cv::imshow (windowName, mSmoothed);
    cv::waitKey (0); // wait for keyboard input before continuing
    cv::destroyWindow (windowName);
    
    ocv_gradientSobel (mSmoothed);

    return;
}


int main()
{
    std::cout << " Gradient Sobel\n";

    // load image from file
    cv::Mat img;
    img = cv::imread ("../images/img1gray.png");
    std::cout << " Image type = " << img.type() << std::endl;

    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    std::cout << " Gray Image type = " << imgGray.type() << std::endl;
    
    //gradientSobel(img);
    mygradientSobel(imgGray);
}
