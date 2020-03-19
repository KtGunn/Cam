#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"

using namespace std;

void showLidarTopview()
{
    std::vector<LidarPoint> lidarPoints;
    readLidarPts("../dat/C51_LidarPts_0000.dat", lidarPoints);

    // NOTE Size (width/cols, height/rows)
    //
    cv::Size worldSize (10.0, 20.0); // width and height of sensor field in m
    cv::Size imageSize (1000*2/5, 2000*2/5); // corresponding top view image in pixel

    // this is too big
    //cv::Size imageSize(1000, 2000); // corresponding top view image in pixel

    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));
    cout << " Image size = " << topviewImg.size() << endl;
    cout << " Image cols = " << topviewImg.cols << " rows = " << topviewImg.rows << endl;

    int cols = topviewImg.cols;
    int rows = topviewImg.rows;

    // For laughs & giggles
    // Putting a random background to the image
    for (int row=0; row < rows; row++) {
      for (int col=0; col < cols; col++) {
        cv::Vec3b vPixel;
        cv::randu (vPixel, cv::Scalar::all(10), cv::Scalar::all(100));
        topviewImg.at<cv::Vec3b>(row,col) = vPixel;
      }
    }



    // plot Lidar points into image
    int csize = 5;
    csize = 2;
    int outside = 0;
    int inside = 0;
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it) {
      float xw = (*it).x; // world position in m with x facing forward from sensor
      float yw = (*it).y; // world position in m with y facing left from sensor
      float zw = (*it).z;

      //  Image origin = top-left, x-horizontal, y-down
      //  World origin = bottom-center, x-up, y-left
      //
      int y = imageSize.height + (-xw * imageSize.height / worldSize.height);
      int x = imageSize.width/2 + (-yw * imageSize.height / worldSize.height) ;

      uint px = x;
      uint py = y;
      cv::Point2f wP (xw,yw);
      cv::Point2i pP (px,py);
      
      if ( x<0 || x>= cols || y<0 || y>=rows ) {
        ++outside;
      } else {
        ++inside;
      }

      // TODO: 
      // 2. Remove all Lidar points on the road surface while preserving 
      // measurements on the obstacles in the scene.
      if (zw > -1) {

        // TODO: 
        // 1. Change the color of the Lidar points such that 
        // X=0.0m corresponds to red while X=20.0m is shown as green.
        short red = (short)(255 * (1-xw/20.0));
        red = (red < 0) ? 0 : red;
        
        short green = (short)(255*xw/20.0);
        green = (green > 255) ? 255 : green;
        
        cv::Scalar color (0,green,red);
        cv::circle (topviewImg, cv::Point(x, y), csize, color, -1);
      }
    }

    cout << "Points inside=" << inside << " outside=" << outside <<endl;
    
    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    showLidarTopview();
}
