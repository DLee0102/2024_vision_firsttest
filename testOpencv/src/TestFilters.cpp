#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat inputimg = imread(argv[1], IMREAD_COLOR);
    Mat outputImg;

    // 展示原图
    imshow("Origin", inputimg);
    waitKey(0);

    // 均值滤波
    blur( inputimg, outputImg, Size( 3, 3 ), Point(-1, -1) );
    imshow("Blur", outputImg);
    waitKey(0);

    // 高斯平滑
    GaussianBlur( inputimg, outputImg, Size( 3, 3 ), 0, 0 );
    imshow("GaussianBlur", outputImg);
    waitKey(0);

    // 中值平滑
    medianBlur ( inputimg, outputImg, 3 );
    imshow("medianBlur", outputImg);
    waitKey(0);

    // 双边平滑
    bilateralFilter ( inputimg, outputImg, 4, 4*2, 4/2 );
    imshow("bilateralFilter", outputImg);
    waitKey(0);
}