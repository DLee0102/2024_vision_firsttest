#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief 膨胀操作
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv)
{
    Mat A = imread(argv[1], IMREAD_COLOR);
    imshow("Origin", A);
    waitKey(0);

    Mat B;
    Mat element = getStructuringElement( MORPH_RECT, Size( 3, 3) );
    dilate( A, B, element );
    imshow("Dilate", B);
    waitKey(0);
}