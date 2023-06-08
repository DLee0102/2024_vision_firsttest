#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/**
 * @brief 腐蚀操作
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char** argv)
{
    Mat A = imread(argv[1]);
    imshow("Origin", A);
    waitKey(0);

    Mat B;
    Mat element = getStructuringElement( MORPH_RECT, Size( 3, 3) );
    erode(A, B, element);

    imshow("Erosion", B);
    waitKey(0);
}