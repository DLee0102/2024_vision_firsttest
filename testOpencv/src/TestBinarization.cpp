#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    Mat A = imread(argv[1], IMREAD_COLOR);
    Mat A_gray;
    Mat B;
    int threshold_value = 100;

    cvtColor( A, A_gray, COLOR_RGB2GRAY);
    imshow("Gray", A_gray);
    waitKey(0);

    threshold( A_gray, B, threshold_value, 255, 0);
    imshow("Binary", B);
    waitKey(0);
}