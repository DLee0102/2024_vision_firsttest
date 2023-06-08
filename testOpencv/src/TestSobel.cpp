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
    Mat grad;
    int ddepth = CV_16S;
    int scale = 1;
    int delta = 0;
    if ( !A.data ) return -1;
    imshow( "Origin", A);
    waitKey(0);

    GaussianBlur( A, A, Size(3, 3), 0, 0, BORDER_DEFAULT); // 按照默认方法（镜像）处理边界

    cvtColor( A, A_gray, COLOR_RGB2GRAY);

    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    /// 求 X方向梯度
    Sobel( A_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    /// 求 Y方向梯度
    Sobel( A_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    imshow( "Sobel", grad );
    waitKey(0);
    imshow( "HalfSobel", abs_grad_y );
    waitKey(0);

    return 0;
}