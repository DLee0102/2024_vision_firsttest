# OpenCV Tutorials
## Mat - The Basic Image Container
- Output image allocation for OpenCV functions is automatic (unless specified otherwise).
- You do not need to think about memory management with OpenCV's C++ interface.
- The assignment operator and the copy constructor only copy the header.
- The underlying matrix of an image may be copied using the cv::Mat::clone() and cv::Mat::copyTo() functions.