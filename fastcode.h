#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
namespace fastcode{
    void maskShowCaller(const Mat & mask, Mat & mask4show);
    void segResultShowCaller(const Mat & img, const Mat & mask, Mat & segResult);
    void maskBinaryCaller(const Mat & mask, Mat & maskResult);
    void thresholdCaller(const Mat & img1, const Mat & img2, const Mat & img3, Mat & maskFG, Mat & maskBG);
    void copyGMMtoGPU(double * src, double * & dst, size_t len);
    void freeGMMonGPU(double * p);
    void GMMCaller(const Mat& img1, const Mat& img2, const Mat& img3, const Mat& mask, double * GMMonGPU, Mat& fromSource, Mat& toSink, double lambda);
}
