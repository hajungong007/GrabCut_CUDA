#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
namespace fastcode{
    void maskShowCaller(const Mat & mask, Mat & mask4show);
    void segResultShowCaller(const Mat & img, const Mat & mask, Mat & segResult);
    void maskBinaryCaller(const Mat & mask, Mat & maskResult);
    void calcNWeightsCaller(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma);
}
