#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/gpu/gpu.hpp>

using namespace cv;

namespace fastcode{
    void maskShowCaller(Mat & mask, Mat & mask4show, cuda::Stream stream);
}
