#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/cuda_devptrs.hpp"
#include "opencv2/core/cuda_types.hpp"
#include "opencv2/cvconfig.h"
#include "opencv2/cudaarithm.hpp"
#include "fastcode.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

// reference: http://stackoverflow.com/questions/24613637/custom-kernel-gpumat-with-float

namespace fastcode{
    // cuda implementation of maskShow
    __global__ void maskShowKernel(const PtrStepSz<uchar> mask, PtrStepSz<uchar> mask4show){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < mask.cols && y < mask.rows){
            switch(mask(x,y)){
                case GC_BGD:
                    mask4show(x,y) = 0;
                    break;
                case GC_PR_BGD:
                    mask4show(x,y) = 1;
                    break;
                case GC_PR_FGD:
                    mask4show(x,y) = 2;
                    break;
                case GC_FGD:
                    mask4show(x,y) = 3;
            }
        }
    }

    void maskShowCaller(const Mat & mask, Mat & mask4show){
        GpuMat gmask;
        gmask.upload(mask);
        mask4show.create(mask.size(), CV_8UC1);
        GpuMat gmask4show;
        gmask4show.upload(mask4show);

        dim3 DimBlock(16,16);
        dim3 DimGrid(static_cast<int>(std::ceil(mask.size().width /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(mask.size().height / 
                        static_cast<double>(DimBlock.y))));
        maskShowKernel<<<DimGrid, DimBlock>>>(gmask, gmask4show);
        gmask4show.download(mask4show);
    }

    // cuda implementation of segResultShow
    __global__ void segResultShowKernel(const PtrStepSz<uchar> mask, PtrStepSz<Vec3b> segResult){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < mask.cols && y < mask.rows){
            if(mask(i,j) == GC_BGD || mask(i, j) == GC_PR_BGD){
                 segResult(i,j) = segResult(i,j)/4;
            }
        }
    }

    void segResultShowCaller(const Mat & img, const Mat & mask, Mat & segResult){
        GpuMat gmask;
        gmask.upload(mask);
        img.copyTo(segResult);
        GpuMat gsegResult;
        gsegResult.upload(segResult);

        dim3 DimBlock(16,16);
        dim3 DimGrid(static_cast<int>(std::ceil(mask.size().width /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(mask.size().height / 
                        static_cast<double>(DimBlock.y))));
        segResultShowKernel<<<DimGrid, DimBlock>>>(gmask, gsegResult);
        gsegResult.download(segResult);
    }
}
