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
        if(x < mask.rows && y < mask.cols){
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
        dim3 DimGrid(static_cast<int>(std::ceil(mask.size().height /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(mask.size().width / 
                        static_cast<double>(DimBlock.y))));
        maskShowKernel<<<DimGrid, DimBlock>>>(gmask, gmask4show);
        gmask4show.download(mask4show);
    }

    // cuda implementation of segResultShow
    __global__ void segResultShowKernel(const PtrStepSz<uchar> mask, PtrStepSz<uchar> segResult){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < mask.rows && y < mask.cols){
            if(mask(x,y) == GC_BGD || mask(x,y) == GC_PR_BGD){
                 segResult(x,y) = segResult(x,y) / 4;
            }
        }
    }

    void segResultShowCaller(const Mat & img, const Mat & mask, Mat & segResult){
        GpuMat gmask;
        gmask.upload(mask);
        img.copyTo(segResult);
        GpuMat gsegResult;
        gsegResult.upload(segResult);

        dim3 DimBlock(32,32);
        dim3 DimGrid(static_cast<int>(std::ceil(mask.size().height /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(mask.size().width / 
                        static_cast<double>(DimBlock.y))));
        segResultShowKernel<<<DimGrid, DimBlock>>>(gmask, gsegResult);
        gsegResult.download(segResult);
    }
    // cuda implementation of maskBinary
    __global__ void maskBinaryKernel(const PtrStepSz<uchar> mask, PtrStepSz<uchar> maskResult){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < mask.rows && y < mask.cols){
            if(mask(x,y) == GC_FGD || mask(x,y) == GC_PR_FGD){
                 maskResult(x,y) = 255;
            }else{
                 maskResult(x,y) = 0;
            }
        }
    }

    void maskBinaryCaller(const Mat & mask, Mat & maskResult){
        GpuMat gmask;
        gmask.upload(mask);
        maskResult.create(mask.size(), CV_8UC1);
        GpuMat gmaskResult;
        gmaskResult.upload(maskResult);

        dim3 DimBlock(32,32);
        dim3 DimGrid(static_cast<int>(std::ceil(mask.size().height /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(mask.size().width / 
                        static_cast<double>(DimBlock.y))));
        maskBinaryKernel<<<DimGrid, DimBlock>>>(gmask, gmaskResult);
        gmaskResult.download(maskResult);
    }
    // cuda implementation of calcNWeights

    __global__ void calcNWeightsKernel(const uchar * img, PtrStepSz<double> left, PtrStepSz<double> upleft, PtrStepSz<double> up, PtrStepSz<double> upright, double beta, double gamma, int rows, int cols){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < rows && y < cols){
            int basexy = 3 * (y * cols + x);
            int basexy_1 = 3 * ((y-1) * cols + x);
            int basex_1y_1 = basexy_1-3;
            int basex_1y = basexy-3;
            int basex_1y1 = basex_1y + 3 * cols;
            if(y-1>=0){
                temp = ((double)img[basexy]-(double)img[basexy_1]) * ((double)img[basexy]-(double)img[basexy_1]) + 
                        ((double)img[basexy+1]-(double)img[basexy_1+1]) * ((double)img[basexy+1]-(double)img[basexy_1+1]) + 
                        ((double)img[basexy+2]-(double)img[basexy_1+2]) * ((double)img[basexy+2]-(double)img[basexy_1+2]);
                left(x,y) = gamma * exp(-beta * temp);
            }else{
                left(x,y) = 0;
            }
            if(x-1>=0 && y-1>=0){
                temp = ((double)img[basexy]-(double)img[basex_1y_1]) * ((double)img[basexy]-(double)img[basex_1y_1]) + 
                        ((double)img[basexy+1]-(double)img[basex_1y_1+1]) * ((double)img[basexy+1]-(double)img[basex_1y_1+1]) + 
                        ((double)img[basexy+2]-(double)img[basex_1y_1+2]) * ((double)img[basexy+2]-(double)img[basex_1y_1+2]);
                upleft(x,y) = gamma / 1.414 * exp(-beta * temp);
            }else{
                upleft(x,y) = 0;
            }
            if(x-1>=0){
                temp = ((double)img[basexy]-(double)img[basex_1y]) * ((double)img[basexy]-(double)img[basex_1y]) + 
                        ((double)img[basexy+1]-(double)img[basex_1y+1]) * ((double)img[basexy+1]-(double)img[basex_1y+1]) + 
                        ((double)img[basexy+2]-(double)img[basex_1y+2]) * ((double)img[basexy+2]-(double)img[basex_1y+2]);
                up(x,y) = gamma * exp(-beta * temp);
            }else{
                up(x,y) = 0;
            }
            if(y+1<=cols && x-1>=0){
                temp = ((double)img[basexy]-(double)img[basex_1y1]) * ((double)img[basexy]-(double)img[basex_1y1]) + 
                        ((double)img[basexy+1]-(double)img[basex_1y1+1]) * ((double)img[basexy+1]-(double)img[basex_1y1+1]) + 
                        ((double)img[basexy+2]-(double)img[basex_1y1+2]) * ((double)img[basexy+2]-(double)img[basex_1y1+2]);
                upright(x,y) = gamma / 1.414 * exp(-beta * temp);
            }else{
                upright(x,y) = 0;
            }
        }
    }


    void calcNWeightsCaller(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma){
        int rows = img.rows;
        int cols = img.cols;
        GpuMat gleftW, gupleftW, gupW, guprightW;
        leftW.create(rows, cols, CV_64FC1);
        upleftW.create(rows, cols, CV_64FC1);
        upW.create(rows, cols, CV_64FC1);
        uprightW.create(rows, cols, CV_64FC1);
        gleftW.upload(leftW);
        gupleftW.upload(upleftW);
        gupW.upload(upW);
        guprightW.upload(uprightW);

        uchar * ptr;
        cudaMalloc((void **)&ptr, rows*cols*3*sizeof(uchar));
        cudaMemcpy(ptr, img.data, rows*cols*3*sizeof(uchar));
        dim3 DimBlock(32,32);
        dim3 DimGrid(static_cast<int>(std::ceil(img.size().height /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(img.size().width / 
                        static_cast<double>(DimBlock.y))));

        calcNWeightsKernel<<<DimGrid, DimBlock>>>(ptr, gleftW, gupleftW, gupW, guprightW, beta, gamma, rows, cols);
        cudaFree(ptr);
        gleftW.download(leftW);
        gupleftW.download(upleftW);
        gupW.download(upW);
        guprightW.download(uprightW);

    }
}
