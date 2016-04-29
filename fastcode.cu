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
    // cuda implementation of threshold BG/FG determine
    __global__ void thresholdKernel(const PtrStepSz<uchar> img1, const PtrStepSz<uchar> img2, const PtrStepSz<uchar> img3, PtrStepSz<uchar> maskFG, PtrStepSz<uchar> maskBG){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < img1.rows && y < img1.cols){
            if(img1(x,y)>30||img2(x,y)>30||img3(x,y)>30) maskFG(x,y) = 1;
            if(img1(x,y)<10&&img2(x,y)<10&&img3(x,y)<10) maskBG(x,y) = 1;
        }
    }

    void thresholdCaller(const Mat & img1, const Mat & img2, const Mat & img3, Mat & maskFG, Mat & maskBG){
        GpuMat gimg1, gimg2, gimg3, gmaskFG, gmaskBG;
        gimg1.upload(img1);
        gimg2.upload(img2);
        gimg3.upload(img3);
        gmaskFG.upload(maskFG);
        gmaskBG.upload(maskBG);

        dim3 DimBlock(32,32);
        dim3 DimGrid(static_cast<int>(std::ceil(img1.size().height /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(img1.size().width / 
                        static_cast<double>(DimBlock.y))));
        thresholdKernel<<<DimGrid, DimBlock>>>(gimg1, gimg2, gimg3, gmaskFG, gmaskBG);
        gmaskFG.download(maskFG);
        gmaskBG.download(maskBG);
    }

    // allocate and free GMM space
    void copyGMMtoGPU(double * src, double * & dst, size_t len){
        cudaMalloc((void**)&dst, sizeof(double)*len);
        cudaMemcpy(dst, src, sizeof(double)*len, cudaMemcpyHostToDevice);
    }

    void freeGMMonGPU(double * p){
        cudaFree(p);
    }

    // GMM in parallel
    __device__ double prob(double x, double y, double z, double * ptrGMM, int ci){
        double res = 0;
        double * coef = ptrGMM;
        double * mean = ptrGMM+5;
        double * invcov = mean + 15;
        double * covD = invcov + 45;
        if(coef[ci] > 0){
            mean += 3*ci;
            invcov += 9*ci;
            x -= mean[0];
            y -= mean[1];
            z -= mean[2];
            double mult = x*(x*invcov[0] + y*invcov[3]+ z*invcov[6]) +
                        y*(x*invcov[1] + y*invcov[4] + z*invcov[7]) + 
                        z*(x*invcov[2] + y*invcov[5] + z*invcov[8]);
            res = 1.0/sqrt(covD[ci]) * exp(-0.5*mult);
        }
        return res;
    }
    __global__ void GMMKernel(const PtrStepSz<uchar> img1, const PtrStepSz<uchar> img2, const PtrStepSz<uchar> img3, const PtrStepSz<uchar> mask, double * GMMonGPU,
                            PtrStepSzf fromSource, PtrStepSzf toSink, double lambda){
        
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if(x < img1.rows && y < img1.cols){
            if(mask(x,y) == GC_PR_BGD || mask(x,y) == GC_PR_FGD){
                int a = img1(x,y), b = img2(x,y), c = img3(x,y);
                double temp = 0.0;
                for(int i = 0; i < 5; i++){
                    temp += GMMonGPU[i] * prob(a,b,c,GMMonGPU, i);
                }
                fromSource(x,y) = -log(temp);
                temp = 0.0;
                GMMonGPU += 70;
                for(int i = 0; i < 5; i++){
                    temp += GMMonGPU[i] * prob(a,b,c,GMMonGPU, i);
                }
                toSink(x,y) = -log(temp);

            }else if(mask(x,y) == GC_BGD){
                fromSource(x,y) = 0;
                toSink(x,y) = lambda;
            }else{
                fromSource(x,y) = lambda;
                toSink(x,y) = 0;
            }
        }
    }

    void GMMCaller(const Mat& img1, const Mat& img2, const Mat& img3, const Mat& mask, double * GMMonGPU, Mat& fromSource, Mat& toSink, double lambda){
        GpuMat gimg1, gimg2, gimg3, gmask, gfromSource, gtoSink;
        gimg1.upload(img1);
        gimg2.upload(img2);
        gimg3.upload(img3);
        gmask.upload(mask);
        fromSource.create(img1.size(), CV_32FC1);
        toSink.create(img1.size(), CV_32FC1);
        gfromSource.upload(fromSource);
        gtoSink.upload(toSink);
        dim3 DimBlock(32,32);
        dim3 DimGrid(static_cast<int>(std::ceil(img1.size().height /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(img1.size().width / 
                        static_cast<double>(DimBlock.y))));
        GMMKernel<<<DimGrid, DimBlock>>>(gimg1, gimg2, gimg3, gmask, GMMonGPU, gfromSource, gtoSink, lambda);
        gfromSource.download(fromSource);
        gtoSink.download(toSink);
    }

}
