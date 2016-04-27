#include "fastcode.h"

// reference: http://stackoverflow.com/questions/24613637/custom-kernel-gpumat-with-float

using namespace cv;
using namespace std;

namespace fastcode{
    __global__ void maskShowKernel(const cuda::PtrStepSz<uchar> mask, cuda::PtrStepSz<uchar> mask4show){
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

    void maskShowCaller(Mat & mask, Mat & mask4show, cuda::Stream stream){
        const cuda::GpuMat gmask = mask.getGpuMat();
        mask4show.create(mask.size(), CV_8UC1);
        cuda::GpuMat gmask4show = mask4show.getGpuMat();

        dim3 DimBlock(16,16);
        dim3 DimGrid(static_cast<int>(std::ceil(mask.size().width /
                        static_cast<double>(DimBlock.x))), 
                        static_cast<int>(std::ceil(mask.size().height / 
                        static_cast<double>(DimBlock.y))));
        cudaStream_t localstream = cuda::StreamAccessor::getStream(stream);
        maskShowKernel<<<DimGrid, DimBlock, 0, localstream>>>(gmask, gmask4show);
        cudaSafeCall(cudaGetLastError());
    }
}
