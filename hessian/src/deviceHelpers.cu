/*
 * deviceHealper.cu
 *
 *  Created on: 2014-12-8
 *      Author: wangjz
 */

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
//#include <opencv2/gpu/>
#include "texture_binder.hpp"
#include <iostream>
#include "deviceHelpers.h"
using namespace cv;
using namespace cv::gpu;

__global__ void performDoubleImage(const PtrStepSz<float> in, PtrStep<float> out)
{
    const int cols = in.cols;
    const int rows = in.rows;

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= cols)
        return;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= rows)
        return;
    const int x2 = x * 2;
    const int y2 = y * 2;
    if((x == cols - 1) && (y == rows - 1))
    {
        return;
    }
    else if (x == cols - 1)
    {
        out(y2,x2)   = in(y,x);
        out(y2+1,x2) = 0.5f*(in(y,x) + in(y+1,x));
        return;
    }
    else if (y == rows - 1)
    {
        out(y2,x2)   = in(y, x);
        out(y2,x2+1) = 0.5f*(in(y, x) + in(y, x+1));
        return;
    }
    else
    {
        out(y2,x2)     = in(y, x);
        out(y2,x2+1)   = 0.5f *(in(y, x)+in(y, x+1));
        out(y2+1,x2)   = 0.5f *(in(y, x)+in(y+1, x));
        out(y2+1,x2+1) = 0.25f*(in(y, x)+in(y, x+1)+in(y+1, x)+in(y+1, x+1));
        return;
    }
}

__global__ void performFinalDoubleImage(const PtrStepSz<float> in, PtrStepSz<float> out)
{
    out(out.rows-1, out.cols-1) = out(in.rows-1, in.cols-1);
}

// TODO CUDA
GpuMat cuDoubleImage(const GpuMat &input)
{
    const int rows = input.rows;
    const int cols = input.cols;
    GpuMat n(input.rows*2, input.cols*2, input.type());
    n.setTo(Scalar::all(0));
    dim3 blocks((31 + cols) / 32, (31 + rows) / 32);
    dim3 threads(32, 32);
    texture<float, 2, cudaReadModeElementType> texRef;
    TextureBinder tb((PtrStepSz<float>)input, texRef);
    performDoubleImage<<<blocks, threads>>>(texRef, n);
    performFinalDoubleImage<<<1, 1>>>(input, n);
//    gpu::resize(input, n, n.size(), 2.0, 2.0, INTER_LINEAR);
    return n;
}

__global__ void performHalfImage(const PtrStep<float> in, PtrStep<float> out,
        int cols, int rows)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= cols)
        return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= rows)
        return;
    /* normalize and write out */
    out(y, x) = in(y * 2, x * 2);
    return;
}

// TODO CUDA
GpuMat cuHalfImage(const GpuMat &input)
{
    const int rows = input.rows / 2;
    const int cols = input.cols / 2;
    // TODO handle error
    dim3 blocks((15 + cols) / 16, (15 + rows) / 16);
    dim3 threads(16, 16);
    GpuMat n(input.rows/2, input.cols/2, input.type());
    performHalfImage<<<blocks, threads>>>(input, n, cols, rows);
    return n;
}
