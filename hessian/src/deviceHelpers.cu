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

texture<float, 2, cudaReadModeElementType> texRef;

__global__ void performDoubleImage(PtrStep<float> out,
        const int cols, const int rows)
{
    //const int cols = in.cols;
    //const int rows = in.rows;
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= cols) || (y >= rows))
    {
        return;
    }
    int x2 = x << 1;
    int y2 = y << 1;
    const float v11 = tex2D(texRef, x, y);
    const float v12 = tex2D(texRef, x + 1, y);
    const float v21 = tex2D(texRef, x, y + 1);
    const float v22 = tex2D(texRef, x + 1, y + 1);

    if((x == cols - 1) && (y == rows - 1))
    {
        return;
    }
    else if (x == cols - 1)
    {
        out(y2,x2)   = v11;
        out(y2+1,x2) = 0.5f*(v11 + v21);
        return;
    }
    else if (y == rows - 1)
    {
        out(y2,x2)   = v11;
        out(y2,x2+1) = 0.5f*(v11 + v12);
        return;
    }
    else
    {
        out(y2,x2)     = v11;
        out(y2,x2+1)   = 0.5f *(v11+v12);
        out(y2+1,x2)   = 0.5f *(v11+v21);
        out(y2+1,x2+1) = 0.25f*(v11+v12+v21+v22);
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
    GpuMat n(input.rows*2, input.cols*2, CV_32FC1);
//    gpu::resize(input, n, n.size(), 2.0, 2.0, INTER_LINEAR);
    n.setTo(Scalar::all(0));
    dim3 blocks((15 + cols*2) / 16, (15 + rows*2) / 16);
    dim3 threads(16, 16);
    TextureBinder tb((PtrStepSz<float>)input, texRef);
//    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//    cudaBindTexture2D(0, texRef, input.data, desc, input.cols, input.rows, input.step);
//    std::cout<<n.rows<<' '<<n.cols<<std::endl;
//    performDoubleImage<<<blocks, threads>>>((float*)n.data, n.step, input.cols, input.rows);
    performDoubleImage<<<blocks, threads>>>(n, input.cols, input.rows);
    performFinalDoubleImage<<<1, 1>>>(input, n);
//    cudaUnbindTexture(texRef);
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
