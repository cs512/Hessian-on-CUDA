/*
 * deviceHealper.cu
 *
 *  Created on: 2014-12-8
 *      Author: wangjz
 */

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/utility.hpp>
//#include <opencv2/gpu/>
#include "texture_binder.hpp"
#include <iostream>
#include "deviceHelpers.h"
//#include "deviceHelpers.cuh"
using namespace cv;
using namespace cv::gpu;

static texture<float, 2, cudaReadModeElementType> texRef;

template <typename ValueType> __device__
void swap(ValueType *a, ValueType *b)
{
   ValueType tmp = *a; *a = *b; *b = tmp;
}

__device__ void cuSolveLinear3x3(float *A, float *b)
{
   // find pivot of first column
   int i = 0;
   float *pr = A;
   float vp = abs(A[0]);
   float tmp = abs(A[3]);
   if (tmp > vp)
   {
      // pivot is in 1st row
      pr = A+3;
      i = 1;
      vp = tmp;
   }
   if (abs(A[6]) > vp)
   {
      // pivot is in 2nd row
      pr = A+6;
      i = 2;
   }

   // swap pivot row with first row
   if (pr != A) { swap(pr, A); swap(pr+1, A+1); swap(pr+2, A+2); swap(b+i, b); }

   // fixup elements 3,4,5,b[1]
   vp = A[3] / A[0]; A[4] -= vp*A[1]; A[5] -= vp*A[2]; b[1] -= vp*b[0];

   // fixup elements 6,7,8,b[2]]
   vp = A[6] / A[0]; A[7] -= vp*A[1]; A[8] -= vp*A[2]; b[2] -= vp*b[0];

   // find pivot in second column
   if (abs(A[4]) < abs(A[7])) { swap(A+7, A+4); swap(A+8, A+5); swap(b+2, b+1); }

   // fixup elements 7,8,b[2]
   vp = A[7] / A[4];
   A[8] -= vp*A[5];
   b[2] -= vp*b[1];

   // solve b by back-substitution
   b[2] = (b[2]                    )/A[8];
   b[1] = (b[1]-A[5]*b[2]          )/A[4];
   b[0] = (b[0]-A[2]*b[2]-A[1]*b[1])/A[0];
}

__global__ void performFinalDoubleImage(const PtrStepSz<float> in, PtrStepSz<float> out)
{
    out(out.rows-1, out.cols-1) = out(in.rows-1, in.cols-1);
}
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


__global__ void performInterpolate(const PtrStepSz<float> im, const float ofsx, const float ofsy, const float a11,
        const float a12, const float a21, const float a22, PtrStepSz<float> res, int *count)
{
    int xId = threadIdx.x + blockIdx.x * blockDim.x;
    if (xId >= res.cols)
    {
        __syncthreads();
        return;
    }
    int i = xId - (res.cols >> 1);
    int yId = threadIdx.y + blockIdx.y * blockDim.y;
    if (yId >= res.rows)
    {
        __syncthreads();
        return;
    }
    int j = yId - (res.rows >> 1);
    if(yId + xId == 0)
    {
        *count = 0;
        __syncthreads();
    }
    else
    {
        __syncthreads();
    }
    const int width = im.cols-1;
    const int height = im.rows-1;
    const float rx = ofsx + j * a12;
    const float ry = ofsy + j * a22;
    float wx = rx + i * a11;
    float wy = ry + i * a21;
    const int x = (int) floor(wx);
    const int y = (int) floor(wy);
    if (x >= 0 && y >= 0 && x < width && y < height)
    {
        // compute weights
        wx -= x; wy -= y;
        // bilinear interpolation
        res(yId, xId) =
            (1.0f - wy) * ((1.0f - wx) * im(y,x)   + wx * im(y,x+1)) +
            (       wy) * ((1.0f - wx) * im(y+1,x) + wx * im(y+1,x+1));
    }
    else
    {
        res(yId, xId) = 0;
//        ret = true;
        *count = 1;
    }
}
//TODO cuda
bool cuInterpolate(const GpuMat &im, const float ofsx, const float ofsy, const float a11,
        const float a12, const float a21, const float a22, GpuMat &res)
{
    // input size (-1 for the safe bilinear interpolation)
    dim3 blocks((15 + res.cols) / 16, (15 + res.rows) / 16);
    dim3 threads(16, 16);
    int *devPtr = NULL;
    cudaMalloc((void **)&devPtr, sizeof(int));
    performInterpolate<<<blocks, threads>>>(im, ofsx, ofsy, a11, a12, a21, a22, res, devPtr);
    int count;
    cudaMemcpy(&count, devPtr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devPtr);
    if(count != 0)
        return false;
    else
        return true;
}
