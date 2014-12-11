/*
 * main.cpp
 *
 *  Created on: 2014-12-3
 *      Author: wangjz
 */

#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <ctime>

#include "hesaff/pyramid.h"
#include "hesaff/helpers.h"
#include "hesaff/affine.h"
#include "hesaff/siftdesc.h"

#include "pyramid.h"
#include "hesaff/helpers.h"
#include "deviceHelpers.h"
using namespace std;

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
using namespace cv;
using namespace cv::gpu;

const int cols = 1920;
const int rows = 1080;

bool matIsEqualToGpuMat(Mat &cpCur, GpuMat &cuMat)
{
    Mat cuCur;
    cuMat.download(cuCur);
    int count = 0;
    for (int eachRow = 0; eachRow < cpCur.rows; eachRow++)
    {
        for(int eachCol = 0; eachCol < cpCur.cols; eachCol++)
        {
            if(cuCur.at<float>(eachRow, eachCol) != cpCur.at<float>(eachRow, eachCol))
            {
                cout<<"CUDA::CPU"<<endl;
                cout<<cuCur.at<float>(eachRow, eachCol)<<"::"<<cpCur.at<float>(eachRow, eachCol);
                cout<<"\t@("<<eachRow<<","<<eachCol<<")"<<endl;
                ++count;
            }
        }
    }
    if(count == 0)
        return true;
    else
        return false;
}

void testOfHessianResponse(HessianDetector &cpDet, CUHessianDetector &cuDet, Mat &testInput)
{
    cout << "test of HessianDetector::hessianResponse" << endl;
    gpu::GpuMat cuTestInput;
    //float *data = testInput.ptr<float>(0);
    cuTestInput.upload(testInput);
    float curSigma = cpDet.par.initialSigma;
    clock_t cuStart = clock();
    gpu::GpuMat cuDeviceCur = cuDet.hessianResponse(cuTestInput, curSigma*curSigma);
    clock_t cuEnd = clock();
    Mat cpCur = cpDet.hessianResponse(testInput, curSigma*curSigma);
    clock_t cpEnd = clock();

    if(matIsEqualToGpuMat(cpCur, cuDeviceCur))
    {
        cout<<"test pass."<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
        cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    }
}

void testOfHalfImage(HessianDetector &cpDet, CUHessianDetector &cuDet, Mat &testInput)
{
    cout << "test of helper.cpp::halfImage" << endl;
    gpu::GpuMat cuTestInput;
    //float *data = testInput.ptr<float>(0);
    cuTestInput.upload(testInput);

    clock_t cuStart = clock();
    gpu::GpuMat cuDeviceCur = cuHalfImage(cuTestInput);
    clock_t cuEnd = clock();
    Mat cpCur = halfImage(testInput);
    clock_t cpEnd = clock();

    if(matIsEqualToGpuMat(cpCur, cuDeviceCur))
    {
        cout<<"test pass."<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
        cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    }
}

void testOfDoubleImage(HessianDetector &cpDet, CUHessianDetector &cuDet, Mat &testInput)
{
    cout << "test of helper.cpp::doubleImage" << endl;
    gpu::GpuMat cuTestInput;
    //float *data = testInput.ptr<float>(0);
    cuTestInput.upload(testInput);

    clock_t cuStart = clock();
    gpu::GpuMat cuDeviceCur = cuDoubleImage(cuTestInput);
    clock_t cuEnd = clock();
    Mat cpCur = doubleImage(testInput);
    clock_t cpEnd = clock();

    if(matIsEqualToGpuMat(cpCur, cuDeviceCur))
    {
        cout<<"test pass."<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
        cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    }
}

void testOfDetectOctaveKeypoints(HessianDetector &cpDet, CUHessianDetector &cuDet, Mat &testInput)
{
    cout << "test of CUHessianDetector::detectPyramidKeypoints" << endl;
    gpu::GpuMat cuTestInput(testInput);
    //float *data = testInput.ptr<float>(0);

    clock_t cuStart = clock();
    //gpu::GpuMat cuDeviceCur =
    cuDet.detectPyramidKeypoints(cuTestInput);
    clock_t cuEnd = clock();
    //Mat cpCur = doubleImage(testInput);
    cpDet.detectPyramidKeypoints(testInput);
    clock_t cpEnd = clock();

    //if(matIsEqualToGpuMat(cpCur, cuDeviceCur))
    //{
    cout<<"test pass."<<endl;
    cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    //}
}

int main(int argc, char **argv)
{

    //gpu::setDevice(0);
    PyramidParams par;
    CUPyramidParams cuPar;
    CUHessianDetector cuDet(cuPar);
    HessianDetector cpDet(par);
    //srand( (unsigned)time( NULL ) );
    Mat testInput(rows, cols, CV_32F, Scalar(0));
    srand( (unsigned)time( NULL ) );
    for (int eachRow = 0; eachRow < rows; eachRow++)
    {
        for(int eachCol = 0; eachCol < cols; eachCol++)
        {
            testInput.at<float>(eachRow, eachCol) = float(rand() % 256);
        }
    }

    testOfHessianResponse(cpDet, cuDet, testInput);
    testOfHalfImage(cpDet, cuDet, testInput);
    testOfDoubleImage(cpDet, cuDet, testInput);
    testOfDetectOctaveKeypoints(cpDet, cuDet, testInput);
	return 0;
}


