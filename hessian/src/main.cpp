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
#include "hostHelpers.h"
#include "affine.h"
using namespace std;

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
using namespace cv;
using namespace cv::gpu;

//const int cols = 1920;
//const int rows = 1080;

struct CUHessianAffineParams
{
   float threshold;
   int   max_iter;
   float desc_factor;
   int   patch_size;
   bool  verbose;
   CUHessianAffineParams()
      {
         threshold = 16.0f/3.0f;
         max_iter = 16;
         desc_factor = 3.0f*sqrt(3.0f);
         patch_size = 41;
         verbose = false;
      }
};

class CUAffineHessianDetector : public CUHessianDetector, CUAffineShape, CUHessianKeypointCallback, CUAffineShapeCallback
{
public:
    const Mat image;
    SIFTDescriptor sift;
    int g_numberOfPoints;
    int g_numberOfShapes;
public:
    CUAffineHessianDetector(const Mat &image, const CUPyramidParams &par, const CUAffineShapeParams &ap, const SIFTDescriptorParams &sp) :
       CUHessianDetector(par),
       CUAffineShape(ap),
       image(image),
       sift(sp)
    {
        this->setHessianKeypointCallback(this);
        this->setAffineShapeCallback(this);
        g_numberOfPoints = 0;
        g_numberOfShapes = 0;
    }

    void onHessianKeypointDetected(const GpuMat &blur, float x, float y, float s, float pixelDistance, int type, float response)
    {
        g_numberOfPoints++;
        findAffineShape(blur, x, y, s, pixelDistance, type, response);
    }
    void onAffineShapeFound(
         const GpuMat &blur, float x, float y, float s, float pixelDistance,
         float a11, float a12,
         float a21, float a22,
         int type, float response, int iters)
    {
        g_numberOfShapes++;
        return;
    }
};

bool matIsEqualToGpuMat(Mat &cpCur, GpuMat &cuMat)
{
    Mat cuCur;
    cuMat.download(cuCur);
    int count = 0;
    for (int eachRow = 0; eachRow < cpCur.rows; eachRow++)
    {
        for(int eachCol = 0; eachCol < cpCur.cols; eachCol++)
        {
            if((fabs((double)cuCur.at<float>(eachRow, eachCol) - (double)cpCur.at<float>(eachRow, eachCol)) >
               (fabs((double)cuCur.at<float>(eachRow, eachCol)) * 0.0005)) &&
               ((fabs((double)cuCur.at<float>(eachRow, eachCol) - (double)cpCur.at<float>(eachRow, eachCol)) > 0.0001)))
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

bool matIsEqualToGpuMat(Mat &cpCur, Mat &cuCur)
{
    int count = 0;
    for (int eachRow = 0; eachRow < cpCur.rows; eachRow++)
    {
        for(int eachCol = 0; eachCol < cpCur.cols; eachCol++)
        {
            if(cuCur.at<unsigned char>(eachRow, eachCol) != cpCur.at<unsigned char>(eachRow, eachCol))
            {
                cout<<"CUDA::CPU"<<endl;
                cout<<(int)cuCur.at<unsigned char>(eachRow, eachCol)<<"::"<<(int)cpCur.at<unsigned char>(eachRow, eachCol);
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

int countOfMat(Mat &cpCur)
{
    int count = 0;
    for (int eachRow = 0; eachRow < cpCur.rows; eachRow++)
    {
        for(int eachCol = 0; eachCol < cpCur.cols; eachCol++)
        {
            if(cpCur.at<unsigned char>(eachRow, eachCol) == 1)
            {
                ++count;
            }
        }
    }
    return count;
}


void testOfHessianResponse(HessianDetector &cpDet, CUHessianDetector &cuDet, Mat &testInput)
{
    cout << "test of HessianDetector::hessianResponse" << endl;
    gpu::GpuMat cuTestInput(testInput);
    CV_Assert(cuTestInput.type() == CV_32FC1);
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
#ifdef DEBUG_H_PK
    vector<Mat>::iterator itg = cuDet.results.begin();
    int count = 0;
    for(vector<Mat>::iterator itc = cpDet.results.begin(); itc != cpDet.results.end(); ++itc)
    {
        cout<<"cpu mat points count:"<<countOfMat(*itc)<<endl;
        cout<<"gpu mat points count:"<<countOfMat(*itg)<<endl;
        if(!matIsEqualToGpuMat(*itg, *itc))
        {
            cout<<"test failed @ level:"<<count<<endl;
        }
        else
        {
            cout<<"test success @ level:"<<count<<endl;
        }
        ++itg;
        ++count;
    }
#endif
    cout<<"test pass."<<endl;
    cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
    cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
}

void testOfGaussianBlur(HessianDetector &cpDet, CUHessianDetector &cuDet, Mat &testInput)
{
    cout << "test of helper.cpp::gaussianBlur" << endl;
    gpu::GpuMat cuTestInput;
    //float *data = testInput.ptr<float>(0);
    cuTestInput.upload(testInput);

    clock_t cuStart = clock();
    gpu::GpuMat cuDeviceCur = cuGaussianBlur(cuTestInput, 2.0);
    clock_t cuEnd = clock();
    Mat cpCur = gaussianBlur(testInput, 2.0);
    clock_t cpEnd = clock();

    if(matIsEqualToGpuMat(cpCur, cuDeviceCur))
    {
        cout<<"test pass."<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
        cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    }
}

void testOfInterpolate(Mat &testInput)
{
    cout << "test of helper.cpp::interpolate" << endl;
    gpu::GpuMat cuTestInput;
    //float *data = testInput.ptr<float>(0);
    cuTestInput.upload(testInput);
    gpu::GpuMat cuDeviceCur(19, 19, CV_32FC1);
    Mat cpCur(19, 19, CV_32FC1);
    clock_t cuStart = clock();
    cuInterpolate(cuTestInput, 7, 6.5, 15, 14, 13, 12, cuDeviceCur);
    clock_t cuEnd = clock();
    interpolate(testInput, 7, 6.5, 15, 14, 13, 12, cpCur);
    clock_t cpEnd = clock();

    if(matIsEqualToGpuMat(cpCur, cuDeviceCur))
    {
        cout<<"test pass."<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
        cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    }
}

void testOfComputeGrad(Mat &testInput)
{
    cout << "test of affine.cpp::computeGradient" << endl;
    gpu::GpuMat cuTestInput;
    //float *data = testInput.ptr<float>(0);
    cuTestInput.upload(testInput);
    gpu::GpuMat cuRes1(cuTestInput.rows, cuTestInput.cols, CV_32FC1);
    gpu::GpuMat cuRes2(cuTestInput.rows, cuTestInput.cols, CV_32FC1);

    Mat cpRes1(testInput.rows, testInput.cols, CV_32FC1);
    Mat cpRes2(testInput.rows, testInput.cols, CV_32FC1);

    clock_t cuStart = clock();
    cuComputeGradient(cuTestInput, cuRes1, cuRes2);
    clock_t cuEnd = clock();
    computeGradient(testInput, cpRes1, cpRes2);
    clock_t cpEnd = clock();

    if(matIsEqualToGpuMat(cpRes1, cuRes1) && matIsEqualToGpuMat(cpRes2, cuRes2))
    {
        cout<<"test pass."<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
        cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"s"<<endl;
    }
}

int main(int argc, char **argv)
{

    //gpu::setDevice(0);
    PyramidParams par;
    CUPyramidParams cuPar;
    CUHessianDetector cuDet(cuPar);
    HessianDetector cpDet(par);
    //srand( (unsigned)time( NULL ) );
//    Mat testInput(rows, cols, CV_32F, Scalar(0));

    Mat tmp = imread("/home/wangjz/testImage.jpg");
    Mat testInput(tmp.rows, tmp.cols, CV_32FC1, Scalar(0));

    float *out = testInput.ptr<float>(0);
    unsigned char *in  = tmp.ptr<unsigned char>(0);

    for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
    {
        *out = (float(in[0]) + in[1] + in[2])/3.0f;
        out++;
        in+=3;
    }

//    testOfHessianResponse(cpDet, cuDet, testInput);
//    testOfHalfImage(cpDet, cuDet, testInput);
//    testOfDoubleImage(cpDet, cuDet, testInput);
//    testOfGaussianBlur(cpDet, cuDet, testInput);
//    testOfDetectOctaveKeypoints(cpDet, cuDet, testInput);
//    testOfInterpolate(testInput);
//    testOfComputeGrad(testInput);

    {
        CUHessianAffineParams par;
        CUPyramidParams p;
        p.threshold = par.threshold;

        CUAffineShapeParams ap;
        ap.maxIterations = par.max_iter;
        ap.patchSize = par.patch_size;
        ap.mrSize = par.desc_factor;

        SIFTDescriptorParams sp;
        sp.patchSize = par.patch_size;

        CUAffineHessianDetector detector(testInput, p, ap, sp);
        clock_t cuStart = clock();
        detector.detectPyramidKeypoints(testInput);
        clock_t cuEnd = clock();
        cout<<"shapes from GPU: "<<detector.g_numberOfShapes<<endl;
        cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
    }

    return 0;
}


