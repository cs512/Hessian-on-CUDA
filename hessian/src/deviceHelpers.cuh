/*
 * deviceHelpers.cuh
 *
 *  Created on: 2014-12-15
 *      Author: wangjz
 */

#ifndef DEVICEHELPERS_CUH_
#define DEVICEHELPERS_CUH_



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


#endif /* DEVICEHELPERS_CUH_ */
