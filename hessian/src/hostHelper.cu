/*
 * hostHealper.cu
 *
 *  Created on: 2014-12-8
 *      Author: wangjz
 */


// TODO CUDA
Mat doubleImage(const Mat &input)
{
   Mat n(input.rows*2, input.cols*2, input.type());
   const float *in = input.ptr<float>(0);

   for (int r = 0; r < input.rows-1; r++)
      for (int c = 0; c < input.cols-1; c++)
      {
         const int r2 = r << 1;
         const int c2 = c << 1;
         n.at<float>(r2,c2)     = in[0];
         n.at<float>(r2+1,c2)   = 0.5f *(in[0]+in[input.step]);
         n.at<float>(r2,c2+1)   = 0.5f *(in[0]+in[1]);
         n.at<float>(r2+1,c2+1) = 0.25f*(in[0]+in[1]+in[input.step]+in[input.step+1]);
         ++in;
      }
   for (int r = 0; r < input.rows-1; r++)
   {
      const int r2 = r << 1;
      const int c2 = (input.cols-1) << 1;
      n.at<float>(r2,c2)   = input.at<float>(r,input.cols-1);
      n.at<float>(r2+1,c2) = 0.5f*(input.at<float>(r,input.cols-1) + input.at<float>(r+1,input.cols-1));
   }
   for (int c = 0; c < input.cols - 1; c++)
   {
      const int r2 = (input.rows-1) << 1;
      const int c2 = c << 1;
      n.at<float>(r2,c2)   = input.at<float>(input.rows-1,c);
      n.at<float>(r2,c2+1) = 0.5f*(input.at<float>(input.rows-1,c) + input.at<float>(input.rows-1,c+1));
   }
   n.at<float>(n.rows-1, n.cols-1) = n.at<float>(input.rows-1, input.cols-1);
   return n;
}
// TODO CUDA
Mat halfImage(const Mat &input)
{
   Mat n(input.rows/2, input.cols/2, input.type());
   float *out = n.ptr<float>(0);
   for (int r = 0, ri = 0; r < n.rows; r++, ri += 2)
      for (int c = 0, ci = 0; c < n.cols; c++, ci += 2)
         *out++ = input.at<float>(ri,ci);
   return n;
}


