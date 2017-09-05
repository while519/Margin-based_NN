#include "mex.h"
#include <stdlib.h>
#include <math.h>


/*
 * MEX file for rtmCGS.m.  For questions, please ask Arthur.
 * 
 * OUTPUTS:
 * plhs[0] = *z
 * plhs[1] = **wp
 * plhs[2] = **dp
 * 
 * INPUTS:
 * prhs[0] = *z
 * prhs[1] = **wp
 * prhs[2] = **dp
 * prhs[3] = *ztot
 * prhs[4] = *w
 * prhs[5] = *d
 * prhs[6] = alpha
 * prhs[7] = beta
 * prhs[8] = Nd
 * prhs[9] = A
 * prhs[10] = eta
 * prhs[11] = nu
 * prhs[12] = exactFlag
 * prhs[13] = randomPercentage
 * prhs[14] = negativePercentage
 *
 */

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
  int N;
  int W;
  int D;
  int T;

  double alpha;
  double beta;
  double wbeta;
  double nu;
  double exactFlag;
  double randomPercentage;  /* percentage of subsampled non-edges */
  double negativePercentage;  /* weight given to non-edge term in sampler */

  double *w;
  double *d;
  double *z;
  double *wp;
  double *dp;
  double *ztot;
  double *Nd;
  double *A;
  double *eta;

  int i, t, tt, wi, di, dd, ddd, prevdi, negcount, calccount, numDocs;
  double totprob, maxprob, currprob, pos, neg;
  double *probs;
  double *positive;
  double *negative;
  double *realNegative;
  double *randomDocs;
  double *randomNums;

  double *z_;
  double *wp_;
  double *dp_;
  
  N = mxGetM(prhs[0]) * mxGetN(prhs[0]);
  W = mxGetM(prhs[1]);
  T = mxGetN(prhs[1]);
  D = mxGetM(prhs[2]);

  z    = mxGetPr(prhs[0]);
  wp   = mxGetPr(prhs[1]);
  dp   = mxGetPr(prhs[2]);
  ztot = mxGetPr(prhs[3]);
  w    = mxGetPr(prhs[4]);
  d    = mxGetPr(prhs[5]);
  alpha    = mxGetScalar(prhs[6]);
  beta    = mxGetScalar(prhs[7]);
  Nd    = mxGetPr(prhs[8]);
  A    = mxGetPr(prhs[9]);
  eta    = mxGetPr(prhs[10]);
  nu    = mxGetScalar(prhs[11]);
  exactFlag    = mxGetScalar(prhs[12]);
  randomPercentage    = mxGetScalar(prhs[13]);
  negativePercentage = mxGetScalar(prhs[14]);
  
  probs = mxMalloc(T * sizeof(double));
  positive = mxMalloc(T * sizeof(double));
  negative = mxMalloc(T * sizeof(double));
  realNegative = mxMalloc(T * sizeof(double));
  randomDocs = mxMalloc(D * sizeof(double));
  randomNums = mxMalloc(D * sizeof(double));
  wbeta = W*beta;
  
  plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(W,T,mxREAL);
  plhs[2] = mxCreateDoubleMatrix(D,T,mxREAL);
  
  z_  = mxGetPr(plhs[0]);
  wp_ = mxGetPr(plhs[1]);
  dp_ = mxGetPr(plhs[2]);
  
  /* copy input arrays to output arrays */
  for (i = 0; i < N; i++) {
    w[i]--;
    d[i]--;
    z_[i] = z[i]-1;
  }
  for (i = 0; i < W*T; i++) wp_[i] = wp[i];
  for (i = 0; i < D*T; i++) dp_[i] = dp[i];

 
  prevdi = -1;

  /******************************************************/
  for (i = 0; i < N; i++) {

    wi = (int)(w[i]);
    di = (int)(d[i]);
    
    t = z_[i];
    ztot[t]--;     
    wp_[t*W + wi]--;
    dp_[t*D + di]--;

   /* exactFlag==3 means that we are skipping the link information altogether and just running LDA */

   if (exactFlag != 3)
   {
    
    if (di != prevdi) /* Most times we only wish to compute the link info once per document per sweep */
    {

      /* Statistics used for weighting */
      negcount = 0;
      calccount = 0;

      /* This section subsamples random non-edge links */
      for (dd=0; dd < D; dd++)
      {
        if (dd==di) continue;

        if (A[dd*D + di] == 0)
        {
          randomNums[dd] = drand48();
          if (randomNums[dd] < randomPercentage)
          {
            randomDocs[calccount] = dd;
            calccount = calccount+1;
            
          }
          negcount = negcount + 1;
        }
      }

      /* Here we compute the terms necessary for our approximations */
      for (t=0; t<T; t++) 
      {
        positive[t] = 0;
        negative[t] = 0;
        realNegative[t] = 0;

        for (dd=0; dd < D; dd++)
        {

          if (dd==di) continue;

          if (A[dd*D + di] > 0)
          {

            /* Link (y = 1) */
            positive[t] = positive[t] + dp_[t*D + dd] / Nd[dd];

          }
          else if (exactFlag == 0)
          {

            /* Non-Link (y = 0) */
            negative[t] = negative[t] + dp_[t*D + dd] / Nd[dd];

          }
          else if (exactFlag == 2)
          {

	    negative[t] = 0;
            if (randomNums[dd] < randomPercentage)
            {
              for(tt = 0; tt < T; tt++)
              {

                if (t == tt)
                {
                  negative[t] = negative[t] + eta[tt] * dp_[tt*D + dd] * (dp_[tt*D + di] + 1);
                }
                else
                {
                  negative[t] = negative[t] + eta[tt] * dp_[tt*D + dd] * dp_[tt*D + di];
                }

              }

              realNegative[t] = realNegative[t] + (log(1 - exp(negative[t] / (Nd[di] * Nd[dd]) + nu)));
            }


          }

        }

        positive[t] = positive[t] * eta[t] / Nd[di];
 
        if (exactFlag == 0) {  
	  negative[t] = negative[t] * eta[t]; 
        }
        else if (exactFlag == 2) 
	{
          /** Need to weight appropriately **/
          if (calccount > 0) {
            realNegative[t] = negcount * realNegative[t] / calccount;
          }
        }


        prevdi = di;
      }


      if (exactFlag == 0)
      {
        for (t = 0; t < T; t++)
        {
        
          realNegative[t] = 0;
          for(tt = 0; tt < T; tt++)
          {

            if (t == tt)
            {
              realNegative[t] = realNegative[t] + negative[tt] * (dp_[tt*D + di] + 1);
            }
            else
            {
              realNegative[t] = realNegative[t] + negative[tt] * dp_[tt*D + di];
            }

          }

          realNegative[t] = realNegative[t] / (Nd[di] * negcount) + nu;
          realNegative[t] = negcount * (log(1 - exp(realNegative[t])));
	}
      
      }


    }
   }

    /** -----------------------------------------------------------------  **/



    totprob = 0;

    for (t = 0; t < T; t++) {


      if (exactFlag == 1)  /* This is the exact case */
      {

        realNegative[t] = 0;
        for (ddd=0; ddd < calccount; ddd++)
        {
          dd = randomDocs[ddd];

          negative[t] = 0;
          for(tt = 0; tt < T; tt++)
          {

            if (t == tt)
            {
              negative[t] = negative[t] + eta[tt] * dp_[tt*D + dd] * (dp_[tt*D + di] + 1);
            }
            else
            {
              negative[t] = negative[t] + eta[tt] * dp_[tt*D + dd] * dp_[tt*D + di];
            }

          }

          realNegative[t] = realNegative[t] + (log(1 - exp(negative[t] / (Nd[di] * Nd[dd]) + nu)));

        }
        
        /** Need to weight appropriately **/
        realNegative[t] = negcount * realNegative[t] / calccount;
      }


      /** probs[t] = (wp_[t*W + wi] + beta) * (dp_[t*D + di] + alpha) / (ztot[t] + wbeta);  **/
      /** probs[t] = exp(positive[t] + realNegative[t]) * (wp_[t*W + wi] + beta) * (dp_[t*D + di] + alpha) / (ztot[t] + wbeta); **/ 
    
      if (exactFlag != 3)
      {
        probs[t] = exp(positive[t] + negativePercentage * realNegative[t]) * (wp_[t*W + wi] + beta) * (dp_[t*D + di] + alpha) / (ztot[t] + wbeta); 
      }
      else
      {
        probs[t] = (wp_[t*W + wi] + beta) * (dp_[t*D + di] + alpha) / (ztot[t] + wbeta); 
      }

      totprob += probs[t];
    }
   

    /* After computing the conditional distribution, we can now sample z */
 
    maxprob  = totprob * drand48();
    currprob = probs[0];
    t = 0;
    while (maxprob > currprob) {
      t++;
      currprob += probs[t];
    }

    z_[i] = t;
    /* Now update sufficient statistics */   
    ztot[t]++;     
    wp_[t*W + wi]++;
    dp_[t*D + di]++;

  }

  for (i = 0; i < N; i++) {
    z_[i]++;
    w[i]++;
    d[i]++;
  }

  mxFree(probs);

}
