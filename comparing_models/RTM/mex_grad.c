#include "mex.h"
#include <math.h>
void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
 int i, d1,d2,t, t2, D, T, DD, iter;
 double val, expval, err, *thetaTemp, *c, *step, *gradTemp;
 double *theta, *dp, *f, *grad, *hansen,*A, *init, *Numiter, *Tol;


    /* Retrieve the input data and Asign the output data*/
    dp  = mxGetPr(prhs[0]);
    A   = mxGetPr(prhs[1]);
    init = mxGetPr(prhs[2]);
    step = mxGetPr(prhs[3]);
    Numiter = mxGetPr(prhs[4]);
    Tol = mxGetPr(prhs[5]);
    
    D = mxGetM(prhs[0]);  T = mxGetN(prhs[0]);

    if (nlhs >= 1){
        plhs[0] = mxCreateDoubleMatrix(T+1, 1, mxREAL);
        theta = mxGetPr(plhs[0]);
    }
    if (nlhs >= 2){
        plhs[1] = mxCreateDoubleMatrix(T+1, 1, mxREAL);
        grad = mxGetPr(plhs[1]);
    }
    if (nlhs >= 3){
        plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
        f = mxGetPr(plhs[2]);
    }
    
    
    /*for (d1=0; d1<D; d1++){
        printf("%f", dp[d1]);
        for (d2=0; d2<D; d2++){
            printf("d1=%d, d2=%d, d1*D+d2=%d, A[] = %f\n", d1, d2, d1*D+d2, *(A+d1*D+d2));
        }
    }*/
    
    c = (double *)calloc(T+1, sizeof(double));
    thetaTemp = (double *)calloc(T+1, sizeof(double));
    gradTemp  = (double *)calloc(T+1, sizeof(double));
    
    for (i=0; i<T+1; i++){
        theta[i] = init[i];
    }
    
    DD = D*(D-1)/2;
    
    c[T] = 1;
    
    for (iter = 1; iter <=Numiter[0]; iter++){
        for (i=0; i<T+1; i++){
          thetaTemp[i] = theta[i];
        }
        for (i=0; i<T+1; i++){
            grad[i] = 0;
        }
        
        for (d1=0; d1<D-1; d1++){
            for (d2=d1+1; d2<D; d2++){
                val = theta[T];
                for (t=0; t<T; t++){
                   c[t] = dp[t*D + d1]*dp[t*D + d2];
                   val += theta[t]*c[t];
                }
                expval = exp(val);
                *f += A[d1*D+d2]*val + (1-A[d1*D+d2])*log(1-expval);
  
                for (t=0; t<T+1; t++){
                   grad[t] += c[t]*(1 - (1-A[d1*D+d2])/(1-expval));
                }
            }
        }
        
        
        for (i=0; i<T+1; i++){
            theta[i] += step[0]*grad[i];
        }
       
            
        err = 0;
        for (i=0; i<T+1; i++) {
            err += fabs(grad[i]);
        }
        
        if (err <= Tol[0]) 
            break;  
          

        /*printf("Iter=%d,grad =[ ", iter);
        for (i=0; i<T+1; i++){
            printf("%f  ", grad[i]);
        }
        printf("]\n-----theta=[");
        for (i=0; i<T+1; i++){
            printf("%f  ", theta[i]);
        }
        printf("]\n, T=%d, D=%d", T,D);
        */
    }
    printf("Iter=%d\n",iter);
              
}
          
                
