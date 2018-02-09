#if !defined(_WIN32)
#define dgemm dgemm_
#endif
#include "mex.h"
#include "string.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    unsigned int n1 = mxGetM(prhs[0]);
    double* B1 = mxGetPr(prhs[0]);
    unsigned int n2 = mxGetM(prhs[1]);
    double* B2 = mxGetPr(prhs[1]);
    unsigned int nwords = mxGetN(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(n1, n2, mxREAL);
	double *Dh = mxGetPr(plhs[0]);
    for(int i=0;i<n1;++i)
    {
        for(int j=0;j<n2;++j)
        {
            Dh[i+j*n1] = 0;
            for(int n=0;n<nwords;++n)
            {   
                Dh[i+j*n1] += B1[i+n*n1] * B2[j+n*n2];
            }
            
        }
            
    }

}