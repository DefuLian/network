#if !defined(_WIN32)
#define dgemm dgemm_
#endif
#include "mex.h"
#include "string.h"
static const unsigned char bit_in_char[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
    2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
    4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
    5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
    5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    unsigned int n1 = mxGetM(prhs[0]);
    unsigned char* B1 = (unsigned char*)mxGetPr(prhs[0]);
    unsigned int n2 = mxGetM(prhs[1]);
    unsigned char* B2 = (unsigned char*)mxGetPr(prhs[1]);
    unsigned int nwords = mxGetN(prhs[1]);
    plhs[0] = mxCreateNumericMatrix(n1, n2, mxUINT16_CLASS, mxREAL);
	unsigned short *Dh = (unsigned short*)mxGetPr(plhs[0]);
    //mexPrintf("%d,%d,%d,%d\n",n1,n2,B1[0],B1[1]);
    for(int i=0;i<n1;++i)
    {
        for(int j=0;j<n2;++j)
        {
            Dh[i+j*n1] = 0;
            for(int n=0;n<nwords;++n)
            {   
                unsigned char y = B1[i+n*n1] ^ B2[j+n*n2];
                Dh[i+j*n1] += bit_in_char[y];
            }
            
        }
            
    }

}