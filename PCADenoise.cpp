//
// Created by hong on 22-6-1.
//
#include "PCADenoise.h"
float scalar_product(float *u, float *v, int n)
{

    double aux = 0.0f;
    double  aux_temp = 0.0f;
    for(int i=0; i < n; i++){
        if (u[i]==0 or v[i]==0)
            aux_temp = 0;
        else
            aux_temp =  u[i] * v[i];
        //aux += (double) u[i] * (double) v[i];
        aux += aux_temp;
    }


    return (float) aux;

}
void hard_pca_coefficients(
        float *S,
        float **F,
        float **U,
        float rsigma2,
        int rows,
        int cols
        )
{
    for(int jj=0; jj < cols; jj++)
        for(int ii=0; ii < rows; ii++) {
            if (S[jj] < rsigma2)
                F[ii][jj] = 0.0f;
            else
                F[ii][jj] = 1.0f;
            U[ii][jj]*=F[ii][jj];
        }
}
void PCADenoise(
        float ** input_3D_Blocks,
        float * mean_noisy_3D,
        float fSigma,
        float fRMult,
        int rows,
        int cols,
        float ** output_3D_Blocks
        ){
    float fSigma2 = fSigma * fSigma;
    float fRSigma2 = fRMult*fRMult*fSigma2*96;
    float ** U = new float *[rows]; // npts*p = npts*5*5*3
    float ** F = new float *[rows]; // npts*p = npts*5*5*3 --- Filter to which smaller V will be set zeros
    float ** V = new float *[cols]; // p*p
    float  * S = new float [cols];    // p or p*p  S----W or D or SIGMA in SVD
    for (int i=0; i<rows;i++) {
        U[i] = new float [cols];
        F[i] = new float [cols];
    }
    for (int i=0; i<cols;i++)
        V[i] = new float [cols];
    // ---- init ---- //
    for (int i=0; i<rows;i++)
        for (int j=0; j<cols;j++){
            U[i][j]  = 0.0;
            F[i][j]  = 0.0;
        }
    for (int i=0; i<cols;i++){
        S[i]=0.0;
        for (int j=0; j<cols;j++){
            V[i][j]  = 0.0;
        }
    }
    compute_pca_svd(input_3D_Blocks,U,V,S,rows,cols);
    hard_pca_coefficients(S,F,U,fRSigma2,rows,cols);
    /// Reconstruct denoised patches and save in output
    //for(int kk =0 ; kk<cols; kk++){
    //    std::cout<<'mean_noisy_3D = '<<(double)mean_noisy_3D[kk]<<std::endl;
    //}
    for(int ii=0; ii < rows; ii++) {
        for (int kk=0; kk < cols; kk++) {
            //float recovery_value = (float) scalar_product(U[ii], V[kk],cols);
            //double mean_noisy = mean_noisy_3D[kk];
            float out_temp = mean_noisy_3D[kk] + (float) scalar_product(U[ii], V[kk],cols);
            output_3D_Blocks[ii][kk] = CLIP(out_temp,0.0f,255.0f);
        }
    }
    for (int i=0; i<rows;i++) {
        delete [] U[i];
        delete [] F[i];
    }
    for (int i=0; i<cols;i++)
        delete [] V[i];
    delete[] U;
    delete[] F;
    delete[] V;
    delete[] S;

}