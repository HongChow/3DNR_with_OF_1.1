//
// Created by hong on 22-5-31.
//

#include "svd_decomposition.h"
float withSignOf(float a, float b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

float svdhypot(float a, float b)
{
    a = fabsf(a);
    b = fabsf(b);
    if(a > b) {
        b /= a;
        return a*sqrt(1.0 + b*b);
    } else if(b) {
        a /= b;
        return b*sqrt(1.0 + a*a);
    }
    return 0.0;
}
void svdrotate_double(float& a, float& b, float c, float s)
{
    float d = a;
    a = +d*c +b*s;
    b = -d*s +b*c;
}


void compute_pca_svd(
        float **  A,
        float ** U,
        float ** V,
        float * W,
        int rows,
        int cols
){
    const float	EPSILON = 0.00001;
    const int SVD_MAX_ITS = 100;
    double g, scale, anorm;
    float * RV1 = new float[cols];
    // ----- U init ----- //
    for (int i=0; i < rows; i++)
        for(int j=0; j < cols; j++)
            U[i][j] = A[i][j];

    // Householder reduction to bidiagonal form:
    anorm = g = scale = 0.0;
    for (int i=0; i< cols; i++) {
        int l = i + 1;
        RV1[i] = scale*g;
        g = scale = 0.0;

        // ----------- V direction ----------- //
        if(i < rows) {
            for (int k=i; k< rows; k++)
                scale += fabsf(U[k][i]);// to debug , L1 范数
            if (scale != 0.0) {
                double invScale=1.0/scale, s=0.0;
                for (int k=i; k< rows; k++) {
                    U[k][i] *= invScale;
                    s += U[k][i] * U[k][i];
                }
                double f = U[i][i];
                g = - withSignOf(sqrt(s),f);
                double h = 1.0 / (f*g - s);
                U[i][i] = f - g;
                for (int j=l; j< cols; j++) {
                    s = 0.0;
                    for (int k=i; k< rows; k++)
                        s += U[k][i] * U[k][j];
                    f = s * h;
                    for (int k=i; k< rows; k++)
                        U[k][j] += f * U[k][i];
                }
                for (int k=i; k< rows; k++)
                    U[k][i] *= scale;
            }
        }

        W[i] = scale * g;
        g = scale = 0.0;
        if ( i< rows && i< cols-1 ) {
            for (int k=l; k< cols; k++)
                scale += fabsf(U[i][k]);
            if (scale != 0.0) {
                double invScale=1.0/scale, s=0.0;
                for (int k=l; k< cols; k++) {
                    U[i][k] *= invScale;
                    s += U[i][k] * U[i][k];
                }
                double f = U[i][l];
                g = - withSignOf(sqrt(s),f);
                double h = 1.0 / (f*g - s);
                U[i][l] = f - g;
                for (int k=l; k< cols; k++)
                    RV1[k] = U[i][k] * h;
                for (int j=l; j< rows; j++) {
                    s = 0.0;
                    for (int k=l; k< cols; k++)
                        s += U[j][k] * U[i][k];
                    for (int k=l; k< cols; k++)
                        U[j][k] += s * RV1[k];
                }
                for (int k=l; k< cols; k++)
                    U[i][k] *= scale;
            }
        }
        anorm = MAX(anorm, fabsf(W[i]) + fabsf(RV1[i]) );
    }

    // Accumulation of right-hand transformations:
    V[cols-1][cols-1] = 1.0;
    for (int i= cols-2; i>=0; i--) {
        V[i][i] = 1.0;
        int l = i+1;
        g = RV1[l];
        if (g != 0.0) {
            double invgUil = 1.0 / (U[i][l]*g);
            for (int j=l; j< cols; j++)
                V[j][i] = U[i][j] * invgUil;
            for (int j=l; j< cols; j++) {
                double s = 0.0;
                for (int k=l; k< cols; k++)
                    s += U[i][k] * V[k][j];
                for (int k=l; k< cols; k++)
                    V[k][j] += s * V[k][i];
            }
        }
        for (int j=l; j< cols; j++)
            V[i][j] = V[j][i] = 0.0;
    }

    // Accumulation of left-hand transformations:
    for (int i=MIN(rows,cols)-1; i>=0; i--) {
        int l = i+1;
        g = W[i];
        for (int j=l; j< cols; j++)
            U[i][j] = 0.0;
        if (g != 0.0) {
            g = 1.0 / g;
            double invUii = 1.0 / U[i][i];
            for (int j=l; j< cols; j++) {
                double s = 0.0;
                for (int k=l; k< rows; k++)
                    s += U[k][i] * U[k][j];
                double f = (s * invUii) * g;
                for (int k=i; k< rows; k++)
                    U[k][j] += f * U[k][i];
            }
            for (int j=i; j< rows; j++)
                U[j][i] *= g;
        } else
            for (int j=i; j< rows; j++)
                U[j][i] = 0.0;
        U[i][i] = U[i][i] + 1.0;
    }

    // Diagonalization of the bidiagonal form:
    for (int k=cols-1; k>=0; k--) { // Loop over singular values
        for (int its=1; its<=SVD_MAX_ITS; its++) {
            bool flag = false;
            int l  = k;
            int nm = k-1;
            while(l>0 && fabsf(RV1[l]) > EPSILON*anorm) { // Test for splitting
                if(fabsf(W[nm]) <= EPSILON*anorm) {
                    flag = true;
                    break;
                }
                l--;
                nm--;
            }
            if (flag) {	// Cancellation of RV1[l], if l > 0
                double c=0.0, s=1.0;
                for (int i=l; i< k+1; i++) {
                    double f = s * RV1[i];
                    RV1[i] = c * RV1[i];
                    if (fabsf(f)<=EPSILON*anorm)
                        break;
                    g = W[i];
                    double h = svdhypot(f,g);
                    W[i] = h;
                    h = 1.0 / h;
                    c = g * h;
                    s = - f * h;
                    for (int j=0; j< rows; j++)
                        svdrotate_double(U[j][nm],U[j][i], c,s);
                }
            }
            double z = W[k];
            if (l==k) {		// Convergence of the singular value
                if (z< 0.0) {	// Singular value is made nonnegative
                    W[k] = -z;
                    for (int j=0; j< cols; j++)
                        V[j][k] = - V[j][k];
                }
                break;
            }

            // Exception if convergence to the singular value not reached:
            if(its==SVD_MAX_ITS) {
                printf("svd::convergence_error\n");
                delete[] RV1;
                exit(-1);
            }
            double x = W[l]; // Get QR shift value from bottom 2x2 minor
            nm = k-1;
            double y = W[nm];
            g = RV1[nm];
            double h = RV1[k];
            double f = ( (y-z)*(y+z) + (g-h)*(g+h) ) / ( 2.0*h*y );
            g = svdhypot(f,1.0);
            f = ( (x-z)*(x+z) + h*(y/(f+withSignOf(g,f)) - h) ) / x;
            // Next QR transformation (through Givens reflections)
            double c=1.0, s=1.0;
            for (int j=l; j<=nm; j++) {
                int i = j+1;
                g = RV1[i];
                y = W[i];
                h = s * g;
                g = c * g;
                z = svdhypot(f,h);
                RV1[j] = z;
                z = 1.0 / z;
                c = f * z;
                s = h * z;
                f = x*c + g*s;
                g = g*c - x*s;
                h = y * s;
                y *= c;
                for(int jj=0; jj < cols; jj++)
                    svdrotate_double(V[jj][j],V[jj][i], c,s);
                z = svdhypot(f,h);
                W[j] = z;
                if (z!=0.0) { // Rotation can be arbitrary if z = 0.0
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = c*g + s*y;
                x = c*y - s*g;
                for(int jj=0; jj < rows; jj++)
                    svdrotate_double(U[jj][j],U[jj][i], c,s);
            }
            RV1[l] = 0.0;
            RV1[k] = f;
            W[k] = x;
        }
    }

    int n = rows;
    int p = cols;
    // ------------ post process ------------ //
    for(int i=0; i < n; i++)
        for(int j=0; j < p; j++)
            U[i][j] *= W[j];

    // Normalize eigenvalues
    float norm = (float) (n-1);
    for(int i=0; i < p; i++)
        W[i] = W[i] * W[i] / norm;

    // If n < p, principal component should be zero from n to p-1
    // Coefficients of these principal components should be zero
    if (n < p) {
        for(int i=n-1; i < p; i++) W[i] = 0.0f;

        for(int j=0; j < n; j++)
            for(int i=n-1; i < p; i++)
                U[j][i] = 0.0f;
    }
    delete[] RV1;
}

// --- initialization of A matrix to the 2D Double Array form --- //