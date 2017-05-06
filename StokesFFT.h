#pragma once


#include <cmath>

#include <complex>

//
#include <mkl_dfti.h>


// complex number
typedef std::complex<double> complex_t;

// 0 + 1i
static const complex_t ImUnit(0.0, 1.0);




//
// 2D Stokes
//
struct StokesFFT2D
{
	double xlo, xhi;
	double ylo, yhi;
	int nx, ny;
	double dx, dy;

	double rho, mu;

	// velocity & pressure
	double *u, *v, *p;

	// source force
	double *f, *g;

	// wave number
	int *kx, *ky;

	// 
	complex_t *uhat, *vhat, *phat;
	complex_t *fhat, *ghat;


};


//
// 3D Stokes
//
struct StokesFFT
{
	//
	double dens, visc;

	//
	double problo[3];
	double probhi[3];
	double problen[3];

	//
	int cellNum[3];
	double cellSize[3];

	//
	// data buffer
	//

	// discrete wave number
	int *ki, *kj, *kk;
	// real wave number scaled by (2*pi/L)
	double *kx, *ky, *kz;

	// velocity
	double *u, *v, *w;
	// pressure
	double *p;

	// source
	double *fx, *fy, *fz;

	// reciprocal space
	complex_t *uhat, *vhat, *what;
	complex_t *phat;
	complex_t *fxhat, *fyhat, *fzhat;

	//
	// DFT
	//
	DFTI_DESCRIPTOR_HANDLE fft;

};

//
void stokes_init(StokesFFT &stokes);

//
void stokes_init_fft(StokesFFT &stokes);

//
int stokes_exec_fft(StokesFFT &stokes, double *u, complex_t *uhat);
int stokes_exec_ifft(StokesFFT &stokes, complex_t *uhat, double *u);


//
void stokes_solve(StokesFFT &stokes);

//
void stokes_output_vtk(StokesFFT &stokes, const char *filename);

//
void stokes_zero_buffer(StokesFFT &stokes, double *u);

//
void stokes_fill_periodic(StokesFFT &stokes, const double *u, double *uper);

////////////////////////////////////////////////////////////////////////////////
//
// FFT utility
//
////////////////////////////////////////////////////////////////////////////////

// MKL FFT requires row-major 
inline int fft_sub2ind(int nx, int ny, int nz, int i, int j, int k) {
	return i*ny*nz + j*nz + k;
}
inline void fft_ind2sub(int nx, int ny, int nz, int ind, int &i, int &j, int &k) {
	i = ind / (ny*nz);
	j = (ind/nz) % ny;
	k = ind % nz;
}

// Get wave number
void fft_wavenum(int n, int k[]);
void fft_wavenum(int n, double L, double k[]);

// Fill the conjugate-even array.
// This is performed for the last dimension.
// In-place operation, assume there is enough space.
void fft_conjeven_fill1d(int n, complex_t uhat[]);
void fft_conjeven_fill2d(int nx, int ny, complex_t uhat[]);
void fft_conjeven_fill3d(int nx, int ny, int nz, complex_t uhat[]);
//
void fft_conjeven_fill3d(complex_t uhat[], int nx, int ny, int nz, int ncx, int ncy, int ncz);

// Normalize IFFT result
void fft_normalize3d(double uback[], int nx, int ny, int nz, int nrx, int nry, int nrz);

//
int fft_r2c_init2d(DFTI_DESCRIPTOR_HANDLE &fft,
	int nx, int ny, int nxbuf, int nybuf);
int fft_r2c_init3d(DFTI_DESCRIPTOR_HANDLE &fft, int nx, int ny, int nz, 
	int nrx, int nry, int nrz, int ncx, int ncy, int ncz);
	

