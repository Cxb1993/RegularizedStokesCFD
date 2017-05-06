

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cassert>

#include <iostream>

//
#include "StokesFFT.h"

// MKL
#include <mkl_dfti.h>

//
#define STOKES_OMP_SCHEDULE schedule(static)



void stokes_init(StokesFFT &stokes) {
	
	// grid
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];
	const double xhi = stokes.probhi[0];
	const double yhi = stokes.probhi[1];
	const double zhi = stokes.probhi[2];
	const double xlen = xhi - xlo;
	const double ylen = yhi - ylo;
	const double zlen = zhi - zlo;

	stokes.problen[0] = xlen;
	stokes.problen[1] = ylen;
	stokes.problen[2] = zlen;

	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = xlen / nx;
	const double dy = ylen / ny;
	const double dz = zlen / nz;

	stokes.cellSize[0] = dx;
	stokes.cellSize[1] = dy;
	stokes.cellSize[2] = dz;


	// wave number
	// integer
	stokes.ki = new int[nx];
	stokes.kj = new int[ny];
	stokes.kk = new int[nz];
	fft_wavenum(nx, stokes.ki);
	fft_wavenum(ny, stokes.kj);
	fft_wavenum(nz, stokes.kk);
	// real, scaled
	stokes.kx = new double[nx];
	stokes.ky = new double[ny];
	stokes.kz = new double[nz];
	fft_wavenum(nx, xlen, stokes.kx);
	fft_wavenum(ny, ylen, stokes.ky);
	fft_wavenum(nz, zlen, stokes.kz);

	// data buffer
	//

	// real space
	//const int nreal = (nx+1) * (ny+1) * (nz+1);
	const int nreal = nx * ny * nz;
	stokes.u = new double[nreal];
	stokes.v = new double[nreal];
	stokes.w = new double[nreal];
	stokes.p = new double[nreal];
	stokes.fx = new double[nreal];
	stokes.fy = new double[nreal];
	stokes.fz = new double[nreal];

	// reciprocal spectral space
	const int nrecp = nx * ny * nz;
	stokes.uhat = new complex_t[nrecp];
	stokes.vhat = new complex_t[nrecp];
	stokes.what = new complex_t[nrecp];
	stokes.phat = new complex_t[nrecp];
	stokes.fxhat = new complex_t[nrecp];
	stokes.fyhat = new complex_t[nrecp];
	stokes.fzhat = new complex_t[nrecp];

	for (int i=0; i<nreal; i++) {
		stokes.u[i] = 0;
		stokes.v[i] = 0;
		stokes.w[i] = 0;
		stokes.p[i] = 0;
		stokes.fx[i] = 0;
		stokes.fy[i] = 0;
		stokes.fz[i] = 0;
	}

	for (int i=0; i<nrecp; i++) {
		stokes.uhat[i] = 0;
		stokes.vhat[i] = 0;
		stokes.what[i] = 0;
		stokes.phat[i] = 0;
		stokes.fxhat[i] = 0;
		stokes.fyhat[i] = 0;
		stokes.fzhat[i] = 0;
	}
}


void stokes_init_fft(StokesFFT &stokes) {

	// to be initialized
	stokes.fft = 0;

	//
	int nx = stokes.cellNum[0];
	int ny = stokes.cellNum[1];
	int nz = stokes.cellNum[2];

	int nrx = nx;
	int nry = ny;
	int nrz = nz;
	int ncx = nx;
	int ncy = ny;
	int ncz = nz;

	int ret = fft_r2c_init3d(stokes.fft, 
		nx, ny, nz, 
		nrx, nry, nrz, 
		ncx, ncy, ncz);

	if (ret != 0) {
		std::cerr << __FUNCTION__ << ": Failed to init FFT" << std::endl;
		exit(1);
	} else {
		std::cout << __FUNCTION__ << ": Init MKL FFT" << std::endl;
	}

}


//
int stokes_exec_fft(StokesFFT &stokes, double *u, complex_t *uhat) {
	// run FFT
	MKL_LONG status = DftiComputeForward(stokes.fft, u, (MKL_Complex16*) uhat);

	if (status == 0) {
		fft_conjeven_fill3d(stokes.cellNum[0], stokes.cellNum[1], stokes.cellNum[2], uhat);
	} else {
		std::cerr << __FUNCTION__ << ": error=" << status << std::endl;
	}

	return status;
}
int stokes_exec_ifft(StokesFFT &stokes, complex_t *uhat, double *u) {
	// run IFFT
	MKL_LONG status = DftiComputeBackward(stokes.fft, (MKL_Complex16*) uhat, u);

	if (status == 0) {
		fft_normalize3d(u, stokes.cellNum[0], stokes.cellNum[1], stokes.cellNum[2], stokes.cellNum[0], stokes.cellNum[1], stokes.cellNum[2]);
	} else {
		std::cerr << __FUNCTION__ << ": error=" << status << std::endl;
	}

	return status;
}


//
void stokes_solve(StokesFFT &stokes) {
	
	const double dens = stokes.dens;
	const double visc = stokes.visc;

	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];

	// transform source term
	stokes_exec_fft(stokes, stokes.fx, stokes.fxhat);
	stokes_exec_fft(stokes, stokes.fy, stokes.fyhat);
	stokes_exec_fft(stokes, stokes.fz, stokes.fzhat);

	// solve in spectral space
#pragma omp parallel for collapse(2) STOKES_OMP_SCHEDULE
	for (int i=0; i<nx; i++) {
	for (int j=0; j<ny; j++) {
	for (int k=0; k<nz; k++) {
		const int idx = i*ny*nz + j*nz + k;

		const double kx = stokes.kx[i];
		const double ky = stokes.ky[j];
		const double kz = stokes.kz[k];
		double kxx = kx * kx;
		double kxy = kx * ky;
		double kxz = kx * kz;
		double kyy = ky * ky;
		double kyz = ky * kz;
		double kzz = kz * kz;
		double k2 = kxx + kyy + kzz;


		complex_t fxh = stokes.fxhat[idx];
		complex_t fyh = stokes.fyhat[idx];
		complex_t fzh = stokes.fzhat[idx];

		if (k2 == 0) {
			stokes.uhat[idx] = 0;
			stokes.vhat[idx] = 0;
			stokes.what[idx] = 0;
			stokes.phat[idx] = 0;
		} else {
			double coef = 1.0 / k2;
			double cvel = coef / (visc * k2);
			double cpres = coef;

			complex_t uh = (k2-kxx)*fxh - kxy*fyh - kxz*fzh;
			complex_t vh = (k2-kyy)*fyh - kxy*fxh - kyz*fzh;
			complex_t wh = (k2-kzz)*fzh - kxz*fxh - kyz*fyh;
			complex_t ph = ImUnit * (-kx*fxh - ky*fyh - kz*fzh);

			stokes.uhat[idx] = cvel * uh;
			stokes.vhat[idx] = cvel * vh;
			stokes.what[idx] = cvel * wh;
			stokes.phat[idx] = cpres * ph;
		}
	}
	}
	}

	// back to real space
	stokes_exec_ifft(stokes, stokes.uhat, stokes.u);
	stokes_exec_ifft(stokes, stokes.vhat, stokes.v);
	stokes_exec_ifft(stokes, stokes.what, stokes.w);
	stokes_exec_ifft(stokes, stokes.phat, stokes.p);

}


void stokes_zero_buffer(StokesFFT &stokes, double *u) {
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	for (int i=0; i<nx*ny*nz; i++) {
		u[i] = 0;
	}
}


//
void stokes_fill_periodic(StokesFFT &stokes, const double *u, double *uper) {
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const int nx1 = nx + 1;
	const int ny1 = ny + 1;
	const int nz1 = nz + 1;

#pragma omp parallel for collapse(3) STOKES_OMP_SCHEDULE
	for (int i=0; i<nx1; i++) {
		for (int j=0; j<ny1; j++) {
			for (int k=0; k<nz1; k++) {
				int ii = i % nx;
				int jj = j % ny;
				int kk = k % nz;
				uper[i*ny1*nz1+j*nz1+k] = u[ii*ny*nz+jj*nz+kk];
			}
		}
	}
}





void fft_wavenum(int n, int ks[]) {
	if (n%2 == 0) {
		// even 
		int nhalf = n / 2;
		for (int i=0; i<nhalf; i++) {
			ks[i] = i;
			ks[i+nhalf] = i - nhalf;
		}
	} else {
		// odd
		int nhalf = (n-1) / 2;
		ks[0] = 0;
		for (int i=1; i<=nhalf; i++) {
			ks[i] = i;
			ks[i+nhalf] = i-1 - nhalf;
		}
	}
}
void fft_wavenum(int n, double L, double ks[]) {
	const double coef = 2.0*M_PI / L;
	if (n%2 == 0) {
		// even 
		int nhalf = n / 2;
		for (int i=0; i<nhalf; i++) {
			ks[i] = coef * i;
			ks[i+nhalf] = coef * (i - nhalf);
		}
	} else {
		// odd
		int nhalf = (n-1) / 2;
		ks[0] = 0;
		for (int i=1; i<=nhalf; i++) {
			ks[i] = coef * i;
			ks[i+nhalf] = coef * (i-1 - nhalf);
		}
	}	
}

void fft_conjeven_fill1d(int n, complex_t uhat[]) {
	if (n % 2 == 0) {
		int nh = n / 2;
		for (int i=nh+1; i<n; i++) {
			uhat[i] = std::conj(uhat[n-i]);
		}
	} else {
		int nh = (n-1) / 2;
		for (int i=nh+1; i<n; i++) {
			uhat[i] = std::conj(uhat[n-i]);
		}
	}
}

void fft_conjeven_fill2d(int nx, int ny, complex_t uhat[]) {

	const int nyh = ny/2 + 1;

	for (int i=0; i<nx; i++) {
		for (int j=nyh; j<ny; j++) {
			int ic = (nx-i) % nx;
			int jc = (ny-j) % ny;

			int ind = i*ny + j;
			int indc = ic*ny + jc;
			
			uhat[ind] = std::conj(uhat[indc]);
		}
	}
}

void fft_conjeven_fill3d(int nx, int ny, int nz, complex_t uhat[]) {

	const int nzh = nz/2 + 1;

#pragma omp parallel for collapse(2) STOKES_OMP_SCHEDULE
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=nzh; k<nz; k++) {
				int ic = (nx-i) % nx;
				int jc = (ny-j) % ny;
				int kc = (nz-k) % nz;

				int ind = i*ny*nz + j*nz + k;
				int indc = ic*ny*nz + jc*nz + kc;

				uhat[ind] = std::conj(uhat[indc]);
			}
		}
	}
}

void fft_conjeven_fill3d(complex_t uhat[], int nx, int ny, int nz, int nbufx, int nbufy, int nbufz) {
	// lowest dimension
	const int nzh = nz/2 + 1;

#pragma omp parallel for collapse(2) STOKES_OMP_SCHEDULE
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=nzh; k<nz; k++) {
				int ic = (nx-i) % nx;
				int jc = (ny-j) % ny;
				int kc = (nz-k) % nz;

				int ind = i*nbufy*nbufz + j*nbufz + k;
				int indc = ic*nbufy*nbufz + jc*nbufz + kc;

				uhat[ind] = std::conj(uhat[indc]);
			}
		}
	}
}

void fft_normalize3d(double uback[], int nx, int ny, int nz, int nrx, int nry, int nrz) {
	// normalization 
	const double coef = 1.0 / (double) (nx*ny*nz);

//#pragma omp parallel for collapse(2) STOKES_OMP_SCHEDULE
//	for (int i=0; i<nx; i++) {
//		for (int j=0; j<ny; j++) {
//			for (int k=0; k<nz; k++) {
//				int idx = i*nry*nrz + j*nrz + k;
//				uback[idx] *= coef;
//			}
//		}
//	}

#pragma omp parallel for STOKES_OMP_SCHEDULE
	for (int idx=0; idx<nrx*nry*nrz; idx++) {
		uback[idx] *= coef;
	}
}



int fft_r2c_init2d(DFTI_DESCRIPTOR_HANDLE &fft,
	int nx, int ny, int nxbuf, int nybuf)
{
	//
	MKL_LONG status = 0;
	fft = 0;

	char version[DFTI_VERSION_LENGTH];
	DftiGetValue(0, DFTI_VERSION, version);
	std::cout << version << std::endl;

	// create 
	const int ndim = 2;
	const MKL_LONG N[ndim] = { nx, ny };
	status = DftiCreateDescriptor(&fft, DFTI_DOUBLE, DFTI_REAL, ndim, N);
	if (status != 0) goto failed;

	// set out-of-place
	status = DftiSetValue(fft, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != 0) goto failed;

	// ???
	if (1) {
		status = DftiSetValue(fft, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
		if (status != 0) goto failed;
	}
	if (0) {
		status = DftiSetValue(fft, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
		if (status != 0) goto failed;
	}

	// set input stride
	MKL_LONG istride[ndim+1] = { 0, nybuf, 1 };
	status = DftiSetValue(fft, DFTI_INPUT_STRIDES, istride);
	if (status != 0) goto failed;

	// set output stride
	MKL_LONG ostride[ndim+1] = { 0, nybuf, 1 };
	status = DftiSetValue(fft, DFTI_OUTPUT_STRIDES, ostride);
	if (status != 0) goto failed;

	// commit
	status = DftiCommitDescriptor(fft);
	if (status != 0) goto failed;

finished:
	return (int) status;

failed:
	std::cerr << __FUNCTION__ << ": error=" << status << std::endl;
	// clean up
	DftiFreeDescriptor(&fft);
	goto finished;

}

int fft_r2c_init3d(DFTI_DESCRIPTOR_HANDLE &fft,
	int nx, int ny, int nz, 
	int nrx, int nry, int nrz, 
	int ncx, int ncy, int ncz)
{
	//
	MKL_LONG status = 0;
	fft = 0;

	char version[DFTI_VERSION_LENGTH];
	DftiGetValue(0, DFTI_VERSION, version);
	std::cout << version << std::endl;

	// create 
	const int ndim = 3;
	const MKL_LONG N[ndim] = { nx, ny, nz };
	status = DftiCreateDescriptor(&fft, DFTI_DOUBLE, DFTI_REAL, ndim, N);
	if (status != 0) goto failed;

	// set out-of-place
	status = DftiSetValue(fft, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != 0) goto failed;

	// ???
	if (1) {
		status = DftiSetValue(fft, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
		if (status != 0) goto failed;
	}
	if (0) {
		status = DftiSetValue(fft, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
		if (status != 0) goto failed;
	}

	// set input stride
	MKL_LONG istride[ndim+1] = { 0, nry*nrz, nrz, 1 };
	status = DftiSetValue(fft, DFTI_INPUT_STRIDES, istride);
	if (status != 0) goto failed;

	// set output stride
	MKL_LONG ostride[ndim+1] = { 0, ncy*ncz, ncz, 1 };
	status = DftiSetValue(fft, DFTI_OUTPUT_STRIDES, ostride);
	if (status != 0) goto failed;

	// commit
	status = DftiCommitDescriptor(fft);
	if (status != 0) goto failed;

finished:
	return (int) status;

failed:
	std::cerr << __FUNCTION__ << ": error=" << status << std::endl;
	// clean up
	DftiFreeDescriptor(&fft);
	fft = 0;
	goto finished;

}


