

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include <iostream>
#include <complex>

#include "StokesFFT.h"

// MKL
#include <mkl_dfti.h>







static std::ostream& operator<< (std::ostream& os, const MKL_Complex16 &c) {
	os << c.real;
	if (c.imag >= 0) {
		os << "+";
	}
	os << c.imag << "i";

	return os;
}


enum ArrayFormat {
	DataRowMajor,
	DataColMajor,
};


void fft_recover_half(int n, MKL_Complex16 uhat[]) {
	fft_conjeven_fill1d(n, (complex_t*) uhat);
}


// in-place recover, assume there is enough space 
void fft_recover_2d(int nx, int ny, MKL_Complex16 uhat[]) {
	fft_conjeven_fill2d(nx, ny, (complex_t*) uhat);
}


static void test_fft1d() {

	//prepare data
	const int ndim = 1;
	const int nbuf = 33;
	double xs[nbuf], us[nbuf];
	for (int i=0; i<nbuf; i++) {
		double xx = M_PI*2.0 / (nbuf-1) * i;
		xs[i] = xx;
		us[i] = sin(xx) + sin(xx*3) + sin(xx*4);

		//std::cout << xs[i] << "=" << ys[i] << std::endl;
	}

	// output buffer
	MKL_Complex16 uhat[nbuf] = { 0 };
	double ub[nbuf] = { 0 };


	//
	MKL_LONG status = 0;
	DFTI_DESCRIPTOR_HANDLE fft = 0;

	char version[DFTI_VERSION_LENGTH];
	DftiGetValue(0, DFTI_VERSION, version);
	std::cout << "version=" << version << std::endl;

	// create 
	status = DftiCreateDescriptor(&fft, DFTI_DOUBLE, DFTI_REAL, ndim, (MKL_LONG) nbuf);
	if (status != 0) goto failed;

	// set out-of-place
	status = DftiSetValue(fft, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
	if (status != 0) goto failed;

	// ???
	if (0) {
	status = DftiSetValue(fft, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
	if (status != 0) goto failed;
	}
	if (0) {
	status = DftiSetValue(fft, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
	if (status != 0) goto failed;
	}

	// commit
	status = DftiCommitDescriptor(fft);
	if (status != 0) goto failed;

	// forward FFT
	status = DftiComputeForward(fft, us, uhat);
	if (status != 0) goto failed;

	// MKL only has half vector for real->complex
	fft_recover_half(nbuf, uhat);

	if (1) {
		std::cout << "uhat=" << std::endl;
		for (int i=0; i<nbuf; i++) {
			std::cout << uhat[i].real << "+" << uhat[i].imag << "i" << std::endl;
		}
	}

	// backward FFT
	status = DftiComputeBackward(fft, uhat, ub);
	if (status != 0) goto failed;

	// MKL needs normalization
	for (int i=0; i<nbuf; i++) {
		ub[i] /= nbuf;
	}

	if (1) {
		std::cout << "ub=" << std::endl;
		for (int i=0; i<nbuf; i++) {
			std::cout << ub[i] << std::endl;
		}
	}

cleanup:
	//
	DftiFreeDescriptor(&fft);

	return;

failed:
	std::cerr << "FFT error=" << status << std::endl;
	goto cleanup;

}


static void test_fft2d() {
		
	const ArrayFormat datfmt = DataRowMajor;

	// data dimension
	const int ndim = 2;
	const int nx = 4;
	const int ny = 3;
	const MKL_LONG N[ndim] = { nx, ny }; 

	//
	const int nxh = nx/2 + 1;
	const int nyh = ny/2 + 1;

	// total number
	const int nbuf = nx * ny;

	// input real 
	double us[nbuf] = { 0 };

	// real -> complex
	//MKL_Complex16 uhat[nbuf] = { 0 };
	complex_t uhat[nbuf];
	
	// complex -> real
	double ub[nbuf] = { 0 };

	{
		const double data[nx][ny] = {
			0.814723686393179,   0.632359246225410,   0.957506835434298,
			0.905791937075619,   0.097540404999410,   0.964888535199277,
			0.126986816293506,   0.278498218867048,   0.157613081677548,
			0.913375856139019,   0.546881519204984,   0.970592781760616,
		};

		if (datfmt == DataRowMajor) {
			// create row-major data
			for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = i*ny + j;
				us[idx] = data[i][j];
			}
			}
		}
	}

	//
	MKL_LONG status = 0;
	DFTI_DESCRIPTOR_HANDLE fft = 0;

	char version[DFTI_VERSION_LENGTH];
	DftiGetValue(0, DFTI_VERSION, version);
	std::cout << "version=" << version << std::endl;

	// create 
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
	if (1) {
		status = DftiSetValue(fft, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
		if (status != 0) goto failed;
	}

	// set input stride
	MKL_LONG rc_istride[ndim+1];
	if (datfmt == DataRowMajor) {
		rc_istride[0] = 0;
		rc_istride[1] = ny;
		rc_istride[2] = 1;
	}
	status = DftiSetValue(fft, DFTI_INPUT_STRIDES, rc_istride);
	if (status != 0) goto failed;

	// set output stride
	MKL_LONG rc_ostride[ndim+1];
	if (datfmt == DataRowMajor) {
		rc_ostride[0] = 0;
		//rc_ostride[1] = nyh;
		rc_ostride[1] = ny;
		rc_ostride[2] = 1;
	}
	status = DftiSetValue(fft, DFTI_OUTPUT_STRIDES, rc_ostride);
	if (status != 0) goto failed;

	// commit
	status = DftiCommitDescriptor(fft);
	if (status != 0) goto failed;

	// forward FFT
	for (int iter=0; iter<5; iter++) {
		status = DftiComputeForward(fft, us, uhat);
		if (status != 0) goto failed;
	}

	if (1) {
		fft_recover_2d(nx, ny, (MKL_Complex16*) uhat);
	}

	if (1) {
		std::cout << "uhat=" << std::endl;
		for (int i=0; i<nx; i++) {
			//for (int j=0; j<nyh; j++) {
			for (int j=0; j<ny; j++) {
				int idx = i*rc_ostride[1] + j*rc_ostride[2];
				std::cout << uhat[idx] << ",";
			}
			std::cout << std::endl;
		}
	}

	// backward FFT
	status = DftiComputeBackward(fft, uhat, ub);
	if (status != 0) goto failed;

	// MKL needs normalization
	for (int i=0; i<nbuf; i++) {
		ub[i] /= nbuf;
	}

	if (1) {
		std::cout << "ub=" << std::endl;
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = i*ny + j;
				std::cout << ub[idx] << ",";
			}
			std::cout << std::endl;
		}
	}

cleanup:
	//
	DftiFreeDescriptor(&fft);

	return;

failed:
	std::cerr << "FFT error=" << status << std::endl;
	goto cleanup;

}


static void test_fft3d() {

	// data dimension
	const int ndim = 3;
	const int nx = 5;
	const int ny = 3;
	const int nz = 4;
	const int nx1 = nx + 1;
	const int ny1 = ny + 1;
	const int nz1 = nz + 1;

	// total number
	const int nbuf = nx * ny * nz;
	const int nper = (nx1) * (ny1) * (nz1);

	// input real 
	double us[nper] = { 0 };

	// real -> complex
	complex_t uhat[nper];
	
	// complex -> real
	double ub[nper] = { 0 };


	{
		double data[nbuf] = {
			0.694828622975817,   0.679702676853675,   0.547215529963803,   0.830828627896291,
			0.381558457093008,   0.959743958516081,   0.254282178971531,   0.757200229110721,
			0.445586200710899,   0.255095115459269,   0.196595250431208,   0.053950118666607,
			0.317099480060861,   0.655098003973841,   0.138624442828679,   0.585264091152724,
			0.765516788149002,   0.340385726666133,   0.814284826068816,   0.753729094278495,
			0.646313010111265,   0.505957051665142,   0.251083857976031,   0.530797553008973,
			0.950222048838355,   0.162611735194631,   0.149294005559057,   0.549723608291140,
			0.795199901137063,   0.585267750979777,   0.243524968724989,   0.380445846975357,
			0.709364830858073,   0.699076722656686,   0.616044676146639,   0.779167230102011,
			0.034446080502909,   0.118997681558377,   0.257508254123736,   0.917193663829810,
			0.186872604554379,   0.223811939491137,   0.929263623187228,   0.567821640725221,
			0.754686681982361,   0.890903252535799,   0.473288848902729,   0.934010684229183,
			0.438744359656398,   0.498364051982143,   0.840717255983663,   0.285839018820374,
			0.489764395788231,   0.751267059305653,   0.349983765984809,   0.075854289563064,
			0.276025076998578,   0.959291425205444,   0.351659507062997,   0.129906208473730,
		};
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					int isrc = i*ny*nz + j*nz + k;
					int idst = i*(ny1)*(nz1) + j*(nz1) + k;
					us[idst] = data[isrc];
				}
			}
		}
	}

	//
	DFTI_DESCRIPTOR_HANDLE fft = 0;

	MKL_LONG status = fft_r2c_init3d(fft, nx, ny, nz, nx1, ny1, nz1, nx1, ny1, nz1);
	if (status != 0) {
		exit(1);
	}


	// forward FFT
	for (int iter=0; iter<5; iter++) {
		status = DftiComputeForward(fft, us, uhat);
		if (status != 0) goto failed;
	}

	if (1) {
		fft_conjeven_fill3d(uhat, nx,ny,nz, nx1,ny1,nz1);
	}

	if (1) {
		std::cout << "uhat=" << std::endl;
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					int idx = i*ny1*nz1 + j*nz1 + k;
					std::cout << uhat[idx] << ",";
				}
				std::cout << std::endl;
			}
		}
	}

	// backward FFT
	for (int iter=0; iter<5; iter++) {
		status = DftiComputeBackward(fft, uhat, ub);
		if (status != 0) goto failed;
	}

	// MKL needs normalization
	for (int i=0; i<nper; i++) {
		ub[i] /= nbuf;
	}

	if (1) {
		std::cout << "ub=" << std::endl;
		for (int i=0; i<nx1; i++) {
			for (int j=0; j<ny1; j++) {
				for (int k=0; k<nz1; k++) {
					int idx = i*ny1*nz1 + j*nz1 + k;
					std::cout << ub[idx] << ",";
				}
				std::cout << std::endl;
			}
		}
	}

cleanup:
	DftiFreeDescriptor(&fft);
	fft = 0;
	return;

failed:
	std::cerr << "FFT error=" << status << std::endl;
	goto cleanup;

}





static int stokes2d_init_fft(const StokesFFT2D &stokes2d, DFTI_DESCRIPTOR_HANDLE &fft)
{
	//
	MKL_LONG status = 0;
	fft = 0;

	char version[DFTI_VERSION_LENGTH];
	DftiGetValue(0, DFTI_VERSION, version);
	std::cout << "version=" << version << std::endl;

	// create 
	const int ndim = 2;
	const int nx = stokes2d.nx;
	const int ny = stokes2d.ny;
	const MKL_LONG N[] = { nx, ny };
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
	MKL_LONG rc_istride[ndim+1] = { 0, ny, 1 };
	status = DftiSetValue(fft, DFTI_INPUT_STRIDES, rc_istride);
	if (status != 0) goto failed;

	// set output stride
	MKL_LONG rc_ostride[ndim+1] = { 0, ny, 1 };
	status = DftiSetValue(fft, DFTI_OUTPUT_STRIDES, rc_ostride);
	if (status != 0) goto failed;

	// commit
	status = DftiCommitDescriptor(fft);
	if (status != 0) goto failed;

quit:
	return (int) status;

failed:
	std::cerr << "FFT error=" << status << std::endl;
	// clean up
	DftiFreeDescriptor(&fft);
	goto quit;
}

int fft2(DFTI_DESCRIPTOR_HANDLE &fft,
	double *rin, complex_t *cout,
	int nx, int ny)
{
	MKL_Complex16 *hat = (MKL_Complex16*) cout;

	MKL_LONG status = DftiComputeForward(fft, rin, hat);

	fft_recover_2d(nx, ny, hat);

	return status;
}

int ifft2(DFTI_DESCRIPTOR_HANDLE &fft,
	complex_t *cin, double *rout,
	int nx, int ny)
{
	MKL_Complex16 *hat = (MKL_Complex16*) cin;
	
	MKL_LONG status = DftiComputeBackward(fft, hat, rout);

	const int ndat = nx * ny;
	for (int i=0; i<ndat; i++) {
		rout[i] /= ndat;
	}

	return status;
}



void test_stokes2d()
{
	StokesFFT2D stokes2d;

	// density & viscosity
	const double rho = 1.0;
	const double mu = 1.0;
	const double nu = mu / rho;

	stokes2d.rho = rho;
	stokes2d.mu = mu;

	// domain
	const double xlo = 0;
	const double xhi = M_PI * 2.0;
	const double ylo = 0;
	const double yhi = M_PI * 2.0;

	stokes2d.xlo = xlo;
	stokes2d.xhi = xhi;
	stokes2d.ylo = ylo;
	stokes2d.yhi = yhi;

	// grid
	const int nx = 64;
	const int ny = 64;
	const int ndat = nx * ny;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;

	stokes2d.nx = nx;
	stokes2d.ny = ny;
	stokes2d.dx = (xhi-xlo) / nx;
	stokes2d.dy = (yhi-ylo) / ny;

	//
	stokes2d.kx = new int[nx];
	stokes2d.ky = new int[ny];
	// calculate wave number
	fft_wavenum(nx, stokes2d.kx);
	fft_wavenum(ny, stokes2d.ky);

	//
	stokes2d.u = new double[ndat];
	stokes2d.v = new double[ndat];
	stokes2d.p = new double[ndat];
	stokes2d.f = new double[ndat];
	stokes2d.g = new double[ndat];

	stokes2d.uhat = new complex_t[ndat];
	stokes2d.vhat = new complex_t[ndat];
	stokes2d.phat = new complex_t[ndat];
	stokes2d.fhat = new complex_t[ndat];
	stokes2d.ghat = new complex_t[ndat];

	for (int i=0; i<ndat; i++) {
		stokes2d.u[i] = 0;
		stokes2d.v[i] = 0;
		stokes2d.p[i] = 0;
		stokes2d.f[i] = 0;
		stokes2d.g[i] = 0;
	}

	// set forcing
	for (int i=0; i<nx; i++) {
	for (int j=0; j<ny; j++) {
		double x = xlo + dx*(double) i;
		double y = ylo + dy*(double) j;

		double f = sin(x)*cos(x) + 2.0*nu*cos(x)*sin(y);
		double g = sin(y)*cos(y) - 2.0*nu*sin(x)*cos(y);

		int ind = i*ny + j;
		stokes2d.f[ind] = f;
		stokes2d.g[ind] = g;
	}
	}

	//
	DFTI_DESCRIPTOR_HANDLE fft = 0;
	MKL_LONG status = stokes2d_init_fft(stokes2d, fft);

	// transform source term
	fft2(fft, stokes2d.f, stokes2d.fhat, nx, ny);
	fft2(fft, stokes2d.g, stokes2d.ghat, nx, ny);

	//
	for (int i=0; i<nx; i++) {
	for (int j=0; j<ny; j++) {
		const int idx = i*ny + j;

		double ki = stokes2d.kx[i];
		double kj = stokes2d.ky[j];
		double k2 = ki*ki + kj*kj;

		complex_t fh = stokes2d.fhat[idx];
		complex_t gh = stokes2d.ghat[idx];

		if (k2 == 0) {
			stokes2d.uhat[idx] = 0;
			stokes2d.vhat[idx] = 0;
			stokes2d.phat[idx] = 0;
		} else {
			double coef = 1.0 / k2;
			double cvel = coef / (nu * k2);
			complex_t uh = kj*kj*fh - ki*kj*gh;
			complex_t vh = ki*ki*gh - ki*kj*fh;
			complex_t ph = ImUnit * (-ki*fh - kj*gh);

			stokes2d.uhat[idx] = cvel * uh;
			stokes2d.vhat[idx] = cvel * vh;
			stokes2d.phat[idx] = coef * ph;
		}
	}
	}

	// back to real space
	ifft2(fft, stokes2d.uhat, stokes2d.u, nx, ny);
	ifft2(fft, stokes2d.vhat, stokes2d.v, nx, ny);
	ifft2(fft, stokes2d.phat, stokes2d.p, nx, ny);

	//
	DftiFreeDescriptor(&fft);
	fft = 0;

	if (1) {
		FILE *fp = fopen("hoge00.csv", "w");

		fprintf(fp, "x,y,z,u,v,p,f,g\n");

		for (int j=0; j<=ny; j++) {
		for (int i=0; i<=nx; i++) {
			int imap = i<nx ? i : 0;
			int jmap = j<ny ? j : 0;
			int idx = imap*ny + jmap;

			double x = xlo + dx * (double) i;
			double y = ylo + dy * (double) j;
			double z = 0;

			fprintf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
				x, y, z, 
				stokes2d.u[idx], stokes2d.v[idx], stokes2d.p[idx],
				stokes2d.f[idx], stokes2d.g[idx]);
		}
		}

		fclose(fp);
	}


	return;
}


void test_stokes3d(int argc, char *argv[]) {
	
	//
	StokesFFT stokes;

	// density & viscosity
	const double rho = 1.0;
	const double mu = 1.0;
	const double nu = mu / rho;

	stokes.dens = rho;
	stokes.visc = mu;

	// domain
	//const double xlo = -5;
	//const double xhi = 5;
	//const double ylo = -5;
	//const double yhi = 5;
	//const double zlo = -5;
	//const double zhi = 5;
	//const double xlo = -1;
	//const double xhi = 1;
	//const double ylo = -1;
	//const double yhi = 1;
	//const double zlo = -1;
	//const double zhi = 1;
	const double xlo = 0;
	const double xhi = 2.0*M_PI;
	const double ylo = 0;
	const double yhi = 2.0*M_PI;
	const double zlo = 0;
	const double zhi = 2.0*M_PI;
	
	stokes.problo[0] = xlo;
	stokes.problo[1] = ylo;
	stokes.problo[2] = zlo;
	stokes.probhi[0] = xhi;
	stokes.probhi[1] = yhi;
	stokes.probhi[2] = zhi;

	// grid
	const int ncell = 64;
	const int nx = ncell;
	const int ny = ncell;
	const int nz = ncell;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;
	const double dz = (zhi-zlo) / nz;

	stokes.cellNum[0] = nx;
	stokes.cellNum[1] = ny;
	stokes.cellNum[2] = nz;

	//
	stokes_init(stokes);
	stokes_init_fft(stokes);

	// set forcing
	if (0) {
		int iforce = nx/2;
		int jforce = ny/2;
		int kforce = nz/2;
		stokes.fz[iforce*ny*nz+jforce*nz+kforce] = 1.0 / (dx*dy*dz);
	}
	if (0) {
		const double aa = 2.0 * M_PI;
		const double bb = 2.0 * M_PI;
		const double cc = 2.0 * M_PI;
		const double A = 1.0;
		const double B = 0.0;
		const double C = -(A+B);

		for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
		for (int k=0; k<nz; k++) {
			double x = xlo + dx*(double) i;
			double y = ylo + dy*(double) j;
			double z = zlo + dz*(double) k;

			double u = A * cos(aa*x) * sin(bb*y) * sin(cc*z);
			double v = B * sin(aa*x) * cos(bb*y) * sin(cc*z);
			double w = C * sin(aa*x) * sin(bb*y) * cos(cc*z);

			double ux = -A * aa * sin(aa*x) * sin(bb*y) * sin(cc*z);
			double uy =  A * bb * cos(aa*x) * cos(bb*y) * sin(cc*z);
			double uz =  A * cc * cos(aa*x) * sin(bb*y) * cos(cc*z);

			double vx =  B * aa * cos(aa*x) * cos(bb*y) * sin(cc*z);
			double vy = -B * bb * sin(aa*x) * sin(bb*y) * sin(cc*z);
			double vz =  B * cc * sin(aa*x) * cos(bb*y) * cos(cc*z);

			double wx =  C * aa * cos(aa*x) * sin(bb*y) * cos(cc*z);
			double wy =  C * bb * sin(aa*x) * cos(bb*y) * cos(cc*z);
			double wz = -C * cc * sin(aa*x) * sin(bb*y) * sin(cc*z);

			double tcoef = -2.0 * nu;
			double ut = tcoef * u;
			double vt = tcoef * v;
			double wt = tcoef * w;

			double fx = - (ut + u*ux + v*uy + w*uz);
			double fy = - (vt + u*vx + v*vy + w*vz);
			double fz = - (wt + u*wx + v*wy + w*wz);

			int ind = i*ny*nz + j*nz + k;
			stokes.fx[ind] = fx;
			stokes.fy[ind] = fy;
			stokes.fz[ind] = fz;
		}
		}
		}
	}
	if (1) {
		for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
		for (int k=0; k<nz; k++) {
			double x = xlo + dx*(double) i;
			double y = ylo + dy*(double) j;
			double z = zlo + dz*(double) k;

			double xx = y;
			double yy = z;

			double f = sin(xx)*cos(xx) + 2.0*nu*cos(xx)*sin(yy);
			double g = sin(yy)*cos(yy) - 2.0*nu*sin(xx)*cos(yy);

			double fx = 0;
			double fy = f;
			double fz = g;

			int ind = i*ny*nz + j*nz + k;
			stokes.fx[ind] = fx;
			stokes.fy[ind] = fy;
			stokes.fz[ind] = fz;
		}
		}
		}
	}

	// solve u,v,w,p
	stokes_solve(stokes);

	if (0) {
		FILE *fp = fopen("hoge3d00.csv", "w");

		fprintf(fp, "x,y,z,u,v,w,p,fx,fy,fz\n");

		for (int k=0; k<=nz; k++) {
		for (int j=0; j<=ny; j++) {
		for (int i=0; i<=nx; i++) {
			int imap = i<nx ? i : 0;
			int jmap = j<ny ? j : 0;
			int kmap = k<nz ? k : 0;
			int idx = imap*ny*nz + jmap*nz + kmap;

			double x = xlo + dx * (double) i;
			double y = ylo + dy * (double) j;
			double z = zlo + dz * (double) k;

			fprintf(fp, "%lf,%lf,%lf" ",%lf,%lf,%lf,%lf" ",%lf,%lf,%lf" "\n",
				x, y, z, 
				stokes.u[idx], stokes.v[idx], stokes.w[idx], stokes.p[idx],
				stokes.fx[idx], stokes.fy[idx], stokes.fz[idx]);
		}
		}
		}

		fclose(fp);
	}

	stokes_output_vtk(stokes, "hoge3d00.vtk");
}


extern void test_stokes_force(int argc, char *argv[]);
extern void test_stokes_bndry(int argc, char *argv[]);
extern void test_stokes_drag(int argc, char *argv[]);
extern void test_stokes_part(int argc, char *argv[]);

void test_stokes_partmove(int argc, char *argv[]);

int main(int argc, char *argv[])
{

	//test_fft1d();

	//test_fft2d();

	//test_fft3d();

	//test_stokes2d();

	//test_stokes3d(argc, argv);

	//test_stokes_force(argc, argv);

	//test_stokes_bndry(argc, argv);
	//test_stokes_drag(argc, argv);

	//test_stokes_part(argc, argv);

	test_stokes_partmove(argc, argv);

	return 0;
}






