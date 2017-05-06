
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cassert>

#include <iostream>
#include <complex>

// MKL headers
#include <mkl_blas.h>
#include <mkl_spblas.h>
#include <mkl_rci.h>
#include <mkl_service.h>

#include "SparseUtil.h"


//
void CSRMat::begin_row(int i) {
	if (i == 0) { // first row
		// first row begins with the first data
		m_ia.push_back(0+IndexBase);
	}
}
void CSRMat::push_data(int i, int j, double a) {
	m_ja.push_back(j+IndexBase);
	m_data.push_back(a);
	ndata += 1;
}
void CSRMat::end_row(int i) {
	m_ia.push_back(ndata+IndexBase);
}

//
void CSRMat::clear() {
	ndata = 0;
	m_ia.clear();
	m_ja.clear();
	m_data.clear();
}

//
void CSRMat::reserve(int capacity) {
	m_ia.reserve(capacity);
	m_ja.reserve(capacity);
	m_data.reserve(capacity);
}

// square matrix version
// y = A.x
void CSRMat::gemv(const double *xin, double *yout, char trans) {
	// check row=col
	if (nrow != ncol) {
		std::cerr << __FUNCTION__ << ": nrow!=ncol" << std::endl;
		exit(1);
	}

	MKL_INT ivar = nrow;
	int *ia = this->ia();
	int *ja = this->ja();
	double *a = this->data();

	mkl_dcsrgemv(&trans, &ivar, a, ia, ja, const_cast<double*>(xin), yout);
}

// general version
// y = alpha*A.x + beta*y
void CSRMat::gemv(double alpha, const double *xin, double beta, double *yout, char trans) {

	// matrix size
	int m = nrow;
	int n = ncol;

	// matrix descriptor
	// desc[0] = general
	// desc[3] = one- or zero-based index
	char matdesc[6] = {0};
	matdesc[0] = 'g';
	matdesc[3] = IndexBase==1 ? 'f' : 'c';

	//
	int *ia = this->ia();
	int *ja = this->ja();
	double *val = this->data();

	mkl_dcsrmv(&trans, &m, &n, 
		&alpha, matdesc, 
		val, ja, ia, ia+1, 
		const_cast<double*>(xin), &beta, yout);
}


// perform ILU(0) decomp.
int CSRMat::ilu0(double *bilu0, double replace_zero_diag) {
	// only apply to square matrix
	assert(nrow == ncol);

	const int npar = 128;
	MKL_INT ipar[npar] = { 0 };
	double dpar[npar] = { 0 };

	// set parameters
	ipar[1] = 6; // output error message to screen
	ipar[5] = 1; // allow output error
	
	if (replace_zero_diag > 0) {
		// check small diagonal value and replace
		ipar[30] = 1;
		dpar[30] = replace_zero_diag;
		dpar[31] = replace_zero_diag;
	}

	// matrix 
	MKL_INT ivar = nrow;
	int *ia = this->ia();
	int *ja = this->ja();
	double *val = this->data();

	// call MKL 
	MKL_INT ierr = 0;
	dcsrilu0(&ivar, val, ia, ja, bilu0, ipar, dpar, &ierr);
	
	if (ierr != 0) {
		std::cerr << __FUNCTION__ << ": ierr=" << ierr << std::endl;
	}

	return (int) ierr;
}

int CSRMat::ilu0(double replace_zero_diag) {
	// only apply to square matrix
	assert(nrow == ncol);

	m_decomp.resize(ndata);
	m_trvec.resize(nrow);

	const int npar = 128;
	MKL_INT ipar[npar] = { 0 };
	double dpar[npar] = { 0 };

	// set parameters
	ipar[1] = 6; // output error message to screen
	ipar[5] = 1; // allow output error
	
	if (replace_zero_diag > 0) {
		// check small diagonal value and replace
		ipar[30] = 1;
		dpar[30] = replace_zero_diag;
		dpar[31] = replace_zero_diag;
	}

	// matrix 
	MKL_INT ivar = nrow;
	int *ia = this->ia();
	int *ja = this->ja();
	double *val = this->data();

	//
	double *bilu0 = &m_decomp[0];

	// call MKL 
	MKL_INT ierr = 0;
	dcsrilu0(&ivar, val, ia, ja, bilu0, ipar, dpar, &ierr);
	
	if (ierr != 0) {
		std::cerr << __FUNCTION__ << ": ierr=" << ierr << std::endl;
	}

	return (int) ierr;
}

void CSRMat::solve_decomp(const double *xin, double *yout) {
	// only apply to square matrix
	assert(nrow == ncol);
	// must already decomposed
	assert(m_decomp.size() == ndata);
	assert(m_trvec.size() == nrow);

	// 
	MKL_INT ivar = nrow;
	int *ia = this->ia();
	int *ja = this->ja();

	//
	double *bilu0 = &m_decomp[0];
	double *trvec = &m_trvec[0];

	char cvar1, cvar, cvar2;

	// Lower solve
	cvar1 = 'L'; 
	cvar = 'N';
	cvar2 = 'U';
	mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, const_cast<double*>(xin), trvec);

	// Upper solve
	cvar1 = 'U'; 
	cvar = 'N';
	cvar2 = 'N';
	mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, trvec, yout);

}



