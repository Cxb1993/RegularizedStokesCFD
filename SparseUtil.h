#pragma once




#include <vector>




//
// Compressed Sparse Row
// 3-array for MKL
//
class CSRMat 
{
public:

	enum { // internal index
		IndexBase = 1, // use one-based indexing
	};

	CSRMat(int m, int n)
		: nrow(m), ncol(n), ndata(0)
	{
		//int ncap = m * n * 40;
		//reserve(ncap);
	}


	//
	int* ia() { return &m_ia[0]; }
	const int* ia() const { return &m_ia[0]; };
	//
	int* ja() { return &m_ja[0]; }
	const int* ja() const { return &m_ja[0]; };
	//
	double *data() { return &m_data[0]; }
	const double *data() const { return &m_data[0]; }

	//
	int num_row() const { return nrow; }
	int num_col() const { return ncol; }
	int num_val() const { return ndata; }

	//
	// serial build per row
	//
	void begin_row(int i);
	void push_data(int i, int j, double a);
	void end_row(int i);

	// clear all
	void clear();

	// pre-alloc memory, do not change data
	void reserve(int capacity);

	//
	// square matrix version
	// y = A.x
	//
	void gemv(const double *xin, double *yout, char trans='N');

	//
	// general version
	// y = alpha*A.x + beta*y
	//
	void gemv(double alpha, const double *xin, double beta, double *yout, char trans='N');

	//
	// perform ILU(0) decomp.
	//
	int ilu0(double *bilu0, double replace_zero_diag=0);
	int ilu0(double replace_zero_diag=0);

	void solve_decomp(const double *xin, double *yout);

	//
	// members 
	//
	const int nrow;
	const int ncol;
	int ndata;

	// CSR 3-array
	std::vector<int> m_ia;
	std::vector<int> m_ja;
	std::vector<double> m_data;

	// 
	std::vector<double> m_decomp;
	std::vector<double> m_trvec;

private:
	// disable copy
	CSRMat(const CSRMat&);
};


//



