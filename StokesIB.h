#pragma once

#include <cmath>

#include "SparseUtil.h"



struct PointCache
{
	double xmin[3];
	double xmax[3];
	int nbin[3];
	double hbin[3];

	int *head;
	int *tail;
	int *next;

};

//
//
//
struct StokesForcing {

	StokesFFT *stokes;

	double xi;
	double alpha;
	double rcut;

	//
	int nall;
	double *xall;
	double *uall;
	double *fall;
	int *idall;

	// boundary point
	int nb;
	double *xb;
	double *ub;
	double *fb;
	int *idb;

	// particle point
	int np;
	double *xp;
	double *up;
	double *fp;
	int *idp;

	int part_num;
	double *part_rad;
	double *part_pos;
	double *part_vel;
	double *part_force;
	int *part_ipbegin;
	int *part_ipend;


	// dirty
	double *tmp[3];

	// 
	PointCache *cache;



	//
	CSRMat *mat_local;
	
	// 
	CSRMat *mat_distrib;

	//
	CSRMat *mat_interp;

	//
	CSRMat *mat_precond;


};


struct StokesIB 
{
	
};


void stokes_forcing_output_csv(const StokesForcing &sf, const char outfilename[]);
void stokes_forcing_output_csv2(const StokesForcing &sf, const char outfilename[]);
void stokes_forcing_output_vtk(const StokesForcing &sf, const char filename[]);
void stokes_forcing_output_particle_vtk(const StokesForcing &sf, const char filename[]);


namespace 
{
	const double PI = M_PI;
	const double TwoPI = PI * 2;
	const double FourPI = PI * 4;
	const double SixPI = PI * 6;
	const double EightPI = PI * 8;
	const double SqrtPI = sqrt(PI);

};


