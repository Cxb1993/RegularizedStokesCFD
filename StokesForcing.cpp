
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cassert>

#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <map>
#include <set>
#include <memory>


#include "StokesFFT.h"
#include "SparseUtil.h"
#include "StokesIB.h"
#include "SpecialFunc.h"

#include <mkl_blas.h>
#include <mkl_cblas.h>
#include <mkl_spblas.h>
#include <mkl_rci.h>
#include <mkl_service.h>

enum ShearWallID 
{
	WallBot2 = 0,
	WallBot,
	WallTop,
	WallTop2,
};

const double rmin_rel = 0.0;

int gen_sphere(
	double asph, const double xcen[3], 
	const double usph[3], const double osph[3],
	int &np, double xp[], double up[],
	const double dh) 
{

	// estimate point number
	const int nb = (int) ceil(2.0/sqrt(3.0) * 4.0*M_PI / pow(dh/asph,2));

	//if (0) { // fibonacci 
	//	const double phi = (1.0 + sqrt(5.0)) / 2.0;

	//	const int nfib = nb;
	//	for (int i=-(nfib-1); i<=nfib-1; i+=2) {
	//		double theta = 2.0 * M_PI * i / phi;
	//		double sphi = (double) i / (double) nfib;
	//		double cphi = sqrt((double) ((nfib+i)*(nfib-i))) / nfib;

	//		double xsph = cphi * sin(theta);
	//		double ysph = cphi * cos(theta);
	//		double zsph = sphi;

	//		int nb = sf.nb;
	//		sf.xb[nb*3+0] = xsph * asph;
	//		sf.xb[nb*3+1] = ysph * asph;
	//		sf.xb[nb*3+2] = zsph * asph;
	//		sf.ub[nb*3+0] = 1.0;

	//		sf.nb += 1;
	//	}
	//}

	if (1) { // general spiral
		double theta = 0;

		for (int i=0; i<nb; i++) {

			double cphi = (double) (-1.0*(nb-1-i) + 1.0*i) / (double) (nb-1);
			double sphi = sqrt(1.0 - cphi*cphi);

			if (i==0 || i==nb-1) {
				theta = 0;
			} else {
				double coef = 3.6;
				theta += coef / (sphi * sqrt((double)nb));
				theta = fmod(theta, 2.0*M_PI);
			}

			// relative to sphere center
			double xx[3] = { asph * sphi * cos(theta), asph * sphi * sin(theta), asph * cphi };


			int ii = np;
			int ioff = ii * 3;

			for (int dir=0; dir<3; dir++) {
				xp[ioff+dir] = xcen[dir] + xx[dir];
				up[ioff+dir] = usph[dir];
			}

			up[ioff+0] += osph[1]*xx[2] - osph[2]*xx[1];
			up[ioff+1] += osph[2]*xx[0] - osph[0]*xx[2];
			up[ioff+2] += osph[0]*xx[1] - osph[1]*xx[0];

			np += 1;
		}
	}

	return nb;
}


int gen_plane_y(double yy, const double uplane[3],
	int &np, double xp[], double up[],
	const double xlo, const double ylo, const double zlo,
	const double xhi, const double yhi, const double zhi,	
	const double dh) 
{
	// estimate wall points
	int nbx = (int) ceil((xhi-xlo)/dh);
	int nbz = (int) ceil((zhi-zlo)/dh);
	double hbx = (xhi-xlo) / nbx;
	double hbz = (zhi-zlo) / nbz;

	int count = 0;

	for (int ii=0; ii<nbx; ii++) {
		for (int kk=0; kk<nbz; kk++) {
			double xx = xlo + hbx * ((double)ii + 0.5);
			double zz = zlo + hbz * ((double)kk + 0.5);

			int ind = np;
			xp[ind*3+0] = xx;
			xp[ind*3+1] = yy;
			xp[ind*3+2] = zz;
			up[ind*3+0] = uplane[0];
			up[ind*3+1] = uplane[1];
			up[ind*3+2] = uplane[2];

			np += 1;
		}
	}

	return count;
}



inline double func_ga(double r, double alpha) {
	double ra = r * alpha;
	double ra2 = ra * ra;

	double coef = alpha*alpha*alpha / pow(M_PI,1.5);
	double ga = coef * exp(-ra2) * (2.5-ra2);
	return ga;
}

inline void func_gl(double u[3], const double f[3],
	double rr, double xx, double yy, double zz, 
	double alpha, double xi, double mu) 
{
	double coef = 0.125 / (M_PI * mu);

	if (rr > 0) { // pair
		const double rr2 = rr * rr;
		const double er = xi * rr;
		const double ar = alpha * rr;

		double c1 = (myerf(er) - myerf(ar)) / rr;
		double c2 = 2.0/sqrt(M_PI) * (xi*exp(-er*er) - alpha*exp(-ar*ar));

		double ee[3][3] = { xx*xx, xx*yy, xx*zz, yy*xx, yy*yy, yy*zz, zz*xx, zz*yy, zz*zz };

		double ee1[3][3], ee2[3][3];
		for (int i=0; i<3; i++) {
			for (int j=0; j<3; j++) {
				ee1[i][j] = ee[i][j] / rr2;
				ee2[i][j] = -ee1[i][j];
			}
			ee1[i][i] += 1;
			ee2[i][i] += 1;
		}

		for (int i=0; i<3; i++) {
			u[i] = 0;
			for (int j=0; j<3; j++) {
				u[i] += (ee1[i][j]*c1 + ee2[i][j]*c2) * f[j];
			}
			u[i] *= coef;
		}

	} else { // self
		double dd = 4.0 * (xi-alpha) / sqrt(M_PI);
		for (int i=0; i<3; i++) {
			u[i] = coef * dd * f[i];
		}
	}
}

inline void func_glmat(double gl[3][3],
	double rr, double xx, double yy, double zz, 
	double alpha, double xi, double mu) 
{
	double coef = 0.125 / (M_PI * mu);

	if (rr > 0) { // pair
		const double rr2 = rr * rr;
		const double er = xi * rr;
		const double ar = alpha * rr;

		double c1 = (myerf(er) - myerf(ar)) / rr;
		double c2 = 2.0/sqrt(M_PI) * (xi*exp(-er*er) - alpha*exp(-ar*ar));

		double ee[3][3] = { xx*xx, xx*yy, xx*zz, yy*xx, yy*yy, yy*zz, zz*xx, zz*yy, zz*zz };

		double ee1[3][3], ee2[3][3];
		for (int i=0; i<3; i++) {
			for (int j=0; j<3; j++) {
				ee1[i][j] = ee[i][j] / rr2;
				ee2[i][j] = -ee1[i][j];
			}
			ee1[i][i] += 1;
			ee2[i][i] += 1;
		}

		for (int i=0; i<3; i++) {
			for (int j=0; j<3; j++) {
				gl[i][j] = ee1[i][j]*c1 + ee2[i][j]*c2;
				gl[i][j] *= coef;
			}
		}

	} else { // self, diagonal matrix
		double dd = 4.0 * (xi-alpha) / sqrt(M_PI);
		for (int i=0; i<3; i++) {
			for (int j=0; j<3; j++) {
				gl[i][j] = 0;
			}
			gl[i][i] = coef * dd;
		}
	}
}


static void stokes_map_grid(
	int &igrid, int &jgrid, int &kgrid,
	int i, int j, int k, int nx, int ny, int nz)
{
	igrid = i;
	if (igrid < 0) igrid += nx;
	if (igrid >= nx) igrid -= nx;
	
	jgrid = j;
	if (jgrid < 0) jgrid += ny;
	if (jgrid >= ny) jgrid -= ny;

	kgrid = k;
	if (kgrid < 0) kgrid += nz;
	if (kgrid >= nz) kgrid -= nz;
}
static int stokes_map_index(int i, int n)
{
	if (i < 0) i += n;
	if (i >= n) i -= n;
	return i;
}
static double stokes_map_dist(double xdiff, double xlen) {
	if (xdiff >= 0.5*xlen) xdiff -= xlen;
	if (xdiff < -0.5*xlen) xdiff += xlen;
	return xdiff;
}


static void stokes_zero_force(StokesFFT &stokes) {
	stokes_zero_buffer(stokes, stokes.fx);
	stokes_zero_buffer(stokes, stokes.fy);
	stokes_zero_buffer(stokes, stokes.fz);
}

static void stokes_set_grid(StokesFFT &stokes,
	double xlo, double ylo, double zlo, 
	double xhi, double yhi, double zhi,
	int nx, int ny, int nz) 
{
	stokes.problo[0] = xlo;
	stokes.problo[1] = ylo;
	stokes.problo[2] = zlo;
	stokes.probhi[0] = xhi;
	stokes.probhi[1] = yhi;
	stokes.probhi[2] = zhi;

	stokes.cellNum[0] = nx;
	stokes.cellNum[1] = ny;
	stokes.cellNum[2] = nz;
}






//
// Accumulate force
// Remember to clear force before calling this.
static void stokes_distrib_forcing(
	StokesForcing &sf, StokesFFT &stokes,
	int nf, const double *xf, const double *ff)
{
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	const double alpha = sf.alpha;
	const double rcut = sf.rcut;
	const int xspan = (int) (rcut/dx) + 1;
	const int yspan = (int) (rcut/dy) + 1;
	const int zspan = (int) (rcut/dz) + 1;

	for (int ib=0; ib<nf; ib++) {
		const int ioff = ib * 3;

		double px = xf[ioff+0];
		double py = xf[ioff+1];
		double pz = xf[ioff+2];
		double fx = ff[ioff+0];
		double fy = ff[ioff+1];
		double fz = ff[ioff+2];

		// 
		int icell = (int) floor((px - xlo) / dx);
		int jcell = (int) floor((py - ylo) / dy);
		int kcell = (int) floor((pz - zlo) / dz);

		for (int ii=icell-xspan; ii<=icell+1+xspan; ii++) {
		for (int jj=jcell-yspan; jj<=jcell+1+yspan; jj++) {
		for (int kk=kcell-zspan; kk<=kcell+1+zspan; kk++) {
			double xx = xlo + dx*ii;
			double yy = ylo + dy*jj;
			double zz = zlo + dz*kk;

			double xdist = xx - px;
			double ydist = yy - py;
			double zdist = zz - pz;
			double dist = sqrt(xdist*xdist + ydist*ydist + zdist*zdist);

			if (dist < rcut) {
				double ga = func_ga(dist, alpha);

				int igrid, jgrid, kgrid;
				stokes_map_grid(igrid, jgrid, kgrid, ii, jj, kk, nx, ny, nz);

				int idx = igrid*ny*nz + jgrid*nz + kgrid;
				stokes.fx[idx] += ga * fx;
				stokes.fy[idx] += ga * fy;
				stokes.fz[idx] += ga * fz;
			}
		}
		}
		}
	} // end distribute forcing
}

static void stokes_build_distrib_forcing_matrix(
	StokesForcing &sf, StokesFFT &stokes, 
	CSRMat &mat,
	int nf, const double *xf)
{
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	const double alpha = sf.alpha;
	const double rcut = sf.rcut;
	const int xspan = (int) (rcut/dx) + 1;
	const int yspan = (int) (rcut/dy) + 1;
	const int zspan = (int) (rcut/dz) + 1;

	//
	mat.clear();
	assert(mat.num_row() == nf);
	assert(mat.num_col() == nx*ny*nz);

	for (int ib=0; ib<nf; ib++) {
		const int ioff = ib * 3;

		double px = xf[ioff+0];
		double py = xf[ioff+1];
		double pz = xf[ioff+2];

		// use map to sort grid index
		std::map<int,double> save;
		double sum = 0;

		// locate particle-in-cell
		int icell = (int) floor((px - xlo) / dx);
		int jcell = (int) floor((py - ylo) / dy);
		int kcell = (int) floor((pz - zlo) / dz);

		// logical grid
		for (int ii=icell-xspan; ii<=icell+1+xspan; ii++) {
		for (int jj=jcell-yspan; jj<=jcell+1+yspan; jj++) {
		for (int kk=kcell-zspan; kk<=kcell+1+zspan; kk++) {
			// logical grid point
			double xx = xlo + dx*ii;
			double yy = ylo + dy*jj;
			double zz = zlo + dz*kk;

			double xdist = xx - px;
			double ydist = yy - py;
			double zdist = zz - pz;
			double dist = sqrt(xdist*xdist + ydist*ydist + zdist*zdist);

			if (dist < rcut) {
				double ga = func_ga(dist, alpha);

				int igrid, jgrid, kgrid;
				stokes_map_grid(igrid, jgrid, kgrid, ii, jj, kk, nx, ny, nz);

				int idx = igrid*ny*nz + jgrid*nz + kgrid;

				save[idx] = ga;
				sum += ga * (dx*dy*dz);
			}
		}
		}
		}

		//
		mat.begin_row(ib);
		for (std::map<int,double>::const_iterator it=save.begin(); it!=save.end(); ++it) {
			int idx = it->first;
			double val = it->second;
			//if (sum > 0) val /= sum;
			mat.push_data(ib, idx, val);
		}
		mat.end_row(ib);

	} // end distribute forcing
}

// accumulate
static void stokes_distrib_forcing_matrix(StokesForcing &sf, 
	CSRMat &mat, int np, const double *fin, double *gout) 
{
	StokesFFT &stokes = *sf.stokes;

	// NOTE matrix is a transpose
	assert(mat.num_row() == np);
	assert(mat.num_col() == stokes.cellNum[0]*stokes.cellNum[1]*stokes.cellNum[2]);

	double acoef = 1.0;
	double bcoef = 1.0;
	char trans = 'T';
	mat.gemv(acoef, fin, bcoef, gout, trans);

}

// interpolate global velocity
// NOTE accumulation
static void stokes_interp_velocity(
	StokesForcing &sf, StokesFFT &stokes,
	int np, const double *xp, double *uout)
{
	const double *ug = stokes.u;
	const double *vg = stokes.v;
	const double *wg = stokes.w;

	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	// interpolate global velocity
	for (int ib=0; ib<np; ib++) {
		const int ioff = ib * 3;
		
		double px = xp[ioff+0];
		double py = xp[ioff+1];
		double pz = xp[ioff+2];

		int i0 = (int) floor((px - xlo) / dx);
		int j0 = (int) floor((py - ylo) / dy);
		int k0 = (int) floor((pz - zlo) / dz);
		int i1 = i0 + 1;
		int j1 = j0 + 1;
		int k1 = k0 + 1;

		double x0 = xlo + dx*i0;
		double y0 = ylo + dy*j0;
		double z0 = zlo + dz*k0;

		double rx = (px-x0) / dx;
		double ry = (py-y0) / dy;
		double rz = (pz-z0) / dz;
		assert(0<=rx && rx<1);
		assert(0<=ry && ry<1);
		assert(0<=rz && rz<1);

		i0 = stokes_map_index(i0, nx);
		i1 = stokes_map_index(i1, nx);
		j0 = stokes_map_index(j0, ny);
		j1 = stokes_map_index(j1, ny);
		k0 = stokes_map_index(k0, nz);
		k1 = stokes_map_index(k1, nz);

		int ind0 = (i0)*ny*nz + (j0)*nz + (k0);
		int ind1 = (i0)*ny*nz + (j0)*nz + (k1);
		int ind2 = (i0)*ny*nz + (j1)*nz + (k0);
		int ind3 = (i0)*ny*nz + (j1)*nz + (k1);
		int ind4 = (i1)*ny*nz + (j0)*nz + (k0);
		int ind5 = (i1)*ny*nz + (j0)*nz + (k1);
		int ind6 = (i1)*ny*nz + (j1)*nz + (k0);
		int ind7 = (i1)*ny*nz + (j1)*nz + (k1);
		double w0 = (1.0-rx) * (1.0-ry) * (1.0-rz);
		double w1 = (1.0-rx) * (1.0-ry) * (rz    );
		double w2 = (1.0-rx) * (ry    ) * (1.0-rz);
		double w3 = (1.0-rx) * (ry    ) * (rz    );
		double w4 = (    rx) * (1.0-ry) * (1.0-rz);
		double w5 = (    rx) * (1.0-ry) * (rz    );
		double w6 = (    rx) * (ry    ) * (1.0-rz);
		double w7 = (    rx) * (ry    ) * (rz    );

		uout[ioff+0] += w0*ug[ind0] + w1*ug[ind1] + w2*ug[ind2] + w3*ug[ind3] + w4*ug[ind4] + w5*ug[ind5] + w6*ug[ind6] + w7*ug[ind7];
		uout[ioff+1] += w0*vg[ind0] + w1*vg[ind1] + w2*vg[ind2] + w3*vg[ind3] + w4*vg[ind4] + w5*vg[ind5] + w6*vg[ind6] + w7*vg[ind7];;
		uout[ioff+2] += w0*wg[ind0] + w1*wg[ind1] + w2*wg[ind2] + w3*wg[ind3] + w4*wg[ind4] + w5*wg[ind5] + w6*wg[ind6] + w7*wg[ind7];
	} // end interpolate global velocity
}

//
// Grid->Point interpolation matrix
static void stokes_build_interp_matrix(
	StokesForcing &sf, StokesFFT &stokes,
	CSRMat &mat, int np, const double *xp) 
{
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	mat.clear();
	assert(mat.num_row() == np);
	assert(mat.num_col() == nx*ny*nz);
	
	//const int interp_meth = 0;
	const int interp_meth = 1;

	// interpolate global velocity
	for (int ib=0; ib<np; ib++) {
		const int ioff = ib * 3;
		
		double px = xp[ioff+0];
		double py = xp[ioff+1];
		double pz = xp[ioff+2];

		//
		std::map<int,double> save;

		// locate grid
		const int iloc = (int) floor((px - xlo) / dx);
		const int jloc = (int) floor((py - ylo) / dy);
		const int kloc = (int) floor((pz - zlo) / dz);
		const double xloc = xlo + dx*iloc;
		const double yloc = ylo + dy*jloc;
		const double zloc = zlo + dz*kloc;

		// relative position in center grid
		const double rx = (px-xloc) / dx;
		const double ry = (py-yloc) / dy;
		const double rz = (pz-zloc) / dz;
		//assert(0<=rx && rx<1);
		//assert(0<=ry && ry<1);
		//assert(0<=rz && rz<1);

		if (interp_meth == 0) { // tri-linear
			int i0 = iloc;
			int j0 = jloc;
			int k0 = kloc;
			int i1 = i0 + 1;
			int j1 = j0 + 1;
			int k1 = k0 + 1;

			i0 = stokes_map_index(i0, nx);
			i1 = stokes_map_index(i1, nx);
			j0 = stokes_map_index(j0, ny);
			j1 = stokes_map_index(j1, ny);
			k0 = stokes_map_index(k0, nz);
			k1 = stokes_map_index(k1, nz);

			int ind0 = (i0)*ny*nz + (j0)*nz + (k0);
			int ind1 = (i0)*ny*nz + (j0)*nz + (k1);
			int ind2 = (i0)*ny*nz + (j1)*nz + (k0);
			int ind3 = (i0)*ny*nz + (j1)*nz + (k1);
			int ind4 = (i1)*ny*nz + (j0)*nz + (k0);
			int ind5 = (i1)*ny*nz + (j0)*nz + (k1);
			int ind6 = (i1)*ny*nz + (j1)*nz + (k0);
			int ind7 = (i1)*ny*nz + (j1)*nz + (k1);
			double w0 = (1.0-rx) * (1.0-ry) * (1.0-rz);
			double w1 = (1.0-rx) * (1.0-ry) * (rz    );
			double w2 = (1.0-rx) * (ry    ) * (1.0-rz);
			double w3 = (1.0-rx) * (ry    ) * (rz    );
			double w4 = (    rx) * (1.0-ry) * (1.0-rz);
			double w5 = (    rx) * (1.0-ry) * (rz    );
			double w6 = (    rx) * (ry    ) * (1.0-rz);
			double w7 = (    rx) * (ry    ) * (rz    );

			save[ind0] = w0;
			save[ind1] = w1;
			save[ind2] = w2;
			save[ind3] = w3;
			save[ind4] = w4;
			save[ind5] = w5;
			save[ind6] = w6;
			save[ind7] = w7;

		} else if (interp_meth == 1) { // tri-cubic
			const double pos[] = { rx,ry,rz };
			double coef[3][4] = { 0 };
			for (int dir=0; dir<3; dir++) {
				double x = pos[dir];
				coef[dir][0] = x * (x-1) * (x-2) / (-6.0);
				coef[dir][1] = (x+1) * (x-1) * (x-2) / (2.0);
				coef[dir][2] = (x+1) * x * (x-2) / (-2.0);
				coef[dir][3] = (x+1) * x * (x-1) / (6.0);
			}

			for (int ii=iloc-1; ii<=iloc+2; ii++) {
			for (int jj=jloc-1; jj<=jloc+2; jj++) {
			for (int kk=kloc-1; kk<=kloc+2; kk++) {
				// tensor product of cx*cy*cz
				double ww = coef[0][ii-iloc+1] * coef[1][jj-jloc+1] * coef[2][kk-kloc+1];

				int imap = stokes_map_index(ii, nx);
				int jmap = stokes_map_index(jj, ny);
				int kmap = stokes_map_index(kk, nz);

				int ind = imap*ny*nz + jmap*nz + kmap;

				save[ind] = ww;
			}
			}
			}

		}


		//
		// current matrix row
		//
		mat.begin_row(ib);
		for (std::map<int,double>::const_iterator it=save.begin(); it!=save.end(); ++it) {
			int ind = it->first;
			double val = it->second;
			mat.push_data(ib, ind, val);
		}
		mat.end_row(ib);
	} 
}

static void stokes_interp_velocity_matrix(
	StokesForcing &sf, StokesFFT &stokes,
	CSRMat &mat, int np, const double *xp, double *uout, const double *uin)
{
	assert(mat.num_row() == np);
	assert(mat.num_col() == stokes.cellNum[0]*stokes.cellNum[1]*stokes.cellNum[2]);

	double acoef = 1.0;
	double bcoef = 1.0;
	char trans = 'N';
	mat.gemv(acoef, uin, bcoef, uout, trans);
}

static void stokes_build_local_forcing_matrix(
	StokesForcing &sf,
	CSRMat &mat, const int np, const double *xp) 
{
	const int nall = np * 3;
	assert(mat.nrow == nall);
	assert(mat.ncol == nall);

	const double alpha = sf.alpha;
	const double xi = sf.xi;
	const double rcut = sf.rcut;

	const double rmin = (1.0/xi) * rmin_rel;

	StokesFFT stokes = *sf.stokes;
	const double mu = stokes.visc;
	const double *xlo = stokes.problo;
	const double *xhi = stokes.probhi;
	const double *xlen = stokes.problen;

	// prepare matrix
	mat.clear();
	mat.reserve(nall * 40);

	// add local velocity
	for (int ib=0; ib<np; ib++) {
		const int ioff = ib * 3;
		double ipx = xp[ioff+0];
		double ipy = xp[ioff+1];
		double ipz = xp[ioff+2];

		//
		std::vector<int> ineigh;
		std::vector<double> uneigh[3];
		{
			int nn = 64;
			ineigh.reserve(nn);
			for (int dir=0; dir<3; dir++) {
				uneigh[dir].reserve(nn*3);
			}
		}

		// loop, include itself
		for (int jb=0; jb<np; jb++) {
			const int joff = jb * 3;
			double jpx = xp[joff+0];
			double jpy = xp[joff+1];
			double jpz = xp[joff+2];

			double xdiff = ipx - jpx;
			double ydiff = ipy - jpy;
			double zdiff = ipz - jpz;

			// periodic wrap
			xdiff = stokes_map_dist(xdiff, xlen[0]);
			ydiff = stokes_map_dist(ydiff, xlen[1]);
			zdiff = stokes_map_dist(zdiff, xlen[2]);

			double dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
			if (dist<rmin && ib!=jb) {
				xdiff = xdiff / dist * rmin;
				ydiff = ydiff / dist * rmin;
				zdiff = zdiff / dist * rmin;
				dist = rmin;
			}

			if (dist < rcut) {
				double gl[3][3];
				func_glmat(gl, dist, xdiff, ydiff, zdiff, 
					alpha, xi, mu);

				ineigh.push_back(jb);
				for (int ii=0; ii<3; ii++) {
					for (int jj=0; jj<3; jj++) {
						uneigh[ii].push_back(gl[ii][jj]);
					}
				}
			}
		}

		// 
		for (int dir=0; dir<3; dir++) {
			int irow = ioff + dir;
			
			mat.begin_row(irow);

			for (int jneigh=0; jneigh<ineigh.size(); jneigh++) {
				int jb = ineigh[jneigh];
				int joff = jb * 3;

				mat.push_data(irow, joff+0, uneigh[dir][jneigh*3+0]);
				mat.push_data(irow, joff+1, uneigh[dir][jneigh*3+1]);
				mat.push_data(irow, joff+2, uneigh[dir][jneigh*3+2]);
			}

			mat.end_row(irow);
		}
	} // end add local
}


static void stokes_build_local_forcing_matrix2(
	StokesForcing &sf,
	CSRMat &mat, const int np, const double *xp,
	const double alpha, const double xi, const double rcut) 
{
	const int nall = np * 3;
	assert(mat.nrow == nall);
	assert(mat.ncol == nall);

	const double rmin = (1.0/xi) * rmin_rel;

	StokesFFT stokes = *sf.stokes;
	const double mu = stokes.visc;
	const double *xlo = stokes.problo;
	const double *xhi = stokes.probhi;
	const double *xlen = stokes.problen;

	const PointCache &cache = *sf.cache;
	assert(cache.hbin[0] >= rcut);
	assert(cache.hbin[1] >= rcut);
	assert(cache.hbin[2] >= rcut);

	// prepare matrix
	mat.clear();
	mat.reserve(nall * 40);

	// add local velocity
	for (int ib=0; ib<np; ib++) {
		const int ioff = ib * 3;
		double ipx = xp[ioff+0];
		double ipy = xp[ioff+1];
		double ipz = xp[ioff+2];

		// save matrix coef.
		std::map<int,std::vector<double> > mapneigh[3];
		std::vector<int> ineigh;
		std::vector<double> uneigh[3];
		{
			int nn = 64;
			ineigh.reserve(nn);
			for (int dir=0; dir<3; dir++) {
				uneigh[dir].reserve(nn*3);
			}
		}

		// get cache grid
		int igx = (int) floor((ipx-cache.xmin[0]) / cache.hbin[0]);
		int igy = (int) floor((ipy-cache.xmin[1]) / cache.hbin[1]);
		int igz = (int) floor((ipz-cache.xmin[2]) / cache.hbin[2]);
		igx = stokes_map_index(igx, cache.nbin[0]);
		igy = stokes_map_index(igy, cache.nbin[1]);
		igz = stokes_map_index(igz, cache.nbin[2]);

		//
		for (int ig=igx-1; ig<=igx+1; ig++) {
		for (int jg=igy-1; jg<=igy+1; jg++) {
		for (int kg=igz-1; kg<=igz+1; kg++) {
			int igrid = stokes_map_index(ig, cache.nbin[0]);
			int jgrid = stokes_map_index(jg, cache.nbin[1]);
			int kgrid = stokes_map_index(kg, cache.nbin[2]);
			int grididx = igrid*cache.nbin[1]*cache.nbin[2] + jgrid*cache.nbin[2] + kgrid;

			if (cache.tail[grididx] != -1) {
				for (int jb=cache.head[grididx]; /* no check */; jb=cache.next[jb]) {
					const int joff = jb * 3;
					double jpx = xp[joff+0];
					double jpy = xp[joff+1];
					double jpz = xp[joff+2];

					double xdiff = ipx - jpx;
					double ydiff = ipy - jpy;
					double zdiff = ipz - jpz;

					// periodic wrap
					xdiff = stokes_map_dist(xdiff, xlen[0]);
					ydiff = stokes_map_dist(ydiff, xlen[1]);
					zdiff = stokes_map_dist(zdiff, xlen[2]);

					double dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
					if (dist<rmin && ib!=jb) {
						xdiff = xdiff / dist * rmin;
						ydiff = ydiff / dist * rmin;
						zdiff = zdiff / dist * rmin;
						dist = rmin;
					}

					if (dist < rcut) {
						double gl[3][3];
						func_glmat(gl, dist, xdiff, ydiff, zdiff, 
							alpha, xi, mu);

						ineigh.push_back(jb);
						for (int ii=0; ii<3; ii++) {
							for (int jj=0; jj<3; jj++) {
								uneigh[ii].push_back(gl[ii][jj]);
							}
						}

						for (int ii=0; ii<3; ii++) {
							for (int jj=0; jj<3; jj++) {
								mapneigh[ii][jb].push_back(gl[ii][jj]);
							}
						}

					}


					// last entry
					if (jb == cache.tail[grididx]) break;
				}
			}
		}
		}
		}

		
		// 
		for (int dir=0; dir<3; dir++) {
			int irow = ioff + dir;
			
			mat.begin_row(irow);

			//for (int jneigh=0; jneigh<ineigh.size(); jneigh++) {
			//	int jb = ineigh[jneigh];
			//	int joff = jb * 3;

			//	mat.push_data(irow, joff+0, uneigh[dir][jneigh*3+0]);
			//	mat.push_data(irow, joff+1, uneigh[dir][jneigh*3+1]);
			//	mat.push_data(irow, joff+2, uneigh[dir][jneigh*3+2]);
			//}

			for (std::map<int,std::vector<double> >::const_iterator it=mapneigh[dir].begin(); it!=mapneigh[dir].end(); ++it) {
				int jb = it->first;
				int joff = jb*3;
				mat.push_data(irow, joff+0, it->second[0]);
				mat.push_data(irow, joff+1, it->second[1]);
				mat.push_data(irow, joff+2, it->second[2]);
			}

			mat.end_row(irow);
		}
	} // end add local
}




static void stokes_apply_local_forcing_matrix(StokesForcing &sf, 
	CSRMat &mat, const int np, const double *fin, double *uout)
{
	assert(np*3 == mat.nrow);
	assert(np*3 == mat.ncol);

	// local forcing matrix should be square
	mat.gemv(fin, uout);
	//mat.gemv(1.0, fin, 0.0, uout);
}

static void stokes_build_precond_matrix(
	StokesForcing &sf,
	CSRMat &mat, const int np, const double *xp) 
{
	const int nall = np * 3;
	assert(mat.nrow == nall);
	assert(mat.ncol == nall);

	//const double alpha = sf.alpha;
	const double alpha = 0;
	const double xi = sf.xi;
	const double rcut = sf.rcut;

	StokesFFT stokes = *sf.stokes;
	const double mu = stokes.visc;
	const double *xlo = stokes.problo;
	const double *xhi = stokes.probhi;
	const double *xlen = stokes.problen;

	// prepare matrix
	mat.clear();
	mat.reserve(nall * 40);

	// add local velocity
	for (int ib=0; ib<np; ib++) {
		const int ioff = ib * 3;
		double ipx = xp[ioff+0];
		double ipy = xp[ioff+1];
		double ipz = xp[ioff+2];

		//
		std::vector<int> ineigh;
		std::vector<double> uneigh[3];
		{
			int nn = 64;
			ineigh.reserve(nn);
			for (int dir=0; dir<3; dir++) {
				uneigh[dir].reserve(nn*3);
			}
		}

		// loop, include it self
		for (int jb=0; jb<np; jb++) {
			const int joff = jb * 3;
			double jpx = xp[joff+0];
			double jpy = xp[joff+1];
			double jpz = xp[joff+2];

			double xdiff = ipx - jpx;
			double ydiff = ipy - jpy;
			double zdiff = ipz - jpz;

			// periodic wrap
			xdiff = stokes_map_dist(xdiff, xlen[0]);
			ydiff = stokes_map_dist(ydiff, xlen[1]);
			zdiff = stokes_map_dist(zdiff, xlen[2]);

			double dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);

			if (dist < rcut) {
				double gl[3][3];
				func_glmat(gl, dist, xdiff, ydiff, zdiff, 
					alpha, xi, mu);

				ineigh.push_back(jb);
				for (int ii=0; ii<3; ii++) {
					for (int jj=0; jj<3; jj++) {
						uneigh[ii].push_back(gl[ii][jj]);
					}
				}
			}
		}

		// 
		for (int dir=0; dir<3; dir++) {
			int irow = ioff + dir;
			
			mat.begin_row(irow);

			for (int jneigh=0; jneigh<ineigh.size(); jneigh++) {
				int jb = ineigh[jneigh];
				int joff = jb * 3;

				mat.push_data(irow, joff+0, uneigh[dir][jneigh*3+0]);
				mat.push_data(irow, joff+1, uneigh[dir][jneigh*3+1]);
				mat.push_data(irow, joff+2, uneigh[dir][jneigh*3+2]);
			}

			mat.end_row(irow);
		}
	} // end add local
}

static void stokes_build_precond_matrix_all(
	StokesForcing &sf,
	CSRMat &mat) 
{
	const int nall = sf.nall;
	const int nb = sf.nb;
	const int np = sf.np;
	const int npart = sf.part_num;

	const int nall3 = nall * 3;
	const int nb3 = nb * 3;
	const int np3 = np * 3;
	const int npart6 = npart * 6;

	const int nsize = nb3 + np3 + npart6;
	assert(mat.num_row() == nsize);
	assert(mat.num_col() == nsize);

	const double *xall = sf.xall;
	const double *xpart = sf.part_pos;
	const double *apart = sf.part_rad;

	//const double alpha = sf.alpha;
	const double alpha = 0;
	const double xi = sf.xi;
	const double rcut = sf.rcut;

	const double rmin = (1.0/xi) * rmin_rel;

	StokesFFT stokes = *sf.stokes;
	const double mu = stokes.visc;
	const double *xlo = stokes.problo;
	const double *xhi = stokes.probhi;
	const double *xlen = stokes.problen;

	const PointCache &cache = *sf.cache;
	assert(cache.hbin[0] >= rcut);
	assert(cache.hbin[1] >= rcut);
	assert(cache.hbin[2] >= rcut);

	// prepare matrix
	mat.clear();
	mat.reserve(nsize * 60);

	//
	// point part
	//
	for (int ib=0; ib<nall; ib++) {
		const int ioff = ib * 3;
		double ipx = xall[ioff+0];
		double ipy = xall[ioff+1];
		double ipz = xall[ioff+2];

		//
		std::map<int,std::vector<double> > mapneigh[3];

		// get cache grid
		int igx = (int) floor((ipx-cache.xmin[0]) / cache.hbin[0]);
		int igy = (int) floor((ipy-cache.xmin[1]) / cache.hbin[1]);
		int igz = (int) floor((ipz-cache.xmin[2]) / cache.hbin[2]);
		igx = stokes_map_index(igx, cache.nbin[0]);
		igy = stokes_map_index(igy, cache.nbin[1]);
		igz = stokes_map_index(igz, cache.nbin[2]);

		// point-point, include it self
		//
		for (int ig=igx-1; ig<=igx+1; ig++) {
		for (int jg=igy-1; jg<=igy+1; jg++) {
		for (int kg=igz-1; kg<=igz+1; kg++) {
			int igrid = stokes_map_index(ig, cache.nbin[0]);
			int jgrid = stokes_map_index(jg, cache.nbin[1]);
			int kgrid = stokes_map_index(kg, cache.nbin[2]);
			int grididx = igrid*cache.nbin[1]*cache.nbin[2] + jgrid*cache.nbin[2] + kgrid;

			if (cache.tail[grididx] != -1) {
				for (int jb=cache.head[grididx]; /* no check */; jb=cache.next[jb]) {
					const int joff = jb * 3;
					double jpx = xall[joff+0];
					double jpy = xall[joff+1];
					double jpz = xall[joff+2];

					double xdiff = ipx - jpx;
					double ydiff = ipy - jpy;
					double zdiff = ipz - jpz;

					// periodic wrap
					xdiff = stokes_map_dist(xdiff, xlen[0]);
					ydiff = stokes_map_dist(ydiff, xlen[1]);
					zdiff = stokes_map_dist(zdiff, xlen[2]);

					double dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);
					if (dist<rmin && ib!=jb) {
						xdiff = xdiff / dist * rmin;
						ydiff = ydiff / dist * rmin;
						zdiff = zdiff / dist * rmin;
						dist = rmin;
					}

					if (dist < rcut) {
						double gl[3][3];
						func_glmat(gl, dist, xdiff, ydiff, zdiff, 
							alpha, xi, mu);

						for (int ii=0; ii<3; ii++) {
							for (int jj=0; jj<3; jj++) {
								mapneigh[ii][jb].push_back(gl[ii][jj]);
							}
						}
					}

					// last entry
					if (jb == cache.tail[grididx]) break;
				}
			}
		}
		}
		}

		int ipart = -1;
		double xprel, yprel, zprel;
		if (ib >= nb) { // point-particle
			ipart = sf.idall[ib];
			assert(0<=ipart && ipart<npart);

			xprel = ipx - xpart[ipart*3+0];
			yprel = ipy - xpart[ipart*3+1];
			zprel = ipz - xpart[ipart*3+2];
		}

		// 
		for (int dir=0; dir<3; dir++) {
			int irow = ioff + dir;
			
			mat.begin_row(irow);

			// point-point part
			for (std::map<int,std::vector<double> >::const_iterator it=mapneigh[dir].begin(); it!=mapneigh[dir].end(); ++it) {
				int jb = it->first;
				int joff = jb*3;
				mat.push_data(irow, joff+0, it->second[0]);
				mat.push_data(irow, joff+1, it->second[1]);
				mat.push_data(irow, joff+2, it->second[2]);
			}

			// point-particle part
			if (ib >= nb) {
				const int ipart6 = ipart*6 + nall3;
				const double sign = 1;
				if (dir == 0) {
					mat.push_data(irow, ipart6+0, -1.0 * sign);
					mat.push_data(irow, ipart6+4, -zprel * sign);
					mat.push_data(irow, ipart6+5, yprel * sign);
				} else if (dir == 1) {
					mat.push_data(irow, ipart6+1, -1.0 * sign);
					mat.push_data(irow, ipart6+3, zprel * sign);
					mat.push_data(irow, ipart6+5, -xprel * sign);
				} else {
					mat.push_data(irow, ipart6+2, -1.0 * sign);
					mat.push_data(irow, ipart6+3, -yprel * sign);
					mat.push_data(irow, ipart6+4, xprel * sign);
				}
			}

			mat.end_row(irow);
		}
	} // end add local

	//
	// particle part
	//
	for (int ipart=0; ipart<npart; ipart++) {
		const int ipart6 = ipart * 6;
		const int ioff = nall3 + ipart6;

		const double *ipos = &xpart[ipart*3];
		const double irad = apart[ipart];

		// particle-point
		if (1) {
			for (int dir=0; dir<6; dir++) {

				const int irow = nall3 + ipart6 + dir;

				mat.begin_row(irow);

				// loop points belong to this particle
				for (int ip=sf.part_ipbegin[ipart]; ip<sf.part_ipend[ipart]; ip++) {
					// reference distance
					double xr = sf.xp[ip*3+0] - xpart[ipart*3+0];
					double yr = sf.xp[ip*3+1] - xpart[ipart*3+1];
					double zr = sf.xp[ip*3+2] - xpart[ipart*3+2];

					if (dir < 3) {
						mat.push_data(irow, ip*3+dir+nb3, -1.0);
					} else {
						if (dir == 3) {
							mat.push_data(irow, ip*3+1+nb3, zr);
							mat.push_data(irow, ip*3+2+nb3, -yr);
						} else if (dir == 4) {
							mat.push_data(irow, ip*3+0+nb3, -zr);
							mat.push_data(irow, ip*3+2+nb3, xr);
						} else {
							mat.push_data(irow, ip*3+0+nb3, yr);
							mat.push_data(irow, ip*3+1+nb3, -xr);
						}
					}
				}

				// add a small diagonal
				double coef = SixPI * stokes.visc * sf.part_rad[ipart];
				mat.push_data(irow, irow, coef * 1.0e-3);

				mat.end_row(irow);
			}
		}

		// particle-particle
		if (0) {
			const double sgn = -1;

			std::map<int,double> mapneigh[6];

			for (int jpart=0; jpart<npart; jpart++) {
				const int jpart6 = jpart * 6;
				const int joff = nall3 + jpart6;

				const double *jpos = &xpart[jpart*3];
				const double jrad = apart[jpart];

				// i<-j interaction
				if (jpart == ipart) { // self
					for (int dir=0; dir<3; dir++) {
						mapneigh[dir][ioff+dir] += SixPI * mu * irad;
					}
					for (int dir=3; dir<6; dir++) {
						mapneigh[dir][ioff+dir] += EightPI * mu * (irad*irad*irad);
					}

				} else { // pair
					
					double xdiff[3]; // relative position
					for (int dir=0; dir<3; dir++) {
						xdiff[dir] = ipos[dir] - jpos[dir];
						xdiff[dir] = stokes_map_dist(xdiff[dir], xlen[dir]);
					}

					// distance
					double rij = sqrt(xdiff[0]*xdiff[0] + xdiff[1]*xdiff[1] + xdiff[2]*xdiff[2]);

					{
						const double rcut = (irad+jrad) * (1.0 + 0.5);
						if (rij >= rcut) continue;
					}

					xdiff[0] /= rij;
					xdiff[1] /= rij;
					xdiff[2] /= rij;

					double ee[3][3], ee1[3][3];
					for (int ii=0; ii<3; ii++) {
						for (int jj=0; jj<3; jj++) {
							ee[ii][jj] = xdiff[ii] * xdiff[jj];
							ee1[ii][jj] = 1.0 - ee[ii][jj];
						}
					}

					// limit minimal particle distance
					{
						const double rmin = (irad+jrad) * 1.001;
						if (rij < rmin) { rij = rmin; }
					}

					const double beta = jrad / irad;
					const double beta2 = beta * beta;
					const double beta3 = beta2 * beta;
					const double beta4 = beta3 * beta;
					const double hij = rij - (irad+jrad); assert(hij > 0);
					
					const double ah = irad / hij;
					//const double ah2 = ah * ah;
					const double logah = log(ah);

					// squeeze mode
					double asq = beta2/((beta+1.0)*(beta+1.0)) * ah;
					asq += (1.0+7.0*beta+beta2)/(5.0*pow(1.0+beta,3)) * logah;

					// shear mode 
					double ash = 4.0*beta*(2.0+beta+2.0*beta2)/(15.0*pow(beta+1.0,3)) * logah;
					ash += 4.0*(16.0-45.0*beta+58.0*beta2-45.0*beta3+16.0*beta4)/(375.0*pow(beta+1.0,4)) * ah * logah;

					// pump mode
					double apu = 0.1*beta*(4.0+beta)/pow(beta+1.0,2) * logah;
					apu += (32.0-33.0*beta+83.0*beta2+43.0*beta3)/(25.0*beta3) * ah * logah;

					//
					for (int ii=0; ii<3; ii++) {
						for (int jj=0; jj<3; jj++) {
							double cc = SixPI*mu*irad * asq * ee[ii][jj];
							cc += SixPI*mu*irad * ash * ee1[ii][jj];

							mapneigh[ii][ioff+jj] += cc;
							mapneigh[ii][joff+jj] -= cc;
						}
					}
					for (int ii=0; ii<3; ii++) {
						for (int jj=0; jj<3; jj++) {
							mapneigh[ii+3][ioff+3+jj] += EightPI*mu*(irad*irad*irad) * apu * ee1[ii][jj];
							mapneigh[ii+3][joff+3+jj] -= EightPI*mu*(irad*irad*irad) * apu * ee1[ii][jj];
						}
					}

				}
			}

			//
			for (int dir=0; dir<6; dir++) {
				const int irow = ioff + dir;

				mat.begin_row(irow);

				for (std::map<int,double>::const_iterator it=mapneigh[dir].begin(); it!=mapneigh[dir].end(); ++it) {
					mat.push_data(irow, it->first, it->second * sgn);
				}

				mat.end_row(irow);
			}

		} // end particle-particle

	}
}




static void stokes_solve_global(StokesForcing &sf, StokesFFT &stokes, 
	const int nf, const double *xf, const double *fin)
{
	// clear forcing
	stokes_zero_force(stokes);

	// distribute forcing
	if (sf.mat_distrib) {
		for (int dir=0; dir<3; dir++) {
			for (int i=0; i<nf; i++) {
				sf.tmp[dir][i] = fin[i*3+dir];
			}
		}
		stokes_distrib_forcing_matrix(sf, *sf.mat_distrib, nf, sf.tmp[0], stokes.fx);
		stokes_distrib_forcing_matrix(sf, *sf.mat_distrib, nf, sf.tmp[1], stokes.fy);
		stokes_distrib_forcing_matrix(sf, *sf.mat_distrib, nf, sf.tmp[2], stokes.fz);
	} else {
		stokes_distrib_forcing(sf, stokes, nf, xf, fin);
	}

	// solve global velocity
	stokes_solve(stokes);
}


void stokes_apply_forcing(StokesForcing &sf, const double *fin, double *uout) {

	//
	StokesFFT &stokes = *sf.stokes;
	const double mu = stokes.visc;
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	//
	const double xi = sf.xi;
	const double alpha = sf.alpha;
	const double rcut = sf.rcut;
	const int xspan = (int) (rcut/dx) + 1;
	const int yspan = (int) (rcut/dy) + 1;
	const int zspan = (int) (rcut/dz) + 1;

	const int nbndry = sf.nb;
	const double *xbndry = sf.xb;

	for (int i=0; i<nbndry*3; i++) {
		uout[i] = 0;
	}

	//
	// local velocity
	//
	if (sf.mat_local) {
		stokes_apply_local_forcing_matrix(sf, *sf.mat_local, nbndry, fin, uout);
	} else {
		for (int ib=0; ib<nbndry; ib++) {
			const int ioff = ib * 3;
			double ibx = xbndry[ioff+0];
			double iby = xbndry[ioff+1];
			double ibz = xbndry[ioff+2];

			uout[ioff+0] = 0;
			uout[ioff+1] = 0;
			uout[ioff+2] = 0;

			for (int jb=0; jb<nbndry; jb++) {
				const int joff = jb * 3;
				double jbx = xbndry[joff+0];
				double jby = xbndry[joff+1];
				double jbz = xbndry[joff+2];

				double xdiff = ibx - jbx;
				double ydiff = iby - jby;
				double zdiff = ibz - jbz;
				double dist = sqrt(xdiff*xdiff + ydiff*ydiff + zdiff*zdiff);

				if (dist < rcut) {
					double ul[3];
					func_gl(ul, &fin[joff], 
						dist, xdiff, ydiff, zdiff, 
						alpha, xi, mu);

					for (int dir=0; dir<3; dir++) {
						uout[ioff+dir] += ul[dir];
					}
				}
			}
		} // end add local
	}

	//
	// global velocity
	//

	// clear forcing
	stokes_zero_force(stokes);

	// distribute forcing
	if (sf.mat_distrib) {
		for (int dir=0; dir<3; dir++) {
			for (int i=0; i<nbndry; i++) {
				sf.tmp[dir][i] = fin[i*3+dir];
			}
		}
		stokes_distrib_forcing_matrix(sf, *sf.mat_distrib, nbndry, sf.tmp[0], stokes.fx);
		stokes_distrib_forcing_matrix(sf, *sf.mat_distrib, nbndry, sf.tmp[1], stokes.fy);
		stokes_distrib_forcing_matrix(sf, *sf.mat_distrib, nbndry, sf.tmp[2], stokes.fz);
	} else {
		stokes_distrib_forcing(sf, stokes, nbndry, xbndry, fin);
	}

	// solve global velocity
	stokes_solve(stokes);

	// interpolate global velocity
	if (sf.mat_interp) {
		for (int dir=0; dir<3; dir++) {
			for (int i=0; i<nbndry; i++) {
				sf.tmp[dir][i] = 0;
			}
		}
		stokes_interp_velocity_matrix(sf, stokes, *sf.mat_interp, nbndry, xbndry, sf.tmp[0], stokes.u);
		stokes_interp_velocity_matrix(sf, stokes, *sf.mat_interp, nbndry, xbndry, sf.tmp[1], stokes.v);
		stokes_interp_velocity_matrix(sf, stokes, *sf.mat_interp, nbndry, xbndry, sf.tmp[2], stokes.w);
		for (int dir=0; dir<3; dir++) {
			for (int i=0; i<nbndry; i++) {
				uout[i*3+dir] += sf.tmp[dir][i];
			}
		}
	} else {
		stokes_interp_velocity(sf, stokes, nbndry, xbndry, uout);
	}
	//for (int ib=0; ib<nbndry; ib++) {
	//	const int ioff = ib * 3;
	//	
	//	double px = xbndry[ioff+0];
	//	double py = xbndry[ioff+1];
	//	double pz = xbndry[ioff+2];

	//	int i0 = (int) floor((px - xlo) / dx);
	//	int j0 = (int) floor((py - ylo) / dy);
	//	int k0 = (int) floor((pz - zlo) / dz);
	//	int i1 = i0 + 1;
	//	int j1 = j0 + 1;
	//	int k1 = k0 + 1;

	//	double x0 = xlo + dx*i0;
	//	double y0 = ylo + dy*j0;
	//	double z0 = zlo + dz*k0;

	//	double rx = (px-x0) / dx;
	//	double ry = (py-y0) / dy;
	//	double rz = (pz-z0) / dz;
	//	assert(0<=rx && rx<1);
	//	assert(0<=ry && ry<1);
	//	assert(0<=rz && rz<1);

	//	i0 = stokes_map_index(i0, nx);
	//	i1 = stokes_map_index(i1, nx);
	//	j0 = stokes_map_index(j0, ny);
	//	j1 = stokes_map_index(j1, ny);
	//	k0 = stokes_map_index(k0, nz);
	//	k1 = stokes_map_index(k1, nz);

	//	int ind0 = (i0)*ny*nz + (j0)*nz + (k0);
	//	int ind1 = (i0)*ny*nz + (j0)*nz + (k1);
	//	int ind2 = (i0)*ny*nz + (j1)*nz + (k0);
	//	int ind3 = (i0)*ny*nz + (j1)*nz + (k1);
	//	int ind4 = (i1)*ny*nz + (j0)*nz + (k0);
	//	int ind5 = (i1)*ny*nz + (j0)*nz + (k1);
	//	int ind6 = (i1)*ny*nz + (j1)*nz + (k0);
	//	int ind7 = (i1)*ny*nz + (j1)*nz + (k1);
	//	double w0 = (1.0-rx) * (1.0-ry) * (1.0-rz);
	//	double w1 = (1.0-rx) * (1.0-ry) * (rz    );
	//	double w2 = (1.0-rx) * (ry    ) * (1.0-rz);
	//	double w3 = (1.0-rx) * (ry    ) * (rz    );
	//	double w4 = (    rx) * (1.0-ry) * (1.0-rz);
	//	double w5 = (    rx) * (1.0-ry) * (rz    );
	//	double w6 = (    rx) * (ry    ) * (1.0-rz);
	//	double w7 = (    rx) * (ry    ) * (rz    );

	//	const double *ug = stokes.u;
	//	const double *vg = stokes.v;
	//	const double *wg = stokes.w;

	//	uout[ioff+0] += w0*ug[ind0] + w1*ug[ind1] + w2*ug[ind2] + w3*ug[ind3] + w4*ug[ind4] + w5*ug[ind5] + w6*ug[ind6] + w7*ug[ind7];
	//	uout[ioff+1] += w0*vg[ind0] + w1*vg[ind1] + w2*vg[ind2] + w3*vg[ind3] + w4*vg[ind4] + w5*vg[ind5] + w6*vg[ind6] + w7*vg[ind7];;
	//	uout[ioff+2] += w0*wg[ind0] + w1*wg[ind1] + w2*wg[ind2] + w3*wg[ind3] + w4*wg[ind4] + w5*wg[ind5] + w6*wg[ind6] + w7*wg[ind7];
	//} // end interpolate global velocity



}


void stokes_apply_forcing2(StokesForcing &sf, const double *fin, double *uout) {

	//
	StokesFFT &stokes = *sf.stokes;
	const double mu = stokes.visc;
	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	//
	const double xi = sf.xi;
	const double alpha = sf.alpha;
	const double rcut = sf.rcut;
	const int xspan = (int) (rcut/dx) + 1;
	const int yspan = (int) (rcut/dy) + 1;
	const int zspan = (int) (rcut/dz) + 1;

	const int nbndry = sf.nall;
	const double *xbndry = sf.xall;

	for (int i=0; i<nbndry*3; i++) {
		uout[i] = 0;
	}


	//
	// local velocity
	//
	if (sf.mat_local) {
		stokes_apply_local_forcing_matrix(sf, *sf.mat_local, nbndry, fin, uout);
	} else {
		std::cerr << __FUNCTION__ << ": no local matrix" << std::endl;
		exit(1);
	}

	//
	// global velocity
	//
	stokes_solve_global(sf, stokes, nbndry, xbndry, fin);

	//
	// interpolate global velocity
	//
	if (sf.mat_interp) {
		for (int dir=0; dir<3; dir++) {
			for (int i=0; i<nbndry; i++) {
				sf.tmp[dir][i] = 0;
			}
		}
		stokes_interp_velocity_matrix(sf, stokes, *sf.mat_interp, nbndry, xbndry, sf.tmp[0], stokes.u);
		stokes_interp_velocity_matrix(sf, stokes, *sf.mat_interp, nbndry, xbndry, sf.tmp[1], stokes.v);
		stokes_interp_velocity_matrix(sf, stokes, *sf.mat_interp, nbndry, xbndry, sf.tmp[2], stokes.w);
		for (int dir=0; dir<3; dir++) {
			for (int i=0; i<nbndry; i++) {
				uout[i*3+dir] += sf.tmp[dir][i];
			}
		}
	} else {
		stokes_interp_velocity(sf, stokes, nbndry, xbndry, uout);
	}
}






void test_stokes_force(int argc, char *argv[]) {
	
	//
	StokesFFT stokes;

	// density & viscosity
	const double rho = 1.0;
	//const double mu = 1.0;
	// to have 6*pi*mu = 1.0
	const double mu = 1.0/(6.0*M_PI);

	stokes.dens = rho;
	stokes.visc = mu;

	// domain
	const double lenhalf = M_PI;
	const double xlo = -lenhalf, xhi = lenhalf;
	const double ylo = -lenhalf, yhi = lenhalf;
	const double zlo = -lenhalf, zhi = lenhalf;
	
	// grid
	const int ncell = 64;
	const int nx = ncell;
	const int ny = ncell;
	const int nz = ncell;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;
	const double dz = (zhi-zlo) / nz;


	//
	stokes_set_grid(stokes, xlo, ylo, zlo, xhi, yhi, zhi, nx, ny, nz);
	stokes_init(stokes);
	stokes_init_fft(stokes);

	// point force
	const int nf = 1;
	double xf[nf][3];
	double ff[nf][3];
	{
		xf[0][0] = (xlo+xhi) / 2; xf[0][1] = (ylo+yhi) / 2; xf[0][2] = (zlo+zhi) / 2;
		ff[0][0] = 1.0; ff[0][1] = 0.0; ff[0][2] = 0.0;
	}

	// regular factor
	const double xi = 2.0 / dx;
	// screen factor
	const double alpha = 0.8 / dx;
	// cutoff distance
	const double rcut = 4.0 / alpha;
	const int xspan = (int) (rcut/dx) + 1;
	const int yspan = (int) (rcut/dy) + 1;
	const int zspan = (int) (rcut/dz) + 1;

	// distribute global force
	stokes_zero_buffer(stokes, stokes.fx);
	stokes_zero_buffer(stokes, stokes.fy);
	stokes_zero_buffer(stokes, stokes.fz);

	if (1) {
	for (int ipf=0; ipf<nf; ipf++) {
		// 
		int icell = (int) floor((xf[ipf][0] - xlo) / dx);
		int jcell = (int) floor((xf[ipf][1] - ylo) / dy);
		int kcell = (int) floor((xf[ipf][2] - zlo) / dz);

		for (int ii=icell-xspan; ii<=icell+1+xspan; ii++) {
		for (int jj=jcell-yspan; jj<=jcell+1+yspan; jj++) {
		for (int kk=kcell-zspan; kk<=kcell+1+zspan; kk++) {
			double xx = xlo + dx*ii;
			double yy = ylo + dy*jj;
			double zz = zlo + dz*kk;
			double xdist = xx - xf[ipf][0];
			double ydist = yy - xf[ipf][1];
			double zdist = zz - xf[ipf][2];
			double dist = sqrt(xdist*xdist + ydist*ydist + zdist*zdist);

			if (dist < rcut) {
				double ga = func_ga(dist, alpha);
				
				int igrid, jgrid, kgrid;
				stokes_map_grid(igrid, jgrid, kgrid, ii, jj, kk, nx, ny, nz);

				int idx = igrid*ny*nz + jgrid*nz + kgrid;
				stokes.fx[idx] += ga * ff[ipf][0];
				stokes.fy[idx] += ga * ff[ipf][1];
				stokes.fz[idx] += ga * ff[ipf][2];
			}
		}
		}
		}
	}
	}

	if (0) {
		int idx = (nx/2)*ny*nz + (ny/2)*nz + (nz/2);
		stokes.fx[idx] = 1.0 / (dx*dy*dz);
		stokes.fy[idx] = 0.0;
		stokes.fz[idx] = 0.0;
	}


	// solve global velocity
	stokes_solve(stokes);

	if (1) {
	// add local velocity
	for (int ipf=0; ipf<nf; ipf++) {
		// 
		int icell = (int) floor((xf[ipf][0] - xlo) / dx);
		int jcell = (int) floor((xf[ipf][1] - ylo) / dy);
		int kcell = (int) floor((xf[ipf][2] - zlo) / dz);

		for (int ii=icell-xspan; ii<=icell+1+xspan; ii++) {
		for (int jj=jcell-yspan; jj<=jcell+1+yspan; jj++) {
		for (int kk=kcell-zspan; kk<=kcell+1+zspan; kk++) {
			double xx = xlo + dx*ii;
			double yy = ylo + dy*jj;
			double zz = zlo + dz*kk;
			double xdist = xx - xf[ipf][0];
			double ydist = yy - xf[ipf][1];
			double zdist = zz - xf[ipf][2];
			double dist = sqrt(xdist*xdist + ydist*ydist + zdist*zdist);

			if (dist < rcut) {
				double ul[3];
				func_gl(ul, ff[ipf], 
					dist, xdist, ydist, zdist, 
					alpha, xi, mu);
				double ga = func_ga(dist, alpha);
				
				int igrid, jgrid, kgrid;
				stokes_map_grid(igrid, jgrid, kgrid, ii, jj, kk, nx, ny, nz);

				int idx = igrid*ny*nz + jgrid*nz + kgrid;
				stokes.u[idx] += ul[0];
				stokes.v[idx] += ul[1];
				stokes.w[idx] += ul[2];
			}
		}
		}
		}
	}
	}


	if (0) {
		double umin = 1.0e30;
		for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
		for (int k=0; k<nz; k++) {
			int idx = i*ny*nz + j*nz + k;
			umin = std::min(umin, stokes.u[idx]);
		}
		}
		}
		for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
		for (int k=0; k<nz; k++) {
			int idx = i*ny*nz + j*nz + k;
			stokes.u[idx] -= umin;
		}
		}
		}
	}

	stokes_output_vtk(stokes, "test_force_3d00.vtk");
}




void test_stokes_bndry(int argc, char *argv[]) {
	
	//
	StokesFFT stokes;

	// to have 6*pi*mu = 1.0
	//const double mu = 1.0/(6.0*M_PI);
	const double mu = 1.0;

	stokes.dens = 1.0;
	stokes.visc = mu;

	// domain
	const double len = 1.0;
	const double xlo = -len/2, xhi = len/2;
	const double zlo = -len/2, zhi = len/2;
	//const double ylo = -lenhalf, yhi = lenhalf;
	const double ylo = -len*2/2, yhi = len*2/2;
	
	// grid
	const int ncell = 64;
	const int nx = ncell;
	const int ny = ncell*2;
	const int nz = ncell;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;
	const double dz = (zhi-zlo) / nz;

	//
	stokes_set_grid(stokes, xlo, ylo, zlo, xhi, yhi, zhi, nx, ny, nz);
	stokes_init(stokes);
	stokes_init_fft(stokes);

	//
	StokesForcing sf;
	sf.stokes = &stokes;

	// regular factor
	const double xi = 1.3 / dx;
	// screen factor
	const double alpha = 0.8 / dx;
	// cutoff distance
	const double rcut = 4.0 / alpha;

	sf.xi = xi;
	sf.alpha = alpha;
	sf.rcut = rcut;

	sf.nb = 0;
	const int nbmax = 50000;
	sf.xb = new double[nbmax*3];
	sf.fb = new double[nbmax*3];
	sf.ub = new double[nbmax*3];
	sf.tmp[0] = new double[nbmax];
	sf.tmp[1] = new double[nbmax];
	sf.tmp[2] = new double[nbmax];
	for (int i=0; i<nbmax; i++) {
		for (int dir=0; dir<3; dir++) {
			sf.xb[i*3+dir] = 0;
			sf.fb[i*3+dir] = 0;
			sf.ub[i*3+dir] = 0;
			sf.tmp[dir][i] = 0;
		}
	}

	{ // set boundary points
		const double dh = dx * 2.1; // typical distance between boundary point
		const double gammadot = 1.0; // shear rate

		if (1) { // add sphere
			const double dsph = 0.95;
			const double asph = dsph/2;
			const double xsph[3] = { 0.0, 0.0, 0.0 };
			const double usph[3] = { 0.0, 0.0, 0.0 };
			const double osph[3] = { 0.0, 0.0, -gammadot/2 };
			gen_sphere(asph, xsph, usph, osph,
				sf.nb, sf.xb, sf.ub, dh);
		}

		if (1) { // add top/bottom wall
			const double ythick = len / 4;
			const double ytop = len / 2;
			const double utop[3] = { ytop*gammadot, 0.0, 0.0 };
			const double ybot = -len / 2;
			const double ubot[3] = { ybot*gammadot, 0.0, 0.0 };

			gen_plane_y(ybot-ythick, ubot, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			gen_plane_y(ybot, ubot, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			gen_plane_y(ytop, utop, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			gen_plane_y(ytop+ythick, utop, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
		}

		if (sf.nb > nbmax) {
			std::cerr << "Nb overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Nb=" << sf.nb << std::endl;
		}
	}

	if (1) {
		std::cout << "Build local forcing matrix" << std::endl;
		
		int nall = sf.nb * 3;

		sf.mat_local = new CSRMat(nall, nall);

		stokes_build_local_forcing_matrix(sf, *sf.mat_local, sf.nb, sf.xb);
	}
	
	if (1) {
		std::cout << "Build precondition matrix" << std::endl;
		
		int nall = sf.nb * 3;

		sf.mat_precond = new CSRMat(nall, nall);

		stokes_build_precond_matrix(sf, *sf.mat_precond, sf.nb, sf.xb);
	} else {
		sf.mat_precond = NULL;
	}

	if (1) {
		std::cout << "Build distribution matrix" << std::endl;
		int nrow = sf.nb;
		int ncol = nx * ny * nz;
		sf.mat_distrib = new CSRMat(nrow, ncol);
		stokes_build_distrib_forcing_matrix(sf, stokes, *sf.mat_distrib, sf.nb, sf.xb);
	} else {
		sf.mat_distrib = NULL;
	}

	if (1) {
		std::cout << "Build interpolation matrix" << std::endl;
		int nrow = sf.nb;
		int ncol = nx * ny * nz;
		sf.mat_interp = new CSRMat(nrow,ncol);
		stokes_build_interp_matrix(sf, stokes, *sf.mat_interp, sf.nb, sf.xb);
	} else {
		sf.mat_interp = NULL;
	}

	// solve boundary force
	{
		std::cout << "Solve forcing" << std::endl;

		const int nunk = sf.nb * 3;

		//
		const int use_precond = 1 && (sf.mat_precond!=NULL);
		const double eps_rel = 1.0e-8;
		const double eps_abs = 0.0;

		const int npar = 128;
		MKL_INT ipar[npar] = { 0 };
		double dpar[npar] = { 0 };

		//
		double *tmp = new double[nunk*(2*nunk+1) + (nunk*(nunk+9))/2 + 1];
		double *rhs = new double[nunk];
		double *sol = sf.fb;
		for (int i=0; i<nunk; i++) {
			rhs[i] = sf.ub[i];
			sol[i] = 0;
		}

		double *bilu0 = NULL;
		double *trvec = NULL;
		if (sf.mat_precond && use_precond) {
			bilu0 = new double[sf.mat_precond->ndata];
			trvec = new double[nunk];
		}

		MKL_INT rci = 0;
		MKL_INT ivar = 0;
		MKL_INT ierr = 0;
		MKL_INT itercount = 0;

		ivar = nunk;

		// init GMRES
		dfgmres_init(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		//if (rci != 0) goto failed;
		if (rci != 0) {
			std::cerr << __FUNCTION__ << ": INIT GMRES failed rci=" << rci << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT GMRES rci=" << rci << std::endl;
		}

		if (use_precond) {
		// init precond
		dcsrilu0(&ivar, sf.mat_precond->data(), sf.mat_precond->ia(), sf.mat_precond->ja(), bilu0, ipar, dpar, &ierr);
		if (ierr != 0) {
			std::cerr << __FUNCTION__ << ": INIT ILU0 failed err=" << ierr << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT ILU0 err=" << ierr << std::endl;
		}
		}

		ipar[8] = 1; // do residual stopping test
		ipar[9] = 0; // no user stopping test
		ipar[10] = use_precond; // no precond
		ipar[11] = 1; // auto test for breakdown
		ipar[14] = 10; // restart
		//
		dpar[0] = eps_rel; // relative error
		dpar[1] = eps_abs; // absolute error

		//
		dfgmres_check(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		//if (rci != 0) goto failed;
		if (rci != 0) {
			std::cerr << __FUNCTION__ << ": CHECK failed rci=" << rci << std::endl;
			exit(1);
		}

		while (1) {
			// loop
			dfgmres(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
			//std::cout << rci << std::endl;

			if (rci == 0) {
				// converged
				break;
			} else if (rci == 1) {
				// compute A*x
				double *fin = &tmp[ipar[21]-1];
				double *uout = &tmp[ipar[22]-1];
				stokes_apply_forcing(sf, fin, uout);
			} 
			else if (rci==3 && use_precond) {
				int *ia = sf.mat_precond->ia();
				int *ja = sf.mat_precond->ja();

				char cvar1, cvar, cvar2;
				cvar1 = 'L'; 
				cvar = 'N';
				cvar2 = 'U';
				mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, &tmp[ipar[21]-1], trvec);
				cvar1 = 'U'; 
				cvar = 'N';
				cvar2 = 'N';
				mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, trvec, &tmp[ipar[22]-1]);
			} 
			else {
				// failed
				break;
			}
		}

		if (rci == 0) {
			// retrieve solution
			dfgmres_get(&ivar, sol, rhs, &rci, ipar, dpar, tmp, &itercount);
			std::cout << "Solved: " << "iter=" << itercount << std::endl;
		} else {
			std::cerr << "Error: " << "rci=" << rci << std::endl;
		}

		// cleanup
		MKL_Free_Buffers();
	} // end solve boundary force

	if (1) { // check velocity is exactly recovered
		double *ubsol = new double[sf.nb*3];
		stokes_apply_forcing(sf, sf.fb, ubsol);

		double uerr = 0;
		for (int ib=0; ib<sf.nb; ib++) {
			const int ioff = ib * 3;
			double iberr = 0;
			for (int dir=0; dir<3; dir++) {
				iberr += pow(ubsol[ioff+dir]-sf.ub[ioff+dir], 2);
			}
			iberr = sqrt(iberr);
			uerr = std::max(uerr, iberr);
		}

		std::cout << "|u-ub|=" << uerr << std::endl;

		delete[] ubsol;
	}

	if (1) { // calculate velocity on grid for visualization
		// note that we already have the global velocity from the last solver iteration
		// so add local velocity only

		double *ug = stokes.u;
		double *vg = stokes.v;
		double *wg = stokes.w;

		const int np = sf.nb;
		const double *xp = sf.xb;
		const double *fp = sf.fb;

		const int xspan = (int) (rcut/dx) + 1;
		const int yspan = (int) (rcut/dy) + 1;
		const int zspan = (int) (rcut/dz) + 1;

		for (int ip=0; ip<np; ip++) {
			const int ioff = ip * 3;
			
			const double *x = &xp[ioff];
			const double *f = &fp[ioff];

			// 
			int icell = (int) floor((x[0] - xlo) / dx);
			int jcell = (int) floor((x[1] - ylo) / dy);
			int kcell = (int) floor((x[2] - zlo) / dz);

			for (int ii=icell-xspan; ii<=icell+1+xspan; ii++) {
			for (int jj=jcell-yspan; jj<=jcell+1+yspan; jj++) {
			for (int kk=kcell-zspan; kk<=kcell+1+zspan; kk++) {
				double xx = xlo + dx*ii;
				double yy = ylo + dy*jj;
				double zz = zlo + dz*kk;
				double xdist = xx - x[0];
				double ydist = yy - x[1];
				double zdist = zz - x[2];
				double dist = sqrt(xdist*xdist + ydist*ydist + zdist*zdist);

				if (dist < rcut) {
					double ul[3];
					func_gl(ul, f, 
						dist, xdist, ydist, zdist, 
						alpha, xi, mu);

					int igrid, jgrid, kgrid;
					stokes_map_grid(igrid, jgrid, kgrid, ii, jj, kk, nx, ny, nz);

					int idx = igrid*ny*nz + jgrid*nz + kgrid;
					ug[idx] += ul[0];
					vg[idx] += ul[1];
					wg[idx] += ul[2];
				}
			}
			}
			}
		} // end loop forcing
	}

	stokes_output_vtk(stokes, "test_bndry00.vtk");
	stokes_forcing_output_csv(sf, "test_bndry00.csv");
}

void test_stokes_drag(int argc, char *argv[]) {
	
	//
	StokesFFT stokes;

	// to have 6*pi*mu = 1.0
	const double mu = 1.0/(6.0*M_PI);
	//const double mu = 1.0;

	stokes.dens = 1.0;
	stokes.visc = mu;

	// domain
	const double len = 1.0;
	const double xlo = -len/2, xhi = len/2;
	const double zlo = -len/2, zhi = len/2;
	const double ylo = -len/2, yhi = len/2;
	
	// grid
	const int ncell = 80;
	const int nx = ncell;
	const int ny = ncell;
	const int nz = ncell;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;
	const double dz = (zhi-zlo) / nz;

	//
	stokes_set_grid(stokes, xlo, ylo, zlo, xhi, yhi, zhi, nx, ny, nz);
	stokes_init(stokes);
	stokes_init_fft(stokes);

	//
	StokesForcing sf;
	sf.stokes = &stokes;

	// regular factor
	const double xi = 1.3 / dx;
	// screen factor
	const double alpha = 0.8 / dx;
	// cutoff distance
	const double rcut = 4.0 / alpha;

	sf.xi = xi;
	sf.alpha = alpha;
	sf.rcut = rcut;

	sf.nb = 0;
	const int nbmax = 50000;
	sf.xb = new double[nbmax*3];
	sf.fb = new double[nbmax*3];
	sf.ub = new double[nbmax*3];
	sf.tmp[0] = new double[nbmax];
	sf.tmp[1] = new double[nbmax];
	sf.tmp[2] = new double[nbmax];
	for (int i=0; i<nbmax; i++) {
		for (int dir=0; dir<3; dir++) {
			sf.xb[i*3+dir] = 0;
			sf.fb[i*3+dir] = 0;
			sf.ub[i*3+dir] = 0;
			sf.tmp[dir][i] = 0;
		}
	}

	{ // set boundary points
		const double dh = dx * 2.1; // typical distance between boundary point

		if (1) { // add sphere
			//const double cfrac = 0.000125;
			//const double cfrac = 0.008;
			//const double cfrac = 0.027;
			//const double cfrac = 0.064;
			const double cfrac = 0.125;
			const double dsph = pow(cfrac*(xhi-xlo)*(yhi-ylo)*(zhi-zlo)*6.0/PI, 1.0/3.0);
			const double asph = dsph/2;
			const double xsph[3] = { 0.0, 0.0, 0.0 };
			const double usph[3] = { 1.0, 0.0, 0.0 };
			const double osph[3] = { 0.0, 0.0, 0.0 };
			gen_sphere(asph, xsph, usph, osph,
				sf.nb, sf.xb, sf.ub, dh);
		}

		if (sf.nb > nbmax) {
			std::cerr << "Nb overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Nb=" << sf.nb << std::endl;
		}
	}

	if (1) {
		std::cout << "Build local forcing matrix" << std::endl;
		
		int nall = sf.nb * 3;

		sf.mat_local = new CSRMat(nall, nall);

		stokes_build_local_forcing_matrix(sf, *sf.mat_local, sf.nb, sf.xb);
	}
	
	if (1) {
		std::cout << "Build precondition matrix" << std::endl;
		
		int nall = sf.nb * 3;

		sf.mat_precond = new CSRMat(nall, nall);

		stokes_build_precond_matrix(sf, *sf.mat_precond, sf.nb, sf.xb);
	} else {
		sf.mat_precond = NULL;
	}

	if (1) {
		std::cout << "Build distribution matrix" << std::endl;
		int nrow = sf.nb;
		int ncol = nx * ny * nz;
		sf.mat_distrib = new CSRMat(nrow, ncol);
		stokes_build_distrib_forcing_matrix(sf, stokes, *sf.mat_distrib, sf.nb, sf.xb);
	} else {
		sf.mat_distrib = NULL;
	}

	if (1) {
		std::cout << "Build interpolation matrix" << std::endl;
		int nrow = sf.nb;
		int ncol = nx * ny * nz;
		sf.mat_interp = new CSRMat(nrow,ncol);
		stokes_build_interp_matrix(sf, stokes, *sf.mat_interp, sf.nb, sf.xb);
	} else {
		sf.mat_interp = NULL;
	}

	// solve boundary force
	{
		std::cout << "Solve forcing" << std::endl;

		const int nunk = sf.nb * 3;

		//
		const int use_precond = 1 && (sf.mat_precond!=NULL);
		const double eps_rel = 1.0e-8;
		const double eps_abs = 0.0;

		const int npar = 128;
		MKL_INT ipar[npar] = { 0 };
		double dpar[npar] = { 0 };

		//
		double *tmp = new double[nunk*(2*nunk+1) + (nunk*(nunk+9))/2 + 1];
		double *rhs = new double[nunk];
		double *sol = sf.fb;
		for (int i=0; i<nunk; i++) {
			rhs[i] = sf.ub[i];
			sol[i] = 0;
		}

		double *bilu0 = NULL;
		double *trvec = NULL;
		if (sf.mat_precond && use_precond) {
			bilu0 = new double[sf.mat_precond->ndata];
			trvec = new double[nunk];
		}

		MKL_INT rci = 0;
		MKL_INT ivar = 0;
		MKL_INT ierr = 0;
		MKL_INT itercount = 0;

		ivar = nunk;

		// init GMRES
		dfgmres_init(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		//if (rci != 0) goto failed;
		if (rci != 0) {
			std::cerr << __FUNCTION__ << ": INIT GMRES failed rci=" << rci << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT GMRES rci=" << rci << std::endl;
		}

		if (use_precond) {
		// init precond
		dcsrilu0(&ivar, sf.mat_precond->data(), sf.mat_precond->ia(), sf.mat_precond->ja(), bilu0, ipar, dpar, &ierr);
		if (ierr != 0) {
			std::cerr << __FUNCTION__ << ": INIT ILU0 failed err=" << ierr << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT ILU0 err=" << ierr << std::endl;
		}
		}

		ipar[8] = 1; // do residual stopping test
		ipar[9] = 0; // no user stopping test
		ipar[10] = use_precond; // no precond
		ipar[11] = 1; // auto test for breakdown
		ipar[14] = 10; // restart
		//
		dpar[0] = eps_rel; // relative error
		dpar[1] = eps_abs; // absolute error

		//
		dfgmres_check(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		//if (rci != 0) goto failed;
		if (rci != 0) {
			std::cerr << __FUNCTION__ << ": CHECK failed rci=" << rci << std::endl;
			exit(1);
		}

		while (1) {
			// loop
			dfgmres(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
			//std::cout << rci << std::endl;

			if (rci == 0) {
				// converged
				break;
			} else if (rci == 1) {
				// compute A*x
				double *fin = &tmp[ipar[21]-1];
				double *uout = &tmp[ipar[22]-1];
				stokes_apply_forcing(sf, fin, uout);
			} 
			else if (rci==3 && use_precond) {
				int *ia = sf.mat_precond->ia();
				int *ja = sf.mat_precond->ja();

				char cvar1, cvar, cvar2;
				cvar1 = 'L'; 
				cvar = 'N';
				cvar2 = 'U';
				mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, &tmp[ipar[21]-1], trvec);
				cvar1 = 'U'; 
				cvar = 'N';
				cvar2 = 'N';
				mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, trvec, &tmp[ipar[22]-1]);
			} 
			else {
				// failed
				break;
			}
		}

		if (rci == 0) {
			// retrieve solution
			dfgmres_get(&ivar, sol, rhs, &rci, ipar, dpar, tmp, &itercount);
			std::cout << "Solved: " << "iter=" << itercount << std::endl;
		} else {
			std::cerr << "Error: " << "rci=" << rci << std::endl;
		}

		// cleanup
		MKL_Free_Buffers();
	} // end solve boundary force

	if (0) { // check velocity is exactly recovered
		double *ubsol = new double[sf.nb*3];
		stokes_apply_forcing(sf, sf.fb, ubsol);

		double uerr = 0;
		for (int ib=0; ib<sf.nb; ib++) {
			const int ioff = ib * 3;
			double iberr = 0;
			for (int dir=0; dir<3; dir++) {
				iberr += pow(ubsol[ioff+dir]-sf.ub[ioff+dir], 2);
			}
			iberr = sqrt(iberr);
			uerr = std::max(uerr, iberr);
		}

		std::cout << "|u-ub|=" << uerr << std::endl;

		delete[] ubsol;
	}

	if (0) { // calculate velocity on grid for visualization
		// note that we already have the global velocity from the last solver iteration
		// so add local velocity only

		double *ug = stokes.u;
		double *vg = stokes.v;
		double *wg = stokes.w;

		const int np = sf.nb;
		const double *xp = sf.xb;
		const double *fp = sf.fb;

		const int xspan = (int) (rcut/dx) + 1;
		const int yspan = (int) (rcut/dy) + 1;
		const int zspan = (int) (rcut/dz) + 1;

		for (int ip=0; ip<np; ip++) {
			const int ioff = ip * 3;
			
			const double *x = &xp[ioff];
			const double *f = &fp[ioff];

			// 
			int icell = (int) floor((x[0] - xlo) / dx);
			int jcell = (int) floor((x[1] - ylo) / dy);
			int kcell = (int) floor((x[2] - zlo) / dz);

			for (int ii=icell-xspan; ii<=icell+1+xspan; ii++) {
			for (int jj=jcell-yspan; jj<=jcell+1+yspan; jj++) {
			for (int kk=kcell-zspan; kk<=kcell+1+zspan; kk++) {
				double xx = xlo + dx*ii;
				double yy = ylo + dy*jj;
				double zz = zlo + dz*kk;
				double xdist = xx - x[0];
				double ydist = yy - x[1];
				double zdist = zz - x[2];
				double dist = sqrt(xdist*xdist + ydist*ydist + zdist*zdist);

				if (dist < rcut) {
					double ul[3];
					func_gl(ul, f, 
						dist, xdist, ydist, zdist, 
						alpha, xi, mu);

					int igrid, jgrid, kgrid;
					stokes_map_grid(igrid, jgrid, kgrid, ii, jj, kk, nx, ny, nz);

					int idx = igrid*ny*nz + jgrid*nz + kgrid;
					ug[idx] += ul[0];
					vg[idx] += ul[1];
					wg[idx] += ul[2];
				}
			}
			}
			}
		} // end loop forcing
	}

	//stokes_output_vtk(stokes, "test_bndry00.vtk");
	//stokes_forcing_output_csv(sf, "test_bndry00.csv");

	if (1) {
		double fx = 0;
		for (int i=0; i<sf.nb; i++) {
			fx += sf.fb[i*3+0];
		}
		std::cout << "Fx=" << fx << std::endl;
	}
}



// up = U + Omega*r
static void stokes_calc_up(StokesForcing &sf,
	double *up, const double *part_vel)
{
	const int np = sf.np;
	const double *xp = sf.xp;
	const int *idp = sf.idp;

	const double *part_pos = sf.part_pos;

	for (int i=0; i<np; i++) {
		const int ipart = idp[i];
		
		double xx[3];
		for (int dir=0; dir<3; dir++) {
			xx[dir] = xp[i*3+dir] - part_pos[ipart*3+dir];
		}

		const double *usph = &part_vel[ipart*6];
		const double *osph = usph + 3;

		up[i*3+0] = usph[0] + osph[1]*xx[2] - osph[2]*xx[1];
		up[i*3+1] = usph[1] + osph[2]*xx[0] - osph[0]*xx[2];
		up[i*3+2] = usph[2] + osph[0]*xx[1] - osph[1]*xx[0];
	}
}

static void stokes_calc_partforce(StokesForcing &sf,
	const double *fp, double *part_force)
{
	const int np = sf.np;
	const double *xp = sf.xp;
	const int *idp = sf.idp;

	const int part_num = sf.part_num;
	const double *part_pos = sf.part_pos;

	for (int i=0; i<part_num*6; i++) {
		part_force[i] = 0;
	}

	for (int i=0; i<np; i++) {
		const int ipart = idp[i];
		
		double xx[3];
		for (int dir=0; dir<3; dir++) {
			xx[dir] = xp[i*3+dir] - part_pos[ipart*3+dir];
		}

		const double *ff = &fp[i*3];

		// force
		part_force[ipart*6+0] += ff[0];
		part_force[ipart*6+1] += ff[1];
		part_force[ipart*6+2] += ff[2];
		// torque
		part_force[ipart*6+3] += xx[1]*ff[2] - xx[2]*ff[1];
		part_force[ipart*6+4] += xx[2]*ff[0] - xx[0]*ff[2];
		part_force[ipart*6+5] += xx[0]*ff[1] - xx[1]*ff[0];
	}

}

// given boundary force + particle velocity
// solve particle force
static void stokes_solve_ib2(
	StokesForcing &sf, StokesFFT &stokes,
	const double *fbin, const double *uin,
	double *ubout, double *fout) 
{
	const int nb = sf.nb;
	const int nb3 = nb * 3;
	const int np = sf.np;
	const int np3 = np * 3;
	const int nall = nb + np;

	const int nunk = np * 3;

	//
	double *rhs = new double[nunk];
	double *sol = new double[nunk];
	for (int i=0; i<nunk; i++) {
		rhs[i] = 0;
		sol[i] = 0;
	}

	// fill forcing vector [fb=fbin;fp=0]
	for (int i=0; i<nb3; i++) {
		sf.fb[i] = fbin[i];
	}
	for (int i=0; i<np3; i++) {
		sf.fp[i] = 0;
	}

	// compute upstar
	stokes_apply_forcing2(sf, sf.fall, sf.uall);

	// U -> up
	stokes_calc_up(sf, rhs, uin);

	// rhs = up - upstar
	for (int i=0; i<np3; i++) {
		rhs[i] -= sf.up[i];
	}

	// fill [fb=0]
	for (int i=0; i<nb3; i++) {
		sf.fb[i] = 0;
	}

	{
		const int use_precond = 1 && (sf.mat_precond!=NULL);
		const double eps_rel = 1.0e-8;
		const double eps_abs = 0.0;

		const int npar = 128;
		MKL_INT ipar[npar] = { 0 };
		double dpar[npar] = { 0 };

		// GMRES buffer
		double *tmp = new double[nunk*(2*nunk+1) + (nunk*(nunk+9))/2 + 1];

		double *bilu0 = NULL;
		double *trvec = NULL;
		if (sf.mat_precond && use_precond) {
			assert(sf.mat_precond->num_row() == nunk);
			assert(sf.mat_precond->num_col() == nunk);

			bilu0 = new double[sf.mat_precond->ndata];
			trvec = new double[nunk];
		}

		MKL_INT rci = 0;
		MKL_INT ivar = nunk;
		MKL_INT ierr = 0;
		MKL_INT itercount = 0;

		// init GMRES
		dfgmres_init(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		if (rci != 0) {
			std::cerr << __FUNCTION__ << ": INIT GMRES failed rci=" << rci << std::endl;
			exit(1);
		}

		if (use_precond) {
			// init precond
			dcsrilu0(&ivar, sf.mat_precond->data(), sf.mat_precond->ia(), sf.mat_precond->ja(), bilu0, ipar, dpar, &ierr);
			if (ierr != 0) {
				std::cerr << __FUNCTION__ << ": INIT ILU0 failed err=" << ierr << std::endl;
				exit(1);
			}
		}

		ipar[8] = 1; // do residual stopping test
		ipar[9] = 0; // no user stopping test
		ipar[10] = use_precond; // no precond
		ipar[11] = 1; // auto test for breakdown
		ipar[14] = 15; // restart
		//
		dpar[0] = eps_rel; // relative error
		dpar[1] = eps_abs; // absolute error

		//
		dfgmres_check(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		if (rci != 0) {
			std::cerr << __FUNCTION__ << ": CHECK failed rci=" << rci << std::endl;
			exit(1);
		}

		while (1) {
			// loop
			dfgmres(&ivar, sol, rhs, &rci, ipar, dpar, tmp);

			if (rci == 0) { // converged
				break;
			} else if (rci == 1) { // compute A*x
				double *fpin = &tmp[ipar[21]-1];
				double *upout = &tmp[ipar[22]-1];

				// fill forcing vector [fb=0;fp=fpin]
				for (int i=0; i<nb3; i++) {
					sf.fb[i] = 0;
				}
				for (int i=0; i<np3; i++) {
					sf.fp[i] = fpin[i];
				}

				stokes_apply_forcing2(sf, sf.fall, sf.uall);

				for (int i=0; i<np3; i++) {
					upout[i] = sf.up[i];
				}
			} 
			else if (rci==3 && use_precond) {
				int *ia = sf.mat_precond->ia();
				int *ja = sf.mat_precond->ja();

				char cvar1, cvar, cvar2;
				cvar1 = 'L'; 
				cvar = 'N';
				cvar2 = 'U';
				mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, &tmp[ipar[21]-1], trvec);
				cvar1 = 'U'; 
				cvar = 'N';
				cvar2 = 'N';
				mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, trvec, &tmp[ipar[22]-1]);
			} 
			else {
				// failed
				break;
			}
		}

		if (rci == 0) {
			// retrieve solution
			dfgmres_get(&ivar, sol, rhs, &rci, ipar, dpar, tmp, &itercount);
			std::cout << __FUNCTION__ << ": Solved: " << "iter=" << itercount << std::endl;

			// fill forcing vector [fb=fbin;fp=sol]
			for (int i=0; i<nb3; i++) {
				sf.fb[i] = fbin[i];
			}
			for (int i=0; i<np3; i++) {
				sf.fp[i] = sol[i];
			}

			stokes_apply_forcing2(sf, sf.fall, sf.uall);

			// boundary velocity
			for (int i=0; i<nb3; i++) {
				ubout[i] = sf.ub[i];
			}
			
			// particle force
			stokes_calc_partforce(sf, sf.fp, fout);
		} else {
			std::cerr << __FUNCTION__ << ": Error: " << "rci=" << rci << std::endl;
			exit(1);
		}

		// cleanup
		MKL_Free_Buffers();

		delete[] tmp;
		if (bilu0) delete[] bilu0;
		if (trvec) delete[] trvec;
	}

	delete[] rhs;
	delete[] sol;
	
}

// solve boundary force + particle velocity
static void stokes_solve_ib(StokesForcing &sf, StokesFFT &stokes) 
{
	const int nunkb = sf.nb * 3;
	const int nunkp = sf.part_num * 6;
	const int nunk = nunkb + nunkp;

	// offset of boundary/particle
	const int ioffb = 0;
	const int ioffp = nunkb;

	//
	const int use_precond = 0 && (sf.mat_precond!=NULL);
	const double eps_rel = 1.0e-6;
	const double eps_abs = 0.0;

	const int npar = 128;
	MKL_INT ipar[npar] = { 0 };
	double dpar[npar] = { 0 };

	//
	double *ubsave = new double[sf.nb*3];
	double *rhs = new double[nunk];
	double *sol = new double[nunk];
	for (int i=0; i<nunkb; i++) {
		rhs[i+ioffb] = sf.ub[i];
		sol[i+ioffb] = 0;

		ubsave[i] = sf.ub[i];
	}
	for (int i=0; i<nunkp; i++) {
		rhs[i+ioffp] = sf.part_force[i];
		sol[i+ioffp] = 0;
	}

	// GMRES buffer
	double *tmp = new double[nunk*(2*nunk+1) + (nunk*(nunk+9))/2 + 1];

	double *bilu0 = NULL;
	double *trvec = NULL;
	if (sf.mat_precond && use_precond) {
		assert(sf.mat_precond->num_row() == nunk);
		assert(sf.mat_precond->num_col() == nunk);

		bilu0 = new double[sf.mat_precond->ndata];
		trvec = new double[nunk];
	}

	MKL_INT rci = 0;
	MKL_INT ivar = nunk;
	MKL_INT ierr = 0;
	MKL_INT itercount = 0;

	// init GMRES
	dfgmres_init(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
	if (rci != 0) {
		std::cerr << __FUNCTION__ << ": INIT GMRES failed rci=" << rci << std::endl;
		exit(1);
	} else {
		std::cout << __FUNCTION__ << ": INIT GMRES rci=" << rci << std::endl;
	}

	if (use_precond) {
		// init precond
		dcsrilu0(&ivar, sf.mat_precond->data(), sf.mat_precond->ia(), sf.mat_precond->ja(), bilu0, ipar, dpar, &ierr);
		if (ierr != 0) {
			std::cerr << __FUNCTION__ << ": INIT ILU0 failed err=" << ierr << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT ILU0 err=" << ierr << std::endl;
		}
	}

	ipar[8] = 1; // do residual stopping test
	ipar[9] = 0; // no user stopping test
	ipar[10] = use_precond; // no precond
	ipar[11] = 1; // auto test for breakdown
	ipar[14] = 40; // restart
	//
	dpar[0] = eps_rel; // relative error
	dpar[1] = eps_abs; // absolute error

	//
	dfgmres_check(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
	if (rci != 0) {
		std::cerr << __FUNCTION__ << ": CHECK failed rci=" << rci << std::endl;
		exit(1);
	}

	while (1) {
		// loop
		dfgmres(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		std::cout << "|res|=" << dpar[4] << "/" << dpar[2] << std::endl;

		if (rci == 0) {
			// converged
			break;
		} else if (rci == 1) { // compute A*x

			double *fbin = &tmp[ipar[21]-1 + ioffb];
			double *uin = &tmp[ipar[21]-1 + ioffp];

			double *ubout = &tmp[ipar[22]-1 + ioffb];
			double *fout = &tmp[ipar[22]-1 + ioffp];

			stokes_solve_ib2(sf, stokes, fbin, uin, ubout, fout);
			//std::cout << fout[0] << "," << fout[1] << "," << fout[2] << "," << std::endl;
		} 
		else if (rci==3 && use_precond) {
			int *ia = sf.mat_precond->ia();
			int *ja = sf.mat_precond->ja();

			char cvar1, cvar, cvar2;
			cvar1 = 'L'; 
			cvar = 'N';
			cvar2 = 'U';
			mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, &tmp[ipar[21]-1], trvec);
			cvar1 = 'U'; 
			cvar = 'N';
			cvar2 = 'N';
			mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, trvec, &tmp[ipar[22]-1]);
		} 
		else {
			// failed
			break;
		}
	}

	if (rci == 0) {
		// retrieve solution
		dfgmres_get(&ivar, sol, rhs, &rci, ipar, dpar, tmp, &itercount);
		std::cout << "Solved: " << "iter=" << itercount << std::endl;

		// put solution to correct place
		// boundary force, boundary velocity
		for (int i=0; i<nunkb; i++) { 
			sf.fb[i] = sol[i+ioffb];
			sf.ub[i] = ubsave[i];
		}
		// particle velocity
		for (int i=0; i<nunkp; i++) { 
			sf.part_vel[i] = sol[i+ioffp];
		}
		// up
		stokes_calc_up(sf, sf.up, sf.part_vel);
	} else {
		std::cerr << "Error: " << "rci=" << rci << std::endl;
	}

	// cleanup
	MKL_Free_Buffers();
	delete[] rhs;
	delete[] sol;
	delete[] tmp;
	if (bilu0) delete[] bilu0;
	if (trvec) delete[] trvec;
	delete[] ubsave;
}


// solve boundary force + particle velocity
static void stokes_solve_forcing(StokesForcing &sf, StokesFFT &stokes) 
{
	const int nunk = sf.nall * 3;

	//
	const int use_precond = 0 && (sf.mat_precond!=NULL);
	const double eps_rel = 1.0e-8;
	const double eps_abs = 0.0;

	const int npar = 128;
	MKL_INT ipar[npar] = { 0 };
	double dpar[npar] = { 0 };

	//
	double *rhs = new double[nunk];
	double *sol = new double[nunk];
	for (int i=0; i<nunk; i++) {
		sol[i] = 0;
		rhs[i] = sf.uall[i];
	}

	// GMRES buffer
	double *tmp = new double[nunk*(2*nunk+1) + (nunk*(nunk+9))/2 + 1];

	double *bilu0 = NULL;
	double *trvec = NULL;
	if (sf.mat_precond && use_precond) {
		assert(sf.mat_precond->num_row() == nunk);
		assert(sf.mat_precond->num_col() == nunk);

		bilu0 = new double[sf.mat_precond->ndata];
		trvec = new double[nunk];
	}

	MKL_INT rci = 0;
	MKL_INT ivar = nunk;
	MKL_INT ierr = 0;
	MKL_INT itercount = 0;

	// init GMRES
	dfgmres_init(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
	if (rci != 0) {
		std::cerr << __FUNCTION__ << ": INIT GMRES failed rci=" << rci << std::endl;
		exit(1);
	} else {
		std::cout << __FUNCTION__ << ": INIT GMRES rci=" << rci << std::endl;
	}

	if (use_precond) {
		// init precond
		dcsrilu0(&ivar, sf.mat_precond->data(), sf.mat_precond->ia(), sf.mat_precond->ja(), bilu0, ipar, dpar, &ierr);
		if (ierr != 0) {
			std::cerr << __FUNCTION__ << ": INIT ILU0 failed err=" << ierr << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT ILU0 err=" << ierr << std::endl;
		}
	}

	ipar[8] = 1; // do residual stopping test
	ipar[9] = 0; // no user stopping test
	ipar[10] = use_precond; // no precond
	ipar[11] = 1; // auto test for breakdown
	ipar[14] = 20; // restart
	//
	dpar[0] = eps_rel; // relative error
	dpar[1] = eps_abs; // absolute error

	//
	dfgmres_check(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
	if (rci != 0) {
		std::cerr << __FUNCTION__ << ": CHECK failed rci=" << rci << std::endl;
		exit(1);
	}

	while (1) {
		// loop
		dfgmres(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		//std::cout << "|res|=" << dpar[4] << "/" << dpar[2] << std::endl;

		if (rci == 0) {
			// converged
			break;
		} else if (rci == 1) { // compute A*x
			double *fin = &tmp[ipar[21]-1];
			double *uout = &tmp[ipar[22]-1];

			stokes_apply_forcing2(sf, fin, uout);
		} 
		else if (rci==3 && use_precond) {
			int *ia = sf.mat_precond->ia();
			int *ja = sf.mat_precond->ja();

			char cvar1, cvar, cvar2;
			cvar1 = 'L'; 
			cvar = 'N';
			cvar2 = 'U';
			mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, &tmp[ipar[21]-1], trvec);
			cvar1 = 'U'; 
			cvar = 'N';
			cvar2 = 'N';
			mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar, bilu0, ia, ja, trvec, &tmp[ipar[22]-1]);
		} 
		else {
			// failed
			break;
		}
	}

	if (rci == 0) {
		// retrieve solution
		dfgmres_get(&ivar, sol, rhs, &rci, ipar, dpar, tmp, &itercount);
		std::cout << "Solved: " << "iter=" << itercount << std::endl;

		for (int i=0; i<nunk; i++) {
			sf.fall[i] = sol[i];
		}
	} else {
		std::cerr << "Error: " << "rci=" << rci << std::endl;
	}

	// cleanup
	MKL_Free_Buffers();
	delete[] rhs;
	delete[] sol;
	delete[] tmp;
	if (bilu0) delete[] bilu0;
	if (trvec) delete[] trvec;
}


//
// 
static void stokes_solve_all(StokesForcing &sf, StokesFFT &stokes) 
{
	const int nb = sf.nb;
	const int np = sf.np;
	const int nall = sf.nall;
	const int npart = sf.part_num;
	assert(nall == nb+np);

	const int nb3 = nb * 3;
	const int np3 = np * 3;
	const int nall3 = nall * 3;
	const int npart6 = npart * 6;

	// unknown dimension
	const int nunk = nb3 + np3 + npart6;

	// offset in vector
	const int ioffb = 0;
	const int ioffp = ioffb + nb3;
	const int ioffpart = ioffp + np3;

	// solver parameter
	const int use_precond = 1 && (sf.mat_precond!=NULL);
	const double eps_rel = 1.0e-6;
	const double eps_abs = 0.0;
	const int max_iter = 20000;
	const int nrestart = std::min(300, nunk);

	const int npar = 128;
	MKL_INT ipar[npar] = { 0 };
	double dpar[npar] = { 0 };

	//
	double *ubsave = new double[nb3];
	double *uptmp = new double[np3];
	// save ub
	cblas_dcopy(nb3, sf.ub, 1, ubsave, 1);

	// 
	double *rhs = new double[nunk];
	double *sol = new double[nunk];

	// make rhs [ ub; 0; -F ]
	for (int i=0; i<nb3; i++) { 
		rhs[i+ioffb] = sf.ub[i];
	}
	for (int i=0; i<np3; i++) {
		rhs[i+ioffp] = 0;
	}
	for (int i=0; i<npart6; i++) {
		rhs[i+ioffpart] = -sf.part_force[i];
		//rhs[i+ioffpart] = sf.part_force[i];
	}

	// initial guess [fb;fp;U]
	for (int i=0; i<nb3; i++) {
		sol[i+ioffb] = sf.fb[i]; 
	}
	for (int i=0; i<np3; i++) {
		sol[i+ioffp] = sf.fp[i];
	}
	for (int i=0; i<npart6; i++) {
		sol[i+ioffpart] = sf.part_vel[i];
	}

	// GMRES buffer
	//double *tmp = new double[nunk*(2*nunk+1) + (nunk*(nunk+9))/2 + 1];
	double *tmp = new double[nunk*(2*nrestart+1) + (nrestart*(nrestart+9))/2 + 1];

	// preconditioner
	double *bilu0 = NULL;
	double *trvec = NULL;
	double *trvec2 = NULL;
	if (use_precond) {
		//assert(sf.mat_precond->num_row() == nunk);
		//assert(sf.mat_precond->num_col() == nunk);

		bilu0 = new double[sf.mat_precond->num_val()];
		trvec = new double[nunk];
		trvec2 = new double[nunk];
	}

	//
	MKL_INT rci = 0;
	MKL_INT ivar = nunk;
	MKL_INT ivar2 = use_precond ? (sf.mat_precond->num_row()) : 0;
	MKL_INT ierr = 0;
	MKL_INT itercount = 0;

	// init GMRES
	dfgmres_init(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
	if (rci != 0) {
		std::cerr << __FUNCTION__ << ": INIT GMRES failed rci=" << rci << std::endl;
		exit(1);
	} else {
		std::cout << __FUNCTION__ << ": INIT GMRES rci=" << rci << std::endl;
	}

	if (use_precond) {
		// init precond
		assert(sf.mat_precond->num_row() == ivar2);
		assert(sf.mat_precond->num_col() == ivar2);

		const double replace_zero_diag = 0;
		if (replace_zero_diag > 0) {
			ipar[30] = 1;
			dpar[30] = replace_zero_diag;
			dpar[31] = replace_zero_diag;
		}

		dcsrilu0(&ivar2, sf.mat_precond->data(), sf.mat_precond->ia(), sf.mat_precond->ja(), bilu0, ipar, dpar, &ierr);
		if (ierr != 0) {
			std::cerr << __FUNCTION__ << ": INIT ILU0 failed err=" << ierr << std::endl;
			exit(1);
		} else {
			std::cout << __FUNCTION__ << ": INIT ILU0 err=" << ierr << std::endl;
		}
	}

	// set parameter
	ipar[4] = max_iter; // iteration
	ipar[8] = 1; // do residual stopping test
	ipar[9] = 0; // no user stopping test
	ipar[10] = use_precond; // no precond
	ipar[11] = 1; // auto test for breakdown
	ipar[14] = nrestart; // restart
	//
	dpar[0] = eps_rel; // relative error
	dpar[1] = eps_abs; // absolute error

	//
	dfgmres_check(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
	if (rci != 0) {
		std::cerr << __FUNCTION__ << ": CHECK failed rci=" << rci << std::endl;
		exit(1);
	}

	while (1) {
		// loop
		dfgmres(&ivar, sol, rhs, &rci, ipar, dpar, tmp);
		//std::cout << "|res|=" << dpar[4] << "/" << dpar[2] << std::endl;

		if (rci == 0) {
			// converged
			break;
		} else if (rci == 1) { // compute A*x
			const int in = ipar[21] - 1;
			const int out = ipar[22] - 1;

			double *fbin = &tmp[in + ioffb];
			double *fpin = &tmp[in + ioffp];
			double *uin = &tmp[in + ioffpart]; 

			double *ubout = &tmp[out + ioffb];
			double *upout = &tmp[out + ioffp];
			double *fout = &tmp[out + ioffpart];

			// apply M.[fb;fp] -> [ub;up]
			stokes_apply_forcing2(sf, fbin, ubout);

			// calc SigmaT.U -> uptmp
			stokes_calc_up(sf, uptmp, uin);

			// up - SigmaT.U
			for (int i=0; i<np3; i++) {
				upout[i] -= uptmp[i];
			}

			// Sigma.fp
			stokes_calc_partforce(sf, fpin, fout);
			for (int i=0; i<npart6; i++) {
				fout[i] = -fout[i];
			}
		} 
		else if (rci==3 && use_precond) {
			double *vecin = &tmp[ipar[21]-1];
			double *vecout = &tmp[ipar[22]-1];
			
			//for (int i=0; i<nb3; i++) {
			//	trvec2[i+ioffb] = vecin[i+ioffb];
			//}
			//stokes_calc_up(sf, trvec2+ioffp, vecin+ioffpart);

			// 
			int *ia = sf.mat_precond->ia();
			int *ja = sf.mat_precond->ja();

			char cvar1, cvar, cvar2;
			cvar1 = 'L'; 
			cvar = 'N';
			cvar2 = 'U';
			mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar2, bilu0, ia, ja, vecin, trvec);
			//mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar2, bilu0, ia, ja, trvec2, trvec);
			cvar1 = 'U'; 
			cvar = 'N';
			cvar2 = 'N';
			mkl_dcsrtrsv(&cvar1, &cvar, &cvar2, &ivar2, bilu0, ia, ja, trvec, vecout);

			if (ivar2 != nunk) {
			//
			//const double mdiag = 1.0 / (8.0*M_PI*stokes.visc) * 4.0*sf.xi/sqrt(M_PI);
			const double mdiag = 1.0 / (8.0*M_PI*stokes.visc);
			const double coef = 1.0 / mdiag;
			for (int i=0; i<npart6; i++) {
				vecout[i+ioffpart] = vecin[i+ioffpart] * coef;
			}
			}
		} 
		else {
			// failed
			break;
		}
	}

	if (rci == 0) {
		// retrieve solution
		dfgmres_get(&ivar, sol, rhs, &rci, ipar, dpar, tmp, &itercount);
		std::cout << "Solved: " << "iter=" << itercount << std::endl;

		// fb
		for (int i=0; i<nb3; i++) { 
			sf.fb[i] = sol[i+ioffb];
		}
		// fp
		for (int i=0; i<np3; i++) { 
			sf.fp[i] = sol[i+ioffp];
		}
		// Upart
		for (int i=0; i<npart6; i++) {
			sf.part_vel[i] = sol[i+ioffpart];
		}

		// restore ub
		for (int i=0; i<nb3; i++) {
			sf.ub[i] = ubsave[i];
		}

		// calc up
		stokes_calc_up(sf, sf.up, sf.part_vel);
	} else {
		std::cerr << "Error: " << "rci=" << rci << std::endl;
	}

	// cleanup
	MKL_Free_Buffers();
	delete[] rhs;
	delete[] sol;
	delete[] tmp;
	if (bilu0) delete[] bilu0;
	if (trvec) delete[] trvec;
	if (trvec2) delete[] trvec2;
	delete[] ubsave;
	delete[] uptmp;
}




void test_stokes_part(int argc, char *argv[]) {
	
	//
	StokesFFT stokes;

	// to have 6*pi*mu = 1.0
	//const double mu = 1.0/(6.0*M_PI);
	const double mu = 1.0;

	stokes.dens = 1.0;
	stokes.visc = mu;

	// domain
	const double len = 1.0;
	const double xlo = -len/2, xhi = len/2;
	const double zlo = -len/2, zhi = len/2;
	//const double ylo = -len*2/2, yhi = len*2/2;
	const double ylo = -len*0.75, yhi = len*0.75;
	
	// grid
	const int ncell = 100;
	const int nx = ncell;
	const int nz = ncell;
	//const int ny = ncell*2;
	const int ny = ncell*3/2;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;
	const double dz = (zhi-zlo) / nz;

	//
	stokes_set_grid(stokes, xlo, ylo, zlo, xhi, yhi, zhi, nx, ny, nz);
	stokes_init(stokes);
	stokes_init_fft(stokes);

	//
	StokesForcing sf;
	sf.stokes = &stokes;

	// regular factor
	// screen factor
	//const double xi = 1.3 / dx;
	//const double alpha = 0.8 / dx;
	//const double xi = 1.8/dx; // n=80
	//const double alpha = 1.2 / dx; // n=80
	const double xi = 1.8/dx; // n=100
	const double alpha = 1.15 / dx; // n=100
	// cutoff distance
	const double rcut = 4.0 / alpha;

	sf.xi = xi;
	sf.alpha = alpha;
	sf.rcut = rcut;

	// allocate buffer
	const int nallmax = 100000;
	const int partmax = 1000;
	{
		sf.nall = 0;
		sf.xall = new double[nallmax*3];
		sf.fall = new double[nallmax*3];
		sf.uall = new double[nallmax*3];
		sf.idall = new int[nallmax];

		sf.tmp[0] = new double[nallmax];
		sf.tmp[1] = new double[nallmax];
		sf.tmp[2] = new double[nallmax];

		for (int i=0; i<nallmax; i++) {
			sf.idall[i] = 0;
			for (int dir=0; dir<3; dir++) {
				sf.xall[i*3+dir] = 0;
				sf.fall[i*3+dir] = 0;
				sf.uall[i*3+dir] = 0;

				sf.tmp[dir][i] = 0;
			}
		}

		// these will take place in ALL
		sf.nb = 0;
		sf.xb = NULL;
		sf.ub = NULL;
		sf.fb = NULL;
		sf.idb = NULL;
		sf.np = 0;
		sf.xp = NULL;
		sf.up = NULL;
		sf.fp = NULL;
		sf.idp = NULL;

		sf.part_num = 0;
		sf.part_rad = new double[partmax];
		sf.part_pos = new double[partmax*3];
		sf.part_vel = new double[partmax*6];
		sf.part_force = new double[partmax*6];
		sf.part_ipbegin = new int[partmax];
		sf.part_ipend = new int[partmax];
		for (int i=0; i<partmax; i++) {
			sf.part_rad[i] = 0;
			for (int dir=0; dir<3; dir++) {
				sf.part_pos[i*3+dir] = 0;
				sf.part_vel[i*6+dir] = 0;
				sf.part_vel[i*6+3+dir] = 0;
				sf.part_force[i*6+dir] = 0;
				sf.part_force[i*6+3+dir] = 0;
			}
			sf.part_ipbegin[i] = 0;
			sf.part_ipend[i] = 0;
		}
	}


	{ // load particles
		int npart = 0; 

		if (0) {
			npart = 1;
			int ipart = 0;
			sf.part_rad[ipart] = 0.1;
			for (int dir=0; dir<3; dir++) {
				sf.part_pos[ipart*3+dir] = 0; // position
				sf.part_vel[ipart*6+dir] = 0; // velocity 
				sf.part_vel[ipart*6+3+dir] = 0; // angular velocity
				sf.part_force[ipart*6+dir] = 0; // force
				sf.part_force[ipart*6+3+dir] = 0; // torque
			}
			sf.part_num += 1;
		}

		if (1) {
			if (argc != 3) {
				std::cerr << "Usage: " << __FUNCTION__ << " <input_dem.csv> <npart>" << std::endl;
				exit(1);
			}
			npart = atoi(argv[2]);
			const char *inputfilename = argv[1];
			FILE *fpdem = fopen(inputfilename, "r");
			if (!fpdem) {
				std::cerr << "Failed to open " << inputfilename << std::endl;
				exit(1);
			} else {
				std::cout << "Read " << inputfilename << std::endl;
			}

			const int buflen = 1024;
			char buf[buflen];

			// skip header
			fgets(buf, buflen, fpdem);

			for (int ipart=0; ipart<npart; ipart++) {
				fgets(buf, buflen, fpdem);

				int idx;
				double dpart, xpart, ypart, zpart;
				sscanf(buf, "%d,%lf,%lf,%lf,%lf", 
					&idx, &dpart, &xpart, &ypart, &zpart);

				sf.part_rad[ipart] = dpart * 0.5;
				sf.part_pos[ipart*3+0] = xpart;
				sf.part_pos[ipart*3+1] = ypart;
				sf.part_pos[ipart*3+2] = zpart;
				for (int dir=0; dir<3; dir++) {
					sf.part_vel[ipart*6+dir] = 0; // velocity 
					sf.part_vel[ipart*6+3+dir] = 0; // angular velocity
					sf.part_force[ipart*6+dir] = 0; // force
					sf.part_force[ipart*6+3+dir] = 0; // torque
				}
				sf.part_num += 1;
			}

			fclose(fpdem);
		}

		if (sf.part_num > partmax) {
			std::cerr << "Particle overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Particle=" << sf.part_num << std::endl;
		}
	}


	{ // set boundary points
		// connect buffer
		sf.nb = 0;
		sf.xb = sf.xall + sf.nall*3;
		sf.ub = sf.uall + sf.nall*3;
		sf.fb = sf.fall + sf.nall*3;
		sf.idb = sf.idall + sf.nall;

		const double dh = dx * 2.1; // typical distance between boundary point
		const double gammadot = 1.0; // shear rate

		if (1) { // add top/bottom wall
			//const double ythick = len / 4;
			const double ythick = len / 8;
			const double ytop = len / 2;
			const double utop[3] = { ytop*gammadot, 0.0, 0.0 };
			const double ybot = -len / 2;
			const double ubot[3] = { ybot*gammadot, 0.0, 0.0 };
			
			int nsave = sf.nb;
			gen_plane_y(ybot-ythick, ubot, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallBot2;
			}

			nsave = sf.nb;
			gen_plane_y(ybot, ubot, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallBot;
			}

			nsave = sf.nb;
			gen_plane_y(ytop, utop, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallTop;
			}

			nsave = sf.nb;
			gen_plane_y(ytop+ythick, utop, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallTop2;
			}
		}

		sf.nall += sf.nb;

		if (sf.nall > nallmax) {
			std::cerr << "Nb overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Nb=" << sf.nb << std::endl;
		}
	}

	{ // generate particle points
		// connect buffer
		sf.np = 0;
		sf.xp = sf.xall + sf.nall*3;
		sf.up = sf.uall + sf.nall*3;
		sf.fp = sf.fall + sf.nall*3;
		sf.idp = sf.idall + sf.nall;
		assert(sf.idp - sf.idb == sf.nb);


		for (int ipart=0; ipart<sf.part_num; ipart++) {
			const double apart = sf.part_rad[ipart];
			const double *xpart = &sf.part_pos[ipart*3];
			const double *upart = &sf.part_vel[ipart*6];
			const double *opart = &sf.part_vel[ipart*6+3];

			const double dh = dx * 2.1; // typical distance between boundary point
			//const double dh = dx * 1.5; // typical distance between boundary point
			//const double dh = dx * 2.0; // typical distance between boundary point
			//const double dh = apart / 2.0;

			int nsave = sf.np;
			gen_sphere(apart, xpart, upart, opart, 
				sf.np, sf.xp, sf.up, dh);

			// point to particle
			for (int i=nsave; i<sf.np; i++) {
				sf.idp[i] = ipart;
			}
			// particle to point
			sf.part_ipbegin[ipart] = nsave;
			sf.part_ipend[ipart] = sf.np;
		}

		sf.nall += sf.np;

		if (sf.nall > nallmax) {
			std::cerr << "Np overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Np=" << sf.np << std::endl;
		}
	}

	{
		std::cout << "Nall=" << sf.nall << std::endl;
	}


	{ // cache points
		sf.cache = new PointCache;
		
		PointCache &cache = *sf.cache;

		cache.xmin[0] = xlo;
		cache.xmin[1] = ylo;
		cache.xmin[2] = zlo;
		cache.xmax[0] = xhi;
		cache.xmax[1] = yhi;
		cache.xmax[2] = zhi;

		for (int dir=0; dir<3; dir++) {
			double len = cache.xmax[dir] - cache.xmin[dir];
			int num = (int) (len / rcut);
			if (num < 1) num = 1;

			cache.nbin[dir] = num;
			cache.hbin[dir] = len / num;
		}

		int ncache = cache.nbin[0] * cache.nbin[1] * cache.nbin[2];
		cache.head = new int[ncache];
		cache.tail = new int[ncache];
		cache.next = new int[nallmax];

		std::cout << "Build point cache" << std::endl;

		// clear cache
		for (int i=0; i<ncache; i++) {
			cache.head[i] = -1;
			cache.tail[i] = -1;
		}
		for (int i=0; i<sf.nall; i++) {
			cache.next[i] = -1;
		}

		// save into cache
		for (int i=0; i<sf.nall; i++) {
			const double *pos = &sf.xall[i*3];

			int loc[3];
			for (int dir=0; dir<3; dir++) {
				loc[dir] = (int) floor((pos[dir]-cache.xmin[dir]) / cache.hbin[dir]);
				if (loc[dir] < 0) loc[dir] += cache.nbin[dir];
				if (loc[dir] >= cache.nbin[dir]) loc[dir] -= cache.nbin[dir];
			}

			int idx = loc[0]*cache.nbin[1]*cache.nbin[2] + loc[1]*cache.nbin[2] + loc[2];

			int oldtail = cache.tail[idx];
			cache.tail[idx] = i;
			if (oldtail == -1) {
				cache.head[idx] = i;
			} else {
				cache.next[oldtail] = i;
			}
		}
	}


	sf.mat_local = NULL;
	if (1) {
		std::cout << "Build local forcing matrix" << std::endl;
		
		int nrow = sf.nall * 3;
		int ncol = nrow;
		sf.mat_local = new CSRMat(nrow, ncol);

		//stokes_build_local_forcing_matrix(sf, *sf.mat_local, sf.nall, sf.xall);
		stokes_build_local_forcing_matrix2(sf, *sf.mat_local, sf.nall, sf.xall, 
			sf.alpha, sf.xi, sf.rcut);
	}
	
	sf.mat_precond = NULL;
	if (0) { // precondition for inner solver
		std::cout << "Build precondition matrix" << std::endl;
		
		int nrow = sf.np * 3;
		int ncol = nrow;
		sf.mat_precond = new CSRMat(nrow, ncol);

		stokes_build_precond_matrix(sf, *sf.mat_precond, sf.np, sf.xp);
	}
	if (0) { // precondition for all
		std::cout << "Build precondition matrix" << std::endl;
		
		int nrow = sf.nall * 3;
		int ncol = nrow;
		sf.mat_precond = new CSRMat(nrow, ncol);

		//stokes_build_precond_matrix(sf, *sf.mat_precond, sf.nall, sf.xall);
		stokes_build_local_forcing_matrix2(sf, *sf.mat_precond, sf.nall, sf.xall, 
			0.0, sf.xi, sf.rcut);
	}
	if (1) { // full precondition matrix
		std::cout << "Build precondition matrix" << std::endl;
		
		int nrow = sf.nall*3 + sf.part_num*6;
		int ncol = nrow;
		sf.mat_precond = new CSRMat(nrow, ncol);

		stokes_build_precond_matrix_all(sf, *sf.mat_precond);
	}


	if (1) {
		std::cout << "Build distribution matrix" << std::endl;

		int nrow = sf.nall;
		int ncol = nx * ny * nz;
		sf.mat_distrib = new CSRMat(nrow, ncol);

		stokes_build_distrib_forcing_matrix(sf, stokes, *sf.mat_distrib, sf.nall, sf.xall);
	} else {
		sf.mat_distrib = NULL;
	}

	sf.mat_interp = NULL;
	if (1) {
		std::cout << "Build interpolation matrix" << std::endl;

		int nrow = sf.nall;
		int ncol = nx * ny * nz;
		sf.mat_interp = new CSRMat(nrow,ncol);

		stokes_build_interp_matrix(sf, stokes, *sf.mat_interp, sf.nall, sf.xall);
	}
	if (0) {
		sf.mat_interp = sf.mat_distrib;
	}

	// solve boundary force + particle velocity
	//stokes_solve_ib(sf, stokes);
	stokes_solve_all(sf, stokes);

	if (1) { // reconstruct flow field
		std::cout << "Re-Compute global velocity" << std::endl;

		//stokes_solve_forcing(sf, stokes);

		stokes_solve_global(sf, stokes, sf.nall, sf.xall, sf.fall);
	}

	// output
	stokes_output_vtk(stokes, "test_part_g00.vtk");
	//stokes_forcing_output_csv2(sf, "test_part00.csv");
	stokes_forcing_output_vtk(sf, "test_part_p00.vtk");
	stokes_forcing_output_particle_vtk(sf, "test_part_part00.vtk");

	if (0) { // check particle state
		stokes_calc_partforce(sf, sf.fp, sf.part_force);
		for (int ipart=0; ipart<sf.part_num; ipart++) {
			std::cout << "part=" << ipart << std::endl;
			std::cout << "velocity=" << ipart << std::endl;
			std::cout << sf.part_vel[ipart*6+0] << "," << sf.part_vel[ipart*6+1] << "," << sf.part_vel[ipart*6+2] << std::endl;
			std::cout << sf.part_vel[ipart*6+3] << "," << sf.part_vel[ipart*6+4] << "," << sf.part_vel[ipart*6+5] << std::endl;
			std::cout << "force=" << ipart << std::endl;
			std::cout << sf.part_force[ipart*6+0] << "," << sf.part_force[ipart*6+1] << "," << sf.part_force[ipart*6+2] << std::endl;
			std::cout << sf.part_force[ipart*6+3] << "," << sf.part_force[ipart*6+4] << "," << sf.part_force[ipart*6+5] << std::endl;
		}
	}

	if (1) { // check shear force
		double fxbot = 0;
		double fxtop = 0;
		for (int i=0; i<sf.nb; i++) {
			if (sf.idb[i] == WallBot) {
				fxbot += sf.fb[i*3+0];
			}
			if (sf.idb[i] == WallTop) {
				fxtop += sf.fb[i*3+0];
			}
		}

		double areay = (xhi-xlo)*(zhi-zlo); // wall area
		double sxbot = fxbot / areay;
		double sxtop = fxtop / areay;

		std::cout << "sxybot=" << sxbot << std::endl;
		std::cout << "sxytop=" << sxtop << std::endl;
		std::cout << "sxyavg=" << (sxtop-sxbot)/2 << std::endl;
	}

}


void test_stokes_partmove(int argc, char *argv[]) {
	
	//
	StokesFFT stokes;

	// to have 6*pi*mu = 1.0
	//const double mu = 1.0/(6.0*M_PI);
	const double mu = 1.0;

	stokes.dens = 1.0;
	stokes.visc = mu;

	// domain
	const double len = 1.0;
	const double xlo = -len/2, xhi = len/2;
	const double zlo = -len/2, zhi = len/2;
	//const double ylo = -len*2/2, yhi = len*2/2;
	const double ylo = -len*0.75, yhi = len*0.75;
	
	// grid
	const int ncell = 80;
	const int nx = ncell;
	const int nz = ncell;
	//const int ny = ncell*2;
	const int ny = ncell*3/2;
	const double dx = (xhi-xlo) / nx;
	const double dy = (yhi-ylo) / ny;
	const double dz = (zhi-zlo) / nz;

	//
	stokes_set_grid(stokes, xlo, ylo, zlo, xhi, yhi, zhi, nx, ny, nz);
	stokes_init(stokes);
	stokes_init_fft(stokes);

	//
	StokesForcing sf;
	sf.stokes = &stokes;

	// regular factor
	// screen factor
	//const double xi = 1.3 / dx;
	//const double alpha = 0.8 / dx;
	//const double xi = 1.8/dx; // n=80
	//const double alpha = 1.2 / dx; // n=80
	const double xi = 1.8/dx; // n=100
	const double alpha = 1.0 / dx; // n=100
	// cutoff distance
	const double rcut = 4.0 / alpha;

	sf.xi = xi;
	sf.alpha = alpha;
	sf.rcut = rcut;

	// allocate buffer
	const int nallmax = 100000;
	const int partmax = 1000;
	{
		sf.nall = 0;
		sf.xall = new double[nallmax*3];
		sf.fall = new double[nallmax*3];
		sf.uall = new double[nallmax*3];
		sf.idall = new int[nallmax];

		sf.tmp[0] = new double[nallmax];
		sf.tmp[1] = new double[nallmax];
		sf.tmp[2] = new double[nallmax];

		for (int i=0; i<nallmax; i++) {
			sf.idall[i] = 0;
			for (int dir=0; dir<3; dir++) {
				sf.xall[i*3+dir] = 0;
				sf.fall[i*3+dir] = 0;
				sf.uall[i*3+dir] = 0;

				sf.tmp[dir][i] = 0;
			}
		}

		// these will take place in ALL
		sf.nb = 0;
		sf.xb = NULL;
		sf.ub = NULL;
		sf.fb = NULL;
		sf.idb = NULL;
		sf.np = 0;
		sf.xp = NULL;
		sf.up = NULL;
		sf.fp = NULL;
		sf.idp = NULL;

		sf.part_num = 0;
		sf.part_rad = new double[partmax];
		sf.part_pos = new double[partmax*3];
		sf.part_vel = new double[partmax*6];
		sf.part_force = new double[partmax*6];
		sf.part_ipbegin = new int[partmax];
		sf.part_ipend = new int[partmax];
		for (int i=0; i<partmax; i++) {
			sf.part_rad[i] = 0;
			for (int dir=0; dir<3; dir++) {
				sf.part_pos[i*3+dir] = 0;
				sf.part_vel[i*6+dir] = 0;
				sf.part_vel[i*6+3+dir] = 0;
				sf.part_force[i*6+dir] = 0;
				sf.part_force[i*6+3+dir] = 0;
			}
			sf.part_ipbegin[i] = 0;
			sf.part_ipend[i] = 0;
		}
	}


	{ // load particles
		int npart = 0; 

		if (1) {
			if (argc != 3) {
				std::cerr << "Usage: " << __FUNCTION__ << " <input_dem.csv> <npart>" << std::endl;
				exit(1);
			}
			npart = atoi(argv[2]);
			const char *inputfilename = argv[1];
			FILE *fpdem = fopen(inputfilename, "r");
			if (!fpdem) {
				std::cerr << "Failed to open " << inputfilename << std::endl;
				exit(1);
			} else {
				std::cout << "Read " << inputfilename << std::endl;
			}

			const int buflen = 1024;
			char buf[buflen];

			// skip header
			fgets(buf, buflen, fpdem);

			for (int ipart=0; ipart<npart; ipart++) {
				fgets(buf, buflen, fpdem);

				int idx;
				double dpart, xpart, ypart, zpart;
				sscanf(buf, "%d,%lf,%lf,%lf,%lf", 
					&idx, &dpart, &xpart, &ypart, &zpart);

				sf.part_rad[ipart] = dpart * 0.5;
				sf.part_pos[ipart*3+0] = xpart;
				sf.part_pos[ipart*3+1] = ypart;
				sf.part_pos[ipart*3+2] = zpart;
				for (int dir=0; dir<3; dir++) {
					sf.part_vel[ipart*6+dir] = 0; // velocity 
					sf.part_vel[ipart*6+3+dir] = 0; // angular velocity
					sf.part_force[ipart*6+dir] = 0; // force
					sf.part_force[ipart*6+3+dir] = 0; // torque
				}
				sf.part_num += 1;
			}

			fclose(fpdem);
		}

		if (sf.part_num > partmax) {
			std::cerr << "Particle overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Particle=" << sf.part_num << std::endl;
		}
	}


	{ // set boundary points
		// connect buffer
		sf.nb = 0;
		sf.xb = sf.xall + sf.nall*3;
		sf.ub = sf.uall + sf.nall*3;
		sf.fb = sf.fall + sf.nall*3;
		sf.idb = sf.idall + sf.nall;

		const double dh = dx * 2.0; // typical distance between boundary point
		const double gammadot = 1.0; // shear rate

		if (1) { // add top/bottom wall
			//const double ythick = len / 4;
			const double ythick = len / 8;
			const double ytop = len / 2;
			const double utop[3] = { ytop*gammadot, 0.0, 0.0 };
			const double ybot = -len / 2;
			const double ubot[3] = { ybot*gammadot, 0.0, 0.0 };
			
			int nsave = sf.nb;
			gen_plane_y(ybot-ythick, ubot, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallBot2;
			}

			nsave = sf.nb;
			gen_plane_y(ybot, ubot, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallBot;
			}

			nsave = sf.nb;
			gen_plane_y(ytop, utop, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallTop;
			}

			nsave = sf.nb;
			gen_plane_y(ytop+ythick, utop, 
				sf.nb, sf.xb, sf.ub, 
				xlo, ylo, zlo, xhi, yhi, zhi, dh);
			for (int i=nsave; i<sf.nb; i++) {
				sf.idb[i] = WallTop2;
			}
		}

		sf.nall += sf.nb;

		if (sf.nall > nallmax) {
			std::cerr << "Nb overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Nb=" << sf.nb << std::endl;
		}
	}

	{ // generate particle points
		// connect buffer
		sf.np = 0;
		sf.xp = sf.xall + sf.nall*3;
		sf.up = sf.uall + sf.nall*3;
		sf.fp = sf.fall + sf.nall*3;
		sf.idp = sf.idall + sf.nall;
		assert(sf.idp - sf.idb == sf.nb);


		for (int ipart=0; ipart<sf.part_num; ipart++) {
			const double apart = sf.part_rad[ipart];
			const double *xpart = &sf.part_pos[ipart*3];
			const double *upart = &sf.part_vel[ipart*6];
			const double *opart = &sf.part_vel[ipart*6+3];

			const double dh = dx * 2.1; // typical distance between boundary point

			int nsave = sf.np;
			gen_sphere(apart, xpart, upart, opart, 
				sf.np, sf.xp, sf.up, dh);

			// point to particle
			for (int i=nsave; i<sf.np; i++) {
				sf.idp[i] = ipart;
			}
			// particle to point
			sf.part_ipbegin[ipart] = nsave;
			sf.part_ipend[ipart] = sf.np;
		}

		sf.nall += sf.np;

		if (sf.nall > nallmax) {
			std::cerr << "Np overflow" << std::endl;
			exit(1);
		} else {
			std::cout << "Np=" << sf.np << std::endl;
		}
	}

	{
		std::cout << "Nall=" << sf.nall << std::endl;
	}


	{ // cache points
		std::cout << "Alloc cache" << std::endl;

		sf.cache = new PointCache;
		
		PointCache &cache = *sf.cache;

		cache.xmin[0] = xlo;
		cache.xmin[1] = ylo;
		cache.xmin[2] = zlo;
		cache.xmax[0] = xhi;
		cache.xmax[1] = yhi;
		cache.xmax[2] = zhi;

		for (int dir=0; dir<3; dir++) {
			double len = cache.xmax[dir] - cache.xmin[dir];
			int num = (int) (len / rcut);
			if (num < 1) num = 1;

			cache.nbin[dir] = num;
			cache.hbin[dir] = len / num;
		}

		int ncache = cache.nbin[0] * cache.nbin[1] * cache.nbin[2];
		cache.head = new int[ncache];
		cache.tail = new int[ncache];
		cache.next = new int[nallmax];

		// clear cache
		for (int i=0; i<ncache; i++) {
			cache.head[i] = -1;
			cache.tail[i] = -1;
		}
		for (int i=0; i<sf.nall; i++) {
			cache.next[i] = -1;
		}

		std::cout << "ncache=" << cache.nbin[0] << "," << cache.nbin[1] << "," << cache.nbin[2] << std::endl;
		std::cout << "hcache=" << cache.hbin[0] << "," << cache.hbin[1] << "," << cache.hbin[2] << std::endl;

	}

	// allocate matrix
	{
		std::cout << "Alloc matrix" << std::endl;

		sf.mat_local = NULL;
		if (1) {
			int nrow = sf.nall * 3;
			int ncol = nrow;
			sf.mat_local = new CSRMat(nrow, ncol);
		}

		sf.mat_precond = NULL;
		if (1) { // full precondition matrix
			int nrow = sf.nall*3 + sf.part_num*6;
			int ncol = nrow;
			sf.mat_precond = new CSRMat(nrow, ncol);
		}

		sf.mat_distrib = NULL;
		if (1) {
			int nrow = sf.nall;
			int ncol = nx * ny * nz;
			sf.mat_distrib = new CSRMat(nrow, ncol);
		}

		sf.mat_interp = NULL;
		if (1) {
			int nrow = sf.nall;
			int ncol = nx * ny * nz;
			sf.mat_interp = new CSRMat(nrow,ncol);
		}
	}

	

	double time = 0;
	int step = 0;
	const double dt = 1.0e-3;
	const int maxstep = 10000;
	int plot = 0;

	std::ofstream ofs("test_part_stress.csv");
	ofs << "step,time,sbot,stop,savg" << std::endl;
	ofs << step << "," << time << "," << 0.0 << ","<< 0.0 << ","<< 0.0 << std::endl;

	// first output
	{
		char filename[128];
		
		sprintf(filename, "output/test_part_p%04d.vtk", plot);
		stokes_forcing_output_vtk(sf, filename);

		sprintf(filename, "output/test_part_part%04d.vtk", plot);
		stokes_forcing_output_particle_vtk(sf, filename);

		plot += 1;
	}

	for (step=1; step<=maxstep; step++) {
		std::cout << "step=" << step << "; time=" << time << std::endl;
		
		// put points in cache
		{
			std::cout << "Build point cache" << std::endl;

			PointCache &cache = *sf.cache;

			// clear cache
			int ncache = cache.nbin[0] * cache.nbin[1] * cache.nbin[2];
			for (int i=0; i<ncache; i++) {
				cache.head[i] = -1;
				cache.tail[i] = -1;
			}
			for (int i=0; i<sf.nall; i++) {
				cache.next[i] = -1;
			}

			// save into cache
			for (int i=0; i<sf.nall; i++) {
				const double *pos = &sf.xall[i*3];

				int loc[3];
				for (int dir=0; dir<3; dir++) {
					loc[dir] = (int) floor((pos[dir]-cache.xmin[dir]) / cache.hbin[dir]);
					if (loc[dir] < 0) loc[dir] += cache.nbin[dir];
					if (loc[dir] >= cache.nbin[dir]) loc[dir] -= cache.nbin[dir];
				}

				int idx = loc[0]*cache.nbin[1]*cache.nbin[2] + loc[1]*cache.nbin[2] + loc[2];

				int oldtail = cache.tail[idx];
				cache.tail[idx] = i;
				if (oldtail == -1) {
					cache.head[idx] = i;
				} else {
					cache.next[oldtail] = i;
				}
			}
		}

		{ // build matrix

			stokes_build_local_forcing_matrix2(sf, *sf.mat_local, sf.nall, sf.xall, 
				sf.alpha, sf.xi, sf.rcut);

			stokes_build_precond_matrix_all(sf, *sf.mat_precond);

			stokes_build_distrib_forcing_matrix(sf, stokes, *sf.mat_distrib, sf.nall, sf.xall);

			stokes_build_interp_matrix(sf, stokes, *sf.mat_interp, sf.nall, sf.xall);
		}

		{ // set particle external force
			
			// clear external force
			for (int ipart=0; ipart<sf.part_num; ipart++) {
				for (int dir=0; dir<6; dir++) {
					sf.part_force[ipart*6+dir] = 0;
				}
			}

			if (1) {
				const double cont_k = 1.0e2;
				const double cont_dist = 0.05;
				for (int ipart=0; ipart<sf.part_num; ipart++) {
					for (int jpart=ipart+1; jpart<sf.part_num; jpart++) {
						const double irad = sf.part_rad[ipart];
						const double jrad = sf.part_rad[jpart];

						const double *ipos = &sf.part_pos[ipart*3];
						const double *jpos = &sf.part_pos[jpart*3];

						double ex = ipos[0] - jpos[0];
						double ey = ipos[1] - jpos[1];
						double ez = ipos[2] - jpos[2];
						
						double rij = sqrt(ex*ex + ey*ey + ez*ez);
						ex /= rij;
						ey /= rij;
						ez /= rij;

						double hij = rij - irad - jrad;
						double hcont = cont_dist * (irad+jrad)/2.0;
						
						if (hij < hcont) {
							double disp = hcont - hij;
							double fx = disp * cont_k * ex;
							double fy = disp * cont_k * ey;
							double fz = disp * cont_k * ez;

							sf.part_force[ipart*6+0] += fx;
							sf.part_force[ipart*6+1] += fy;
							sf.part_force[ipart*6+2] += fz;

							sf.part_force[jpart*6+0] -= fx;
							sf.part_force[jpart*6+1] -= fy;
							sf.part_force[jpart*6+2] -= fz;
						}

					}
				}
			}
		}

		//
		stokes_solve_all(sf, stokes);

		// move particle
		for (int ipart=0; ipart<sf.part_num; ipart++) {
			double *xpart = &sf.part_pos[ipart*3];
			const double *upart = &sf.part_vel[ipart*6];

			double xold[3] = { xpart[0], xpart[1], xpart[2] };

			for (int dir=0; dir<3; dir++) {
				xpart[dir] += dt * upart[dir];
			}

			if (xpart[0] <  xlo) xpart[0] += (xhi-xlo);
			if (xpart[0] >= xhi) xpart[0] -= (xhi-xlo);
			if (xpart[1] <  ylo) xpart[0] += (yhi-ylo);
			if (xpart[1] >= yhi) xpart[0] -= (yhi-ylo);
			if (xpart[2] <  zlo) xpart[0] += (zhi-zlo);
			if (xpart[2] >= zhi) xpart[0] -= (zhi-zlo);

			// update points
			for (int ip=sf.part_ipbegin[ipart]; ip<sf.part_ipend[ipart]; ip++) {
				for (int dir=0; dir<3; dir++) {
					sf.xp[ip*3+dir] += xpart[dir] - xold[dir];
				}
			}
		}



		time += dt;


		if (step%100 == 0) {
			char filename[128];

			sprintf(filename, "output/test_part_p%04d.vtk", plot);
			stokes_forcing_output_vtk(sf, filename);

			sprintf(filename, "output/test_part_part%04d.vtk", plot);
			stokes_forcing_output_particle_vtk(sf, filename);

			plot += 1;
		}

		if (1) { // check shear force
			double fxbot = 0;
			double fxtop = 0;
			for (int i=0; i<sf.nb; i++) {
				if (sf.idb[i] == WallBot) {
					fxbot += sf.fb[i*3+0];
				}
				if (sf.idb[i] == WallTop) {
					fxtop += sf.fb[i*3+0];
				}
			}

			double areay = (xhi-xlo)*(zhi-zlo); // wall area
			double sxbot = fxbot / areay;
			double sxtop = fxtop / areay;
			double sxavg = (sxtop-sxbot)/2;

			ofs << step << "," << time << "," << sxbot << ","<< sxtop << ","<< sxavg << std::endl;
			ofs.flush();
		}

	}


	
}






