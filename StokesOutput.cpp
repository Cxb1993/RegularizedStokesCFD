
#include <cstdlib>
#include <cstdio>

#include <iostream>

#include "StokesFFT.h"
#include "StokesIB.h"


template<typename T>
void writeBigEndian(FILE* fp,const T* data,size_t counts)
{
	const char* p=(const char*)data;

	const size_t buflen = 256;
	char buffer[sizeof(T)*buflen];

	for(size_t written=0; written<counts; written+=buflen){
		size_t toWrite = std::min(buflen,counts-written);

		for(int i=0;i<toWrite;i++){
			for(int j=0;j<sizeof(T);j++){
				//swap bytes
				buffer[i*sizeof(T) + j] = p[sizeof(T)*(i+1) - j - 1];
			}
		}

		fwrite(buffer,sizeof(T) * toWrite,1,fp);
	}
}


void stokes_output_vtk(StokesFFT &stokes, const char *filename) {

	FILE *fp = fopen(filename, "wb");
	if(!fp){
		std::cerr << __FUNCTION__ << ": Failed to write " << filename << std::endl;
		return;
	}

	const int nx = stokes.cellNum[0];
	const int ny = stokes.cellNum[1];
	const int nz = stokes.cellNum[2];
	const double dx = stokes.cellSize[0];
	const double dy = stokes.cellSize[1];
	const double dz = stokes.cellSize[2];
	const double xlo = stokes.problo[0];
	const double ylo = stokes.problo[1];
	const double zlo = stokes.problo[2];

	//header
	fprintf(fp,"# vtk DataFile Version 2.0\n");
	//title
	fprintf(fp,"Fluid Data\n");
	//type
	fprintf(fp,"BINARY\n");
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	
	//xyz
	fprintf(fp,"DIMENSIONS %d %d %d\n", nx+1, ny+1, nz+1);

	fprintf(fp,"X_COORDINATES %d float\n",nx+1);
	for( int i = 0; i <= nx; i++ ){
		float f = dx * i + xlo;
		writeBigEndian(fp,&f,1);
	}
	fprintf(fp,"\n");

	fprintf(fp,"Y_COORDINATES %d float\n",ny+1);
	for( int i = 0; i <= ny; i++ ){
		float f = dy * i + ylo;
		writeBigEndian(fp,&f,1);
	}
	fprintf(fp,"\n");
	fprintf(fp,"Z_COORDINATES %d float\n",nz+1);
	for( int i = 0; i <= nz; i++ ){
		float f=dz * i + zlo;
		writeBigEndian(fp,&f,1);
	}
	fprintf(fp,"\n");

	//time
	fprintf(fp,"FIELD FieldData 1\n");
	fprintf(fp,"TIME 1 1 float\n");
	{
		float tt = 0;
		writeBigEndian(fp,&tt,1);
	}

	//fprintf(fp,"CELL_DATA %d\n",grid->x*grid->y*grid->z);

	//std::vector<float> tmpArray(grid->x);

	////P
	//if (1) {
	//	write_cell_scalar(fp, "pressure", fd->p, grid);
	//} else {
	//	fprintf(fp,"SCALARS pressure float 1\n");
	//	fprintf(fp,"LOOKUP_TABLE default\n");
	//	for( int k = 1; k <= grid->z; k++ ){
	//		for( int j = 1; j <= grid->y; j++ ){
	//			for( int i = 1; i <= grid->x; i++ ){
	//				//writeBigEndianCastFloat(fp,&fd->p[i][j][k],1);
	//				tmpArray[i-1]=fd->p[i][j][k];
	//			}
	//			writeBigEndian(fp,&tmpArray[0],tmpArray.size());
	//		}
	//	}
	//	fprintf(fp,"\n");
	//}


	fprintf(fp, "POINT_DATA %d\n", (nx+1)*(ny+1)*(nz+1));

	// u,v,w
	fprintf(fp,"VECTORS velocity float\n");
	for (int k=0; k<=nz; k++) {
		for (int j=0; j<=ny; j++) {
			for (int i=0; i<=nx; i++) {
				int imap = i<nx ? i : 0;
				int jmap = j<ny ? j : 0;
				int kmap = k<nz ? k : 0;
				int idx = imap*ny*nz + jmap*nz + kmap;

				float data[] = { stokes.u[idx], stokes.v[idx], stokes.w[idx] };
				writeBigEndian(fp, data, 3);
			}
		}
	}
	fprintf(fp,"\n");

	// p
	fprintf(fp,"SCALARS pressure float 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (int k=0; k<=nz; k++) {
		for (int j=0; j<=ny; j++) {
			for (int i=0; i<=nx; i++) {
				int imap = i<nx ? i : 0;
				int jmap = j<ny ? j : 0;
				int kmap = k<nz ? k : 0;
				int idx = imap*ny*nz + jmap*nz + kmap;

				float data[] = { stokes.p[idx] };
				writeBigEndian(fp, data, 1);
			}
		}
	}
	fprintf(fp,"\n");


	fclose(fp);
	std::cout << __FUNCTION__ << ": Saved " << filename << std::endl;
}


void stokes_forcing_output_csv(
	const StokesForcing &sf,
	const char outfilename[]) 
{
	FILE *fp = fopen(outfilename, "w");
	if (!fp) {
		std::cerr << "Failed to write " << outfilename << std::endl;
		exit(1);
	}

	fprintf(fp, "x,y,z,u,v,w,fx,fy,fz\n");

	for (int i=0; i<sf.nb; i++) {
		int i0 = i * 3;
		int i1 = i0 + 1;
		int i2 = i0 + 2;

		fprintf(fp, "%lf,%lf,%lf" ",%lf,%lf,%lf" ",%lf,%lf,%lf" "\n",
			sf.xb[i0],sf.xb[i1],sf.xb[i2],
			sf.ub[i0],sf.ub[i1],sf.ub[i2],
			sf.fb[i0],sf.fb[i1],sf.fb[i2]);
	}


	fclose(fp);
	std::cout << __FUNCTION__ << ": Saved " << outfilename << std::endl;
}

void stokes_forcing_output_csv2(
	const StokesForcing &sf,
	const char outfilename[]) 
{
	FILE *fp = fopen(outfilename, "w");
	if (!fp) {
		std::cerr << "Failed to write " << outfilename << std::endl;
		exit(1);
	}

	fprintf(fp, "x,y,z,u,v,w,fx,fy,fz,type,owner\n");

	for (int i=0; i<sf.nall; i++) {
		int i0 = i * 3;
		int i1 = i0 + 1;
		int i2 = i0 + 2;

		int type = i<sf.nb ? 0 : 1;
		int owner = sf.idall[i];

		fprintf(fp, "%lf,%lf,%lf" ",%lf,%lf,%lf" ",%lf,%lf,%lf" ",%d,%d" "\n",
			sf.xall[i0],sf.xall[i1],sf.xall[i2],
			sf.uall[i0],sf.uall[i1],sf.uall[i2],
			sf.fall[i0],sf.fall[i1],sf.fall[i2],
			type, owner);
	}


	fclose(fp);
	std::cout << __FUNCTION__ << ": Saved " << outfilename << std::endl;
}


void stokes_forcing_output_vtk(
	const StokesForcing &sf,
	const char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if(!fp){
		std::cerr << __FUNCTION__ << ": Failed to write " << filename << std::endl;
		return;
	}

	const int nall = sf.nall;
	const double *xall = sf.xall;
	const double *uall = sf.uall;
	const double *fall = sf.fall;
	const int *owner = sf.idall;

	//header
	fprintf(fp,"# vtk DataFile Version 2.0\n");
	//title
	fprintf(fp,"Point Data\n");
	//type
	fprintf(fp,"BINARY\n");
	fprintf(fp,"DATASET UNSTRUCTURED_GRID\n");

	//time
	fprintf(fp,"FIELD FieldData 1\n");
	fprintf(fp,"TIME 1 1 float\n");
	{
		float tt = 0;
		writeBigEndian(fp,&tt,1);
	}

	// positition
	fprintf(fp, "POINTS %d float\n", nall);
	for (int i=0; i<nall; i++) {
		float tmp[3] = { xall[i*3+0], xall[i*3+1], xall[i*3+2] };
		writeBigEndian(fp, tmp, 3);
	}
	fprintf(fp, "\n");

	// cell
	fprintf(fp, "CELLS %d %d\n", nall, nall*2);
	for (int i=0; i<nall; i++) {
		int tmp[] = { 1, i };
		writeBigEndian(fp, tmp, 2);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", nall);
	for (int i=0; i<nall; i++) {
		int tmp[] = { 2 };
		writeBigEndian(fp, tmp, 1);
	}
	fprintf(fp, "\n");


	//fprintf(fp,"CELL_DATA %d\n",grid->x*grid->y*grid->z);

	//std::vector<float> tmpArray(grid->x);

	////P
	//if (1) {
	//	write_cell_scalar(fp, "pressure", fd->p, grid);
	//} else {
	//	fprintf(fp,"SCALARS pressure float 1\n");
	//	fprintf(fp,"LOOKUP_TABLE default\n");
	//	for( int k = 1; k <= grid->z; k++ ){
	//		for( int j = 1; j <= grid->y; j++ ){
	//			for( int i = 1; i <= grid->x; i++ ){
	//				//writeBigEndianCastFloat(fp,&fd->p[i][j][k],1);
	//				tmpArray[i-1]=fd->p[i][j][k];
	//			}
	//			writeBigEndian(fp,&tmpArray[0],tmpArray.size());
	//		}
	//	}
	//	fprintf(fp,"\n");
	//}

	//
	fprintf(fp, "POINT_DATA %d\n", nall);

	// type
	fprintf(fp,"SCALARS type int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (int i=0; i<nall; i++) {
		int tmp[] = { (i<sf.nb ? 0 : 1) };
		writeBigEndian(fp, tmp, 1);
	}
	fprintf(fp,"\n");

	// owner
	fprintf(fp,"SCALARS owner int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (int i=0; i<nall; i++) {
		int tmp[] = { owner[i] };
		writeBigEndian(fp, tmp, 1);
	}
	fprintf(fp,"\n");

	// velocity
	fprintf(fp,"VECTORS velocity float\n");
	for (int i=0; i<nall; i++) {
		float tmp[] = { uall[i*3+0], uall[i*3+1], uall[i*3+2] };
		writeBigEndian(fp, tmp, 3);
	}
	fprintf(fp,"\n");

	// forcing
	fprintf(fp,"VECTORS force float\n");
	for (int i=0; i<nall; i++) {
		float tmp[] = { fall[i*3+0], fall[i*3+1], fall[i*3+2] };
		writeBigEndian(fp, tmp, 3);
	}
	fprintf(fp,"\n");


	fclose(fp);
	std::cout << __FUNCTION__ << ": Saved " << filename << std::endl;

}


void stokes_forcing_output_particle_vtk(
	const StokesForcing &sf,
	const char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if(!fp){
		std::cerr << __FUNCTION__ << ": Failed to write " << filename << std::endl;
		return;
	}

	const int nall = sf.part_num;
	const double *xall = sf.part_pos;
	const double *rall = sf.part_rad;
	const double *uall = sf.part_vel;

	//header
	fprintf(fp,"# vtk DataFile Version 2.0\n");
	//title
	fprintf(fp,"Point Data\n");
	//type
	fprintf(fp,"BINARY\n");
	fprintf(fp,"DATASET UNSTRUCTURED_GRID\n");

	//time
	fprintf(fp,"FIELD FieldData 1\n");
	fprintf(fp,"TIME 1 1 float\n");
	{
		float tt = 0;
		writeBigEndian(fp,&tt,1);
	}

	// positition
	fprintf(fp, "POINTS %d float\n", nall);
	for (int i=0; i<nall; i++) {
		float tmp[3] = { xall[i*3+0], xall[i*3+1], xall[i*3+2] };
		writeBigEndian(fp, tmp, 3);
	}
	fprintf(fp, "\n");

	// cell
	fprintf(fp, "CELLS %d %d\n", nall, nall*2);
	for (int i=0; i<nall; i++) {
		int tmp[] = { 1, i };
		writeBigEndian(fp, tmp, 2);
	}
	fprintf(fp, "\n");

	fprintf(fp, "CELL_TYPES %d\n", nall);
	for (int i=0; i<nall; i++) {
		int tmp[] = { 2 };
		writeBigEndian(fp, tmp, 1);
	}
	fprintf(fp, "\n");


	//fprintf(fp,"CELL_DATA %d\n",grid->x*grid->y*grid->z);

	//std::vector<float> tmpArray(grid->x);

	////P
	//if (1) {
	//	write_cell_scalar(fp, "pressure", fd->p, grid);
	//} else {
	//	fprintf(fp,"SCALARS pressure float 1\n");
	//	fprintf(fp,"LOOKUP_TABLE default\n");
	//	for( int k = 1; k <= grid->z; k++ ){
	//		for( int j = 1; j <= grid->y; j++ ){
	//			for( int i = 1; i <= grid->x; i++ ){
	//				//writeBigEndianCastFloat(fp,&fd->p[i][j][k],1);
	//				tmpArray[i-1]=fd->p[i][j][k];
	//			}
	//			writeBigEndian(fp,&tmpArray[0],tmpArray.size());
	//		}
	//	}
	//	fprintf(fp,"\n");
	//}

	//
	fprintf(fp, "POINT_DATA %d\n", nall);

	// radius
	fprintf(fp,"SCALARS radius float 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (int i=0; i<nall; i++) {
		float tmp[] = { rall[i] };
		writeBigEndian(fp, tmp, 1);
	}
	fprintf(fp,"\n");

	// velocity
	fprintf(fp,"VECTORS velocity float\n");
	for (int i=0; i<nall; i++) {
		float tmp[] = { uall[i*6+0], uall[i*6+1], uall[i*6+2] };
		writeBigEndian(fp, tmp, 3);
	}
	fprintf(fp,"\n");

	// angular velocity
	fprintf(fp,"VECTORS angvel float\n");
	for (int i=0; i<nall; i++) {
		float tmp[] = { uall[i*6+3], uall[i*6+4], uall[i*6+5] };
		writeBigEndian(fp, tmp, 3);
	}
	fprintf(fp,"\n");



	fclose(fp);
	std::cout << __FUNCTION__ << ": Saved " << filename << std::endl;

}



