#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <cstdio>
#include <iomanip>
#include "dcdread.h"
#include<assert.h>
#include <nvtx3/nvToolsExt.h>

using namespace std;
//Note: CUDA error handling code. This is required as many CUDA calls are async in nature
static void HandleError( cudaError_t err,
		const char *file,
		int line ) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
				file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Note declaration of GPU function by using keyword __global__
__global__ void pair_gpu(const double* d_x, const double* d_y, const double* d_z,  unsigned long long int *d_g2, int numatm, int nconf, 
		const double xbox, const double ybox, const double zbox,  int d_bin,  unsigned long long int bl);

int main(int argc , char* argv[] )
{
	double xbox,ybox,zbox;
	double* h_x,*h_y,*h_z;
	double* d_x,*d_y,*d_z;
	unsigned long long int *h_g2,*d_g2;
	int nbin;
	int nthreads,device;
	int numatm,nconf,inconf;
	unsigned long long int near2;
	string file;

	///////////////////////////////////////////////////////////////

	inconf = 10;
	nbin=2000;
	file = "../input/alk.traj.dcd";
	device = 0;
	nthreads = 128;
	HANDLE_ERROR (cudaSetDevice(device));//pick the device to use
	///////////////////////////////////////
	std::ifstream infile;
	infile.open(file.c_str());
	if(!infile){
		cout<<"file "<<file.c_str()<<" not found\n";
		return 1;
	}
	assert(infile);


	ofstream pairfile,stwo;
	pairfile.open("RDF.dat");
	stwo.open("Pair_entropy.dat");

	/////////////////////////////////////////////////////////
	dcdreadhead(&numatm,&nconf,infile);
	cout<<"Dcd file has "<< numatm << " atoms and " << nconf << " frames"<<endl;
	if (inconf>nconf) cout << "nconf is reset to "<< nconf <<endl;
	else
	{nconf=inconf;}
	cout<<"Calculating RDF for " << nconf << " frames"<<endl;
	////////////////////////////////////////////////////////

	unsigned long long int sizef= nconf*numatm*sizeof(double);
	unsigned long long int sizebin= nbin*sizeof(unsigned long long int);

	//Note:Allocate memory on CPU
	HANDLE_ERROR(cudaHostAlloc((void **)&h_x, sizef, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void **)&h_y, sizef, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void **)&h_z, sizef, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void **)&h_g2, sizebin, cudaHostAllocDefault));

	//Todo: Allocate memory on GPU.
	HANDLE_ERROR(cudaMalloc(); //For d_x
			HANDLE_ERROR(cudaMalloc(); //For d_y
				HANDLE_ERROR(cudaMalloc(); //For d_z
					HANDLE_ERROR(cudaMalloc(); //For d_g2

						HANDLE_ERROR (cudaPeekAtLastError());

						memset(h_g2,0,sizebin);



						/////////reading cordinates//////////////////////////////////////////////
						nvtxRangePush("Read_File");
						double ax[numatm],ay[numatm],az[numatm];
						for (int i=0;i<nconf;i++) {
						dcdreadframe(ax,ay,az,infile,numatm,xbox,ybox,zbox);
						for (int j=0;j<numatm;j++){
						h_x[i*numatm+j]=ax[j];
						h_y[i*numatm+j]=ay[j];
						h_z[i*numatm+j]=az[j];
						}
						}
						nvtxRangePop(); //pop for REading file


						nvtxRangePush("Pair_Calculation");
						//Todo: Copy the data from Host to Device before calculation on GPU
						HANDLE_ERROR(cudaMemcpy(dest, source, ,));
						HANDLE_ERROR(cudaMemcpy(dest, source, , ));
						HANDLE_ERROR(cudaMemcpy(dest, source, , ));
						HANDLE_ERROR(cudaMemcpy(dest, source, , ));

						cout<<"Reading of input file and transfer to gpu is completed"<<endl;
						//////////////////////////////////////////////////////////////////////////

						near2=nthreads*(int(0.5*numatm*(numatm-1)/nthreads)+1);
						unsigned long long int nblock = (near2/nthreads);

						cout<<"Initial blocks are "<<nblock<<" "<<", now changing to ";

						int maxblock=65535;
						int bl;
						int blockloop= int(nblock/maxblock);
						if (blockloop != 0) {
							nblock=maxblock;
						}
						cout<<nblock<<" and will run over "<<(blockloop+1)<<" blockloops"<<endl;

						for (bl=0;bl<(blockloop+1);bl++) {
							//cout <<bl<<endl;
							//Todo: Fill the number of blocks and threads and pass the right device pointers
							pair_gpu<<< , >>> (, , , , numatm, nconf, xbox, ybox, zbox, nbin, bl);

							HANDLE_ERROR (cudaPeekAtLastError());
							HANDLE_ERROR(cudaDeviceSynchronize());
						}

						//Todo: Copy d_ge back from Device to Host
						HANDLE_ERROR(cudaMemcpy(dest, source, , ));

						nvtxRangePop(); //Pop for Pair Calculation

						double pi=acos(-1.0l);
						double rho=(numatm)/(xbox*ybox*zbox);
						double norm=(4.0l*pi*rho)/3.0l;
						double rl,ru,nideal;
						double g2[nbin];
						double r,gr,lngr,lngrbond,s2=0.0l,s2bond=0.0l;
						double box=min(xbox,ybox);
						box=min(box,zbox);
						double del=box/(2.0l*nbin);
						nvtxRangePush("Entropy_Calculation");
						for (int i=0;i<nbin;i++) {
							//      cout<<i+1<<" "<<h_g2[i]<<endl;
							rl=(i)*del;
							ru=rl+del;
							nideal=norm*(ru*ru*ru-rl*rl*rl);
							g2[i]=(double)h_g2[i]/((double)nconf*(double)numatm*nideal);
							r=(i)*del;
							pairfile<<(i+0.5l)*del<<" "<<g2[i]<<endl;
							if (r<2.0l) {
								gr=0.0l;
							}
							else {
								gr=g2[i];
							}
							if (gr<1e-5) {
								lngr=0.0l;
							}
							else {
								lngr=log(gr);
							}

							if (g2[i]<1e-6) {
								lngrbond=0.0l;
							}
							else {
								lngrbond=log(g2[i]);
							}
							s2=s2-2.0l*pi*rho*((gr*lngr)-gr+1.0l)*del*r*r;
							s2bond=s2bond-2.0l*pi*rho*((g2[i]*lngrbond)-g2[i]+1.0l)*del*r*r;

						}
						nvtxRangePush("Entropy_Calculation");
						stwo<<"s2 value is "<<s2<<endl;
						stwo<<"s2bond value is "<<s2bond<<endl;



						//Note: Freeing up the GPU memory
						cout<<"\n\n\n#Freeing Device memory"<<endl;
						HANDLE_ERROR(cudaFree(d_x));
						HANDLE_ERROR(cudaFree(d_y));
						HANDLE_ERROR(cudaFree(d_z));
						HANDLE_ERROR(cudaFree(d_g2));

						cout<<"#Freeing Host memory"<<endl;
						HANDLE_ERROR(cudaFreeHost ( h_x ) );
						HANDLE_ERROR(cudaFreeHost ( h_y ) );
						HANDLE_ERROR(cudaFreeHost ( h_z ) );
						HANDLE_ERROR(cudaFreeHost ( h_g2 ) );

						cout<<"#Number of atoms processed: "<<numatm<<endl<<endl;
						cout<<"#Number of confs processed: "<<nconf<<endl<<endl;
						cout<<"#number of threads used: "<<nthreads<<endl<<endl;
						return 0;
}

//Todo: Convert the call to GPU call by adding right keyword
void pair_gpu(
		const double* d_x, const double* d_y, const double* d_z, 
		unsigned long long int *d_g2, int numatm, int nconf, 
		const double xbox,const double ybox,const double zbox,int d_bin,  unsigned long long int bl)
{
	double r,cut,dx,dy,dz;
	int ig2,id1,id2;
	double box;
	box=min(xbox,ybox);
	box=min(box,zbox);

	double del=box/(2.0*d_bin);
	cut=box*0.5;
	int thisi;
	double n;

	//Todo: Write indexing logic using threads and blocks
	int i = 


		int maxi = min(int(0.5*numatm*(numatm-1)-(bl*65535*128)),(65535*128));

	if ( i < maxi ) {
		thisi=bl*65535*128+i;

		n=(0.5)*(1+ ((double) sqrt (1.0+4.0*2.0*thisi)));
		id1=int(n);
		id2=thisi-(0.5*id1*(id1-1));

		for (int frame=0;frame<nconf;frame++){
			dx=d_x[frame*numatm+id1]-d_x[frame*numatm+id2];
			dy=d_y[frame*numatm+id1]-d_y[frame*numatm+id2];
			dz=d_z[frame*numatm+id1]-d_z[frame*numatm+id2];

			dx=dx-xbox*(round(dx/xbox));
			dy=dy-ybox*(round(dy/ybox));
			dz=dz-zbox*(round(dz/zbox));

			r=sqrtf(dx*dx+dy*dy+dz*dz);
			if (r<cut) {
				ig2=(int)(r/del);
				//Note: Usage of atomic function
				atomicAdd(&d_g2[ig2],2) ;
			}
		}
	}
}


