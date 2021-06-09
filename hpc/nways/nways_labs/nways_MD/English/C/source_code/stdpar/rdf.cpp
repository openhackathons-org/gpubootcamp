// Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include "dcdread.h"
#include <assert.h>

#include <algorithm>
#include <vector>
#include <atomic>
#include </opt/nvidia/hpc_sdk/Linux_x86_64/21.3/cuda/11.2/include>

//Note: The addition of execution header file
#include <execution>

#ifdef USE_COUNTING_ITERATOR
#include <thrust/iterator/counting_iterator.h>
#endif

void pair_gpu(double *d_x, double *d_y, double *d_z,
			  std::atomic<int> *d_g2, int numatm, int nconf,
			  const double xbox, const double ybox, const double zbox, int d_bin);

int main(int argc, char *argv[])
{
	double xbox, ybox, zbox;

	int nbin;
	int numatm, nconf, inconf;
	string file;

	///////////////////////////////////////////////////////////////

	inconf = 10;
	nbin = 2000;
	file = "../input/alk.traj.dcd";
	///////////////////////////////////////
	std::ifstream infile;
	infile.open(file.c_str());
	if (!infile)
	{
		cout << "file " << file.c_str() << " not found\n";
		return 1;
	}
	assert(infile);

	ofstream pairfile, stwo;
	pairfile.open("RDF.dat");
	stwo.open("Pair_entropy.dat");

	/////////////////////////////////////////////////////////

	dcdreadhead(&numatm, &nconf, infile);
	cout << "Dcd file has " << numatm << " atoms and " << nconf << " frames" << endl;
	if (inconf > nconf)
		cout << "nconf is reset to " << nconf << endl;
	else
	{
		nconf = inconf;
	}
	cout << "Calculating RDF for " << nconf << " frames" << endl;
	////////////////////////////////////////////////////////

	double *h_x = new double[nconf * numatm];
	double *h_y = new double[nconf * numatm];
	double *h_z = new double[nconf * numatm];

	//Note: We are using standard std atomic which gets mapped to respective atomic operations on GPU

	std::atomic<int> *h_g2 = new std::atomic<int>[nbin];
	std::fill(std::execution::par, h_g2, h_g2 + nbin, 0);

	/////////reading cordinates//////////////////////////////////////////////
	nvtxRangePush("Read_File");
	double ax[numatm], ay[numatm], az[numatm];
	for (int i = 0; i < nconf; i++)
	{
		dcdreadframe(ax, ay, az, infile, numatm, xbox, ybox, zbox);
		for (int j = 0; j < numatm; j++)
		{
			h_x[i * numatm + j] = ax[j];
			h_y[i * numatm + j] = ay[j];
			h_z[i * numatm + j] = az[j];
		}
	}
	nvtxRangePop(); //pop for Reading file
	cout << "Reading of input file is completed" << endl;
	//////////////////////////////////////////////////////////////////////////
	nvtxRangePush("Pair_Calculation");
	pair_gpu(h_x, h_y, h_z, h_g2, numatm, nconf, xbox, ybox, zbox, nbin);
	nvtxRangePop(); //Pop for Pair Calculation

	double pi = acos(-1.0l);
	double rho = (numatm) / (xbox * ybox * zbox);
	double norm = (4.0l * pi * rho) / 3.0l;
	double rl, ru, nideal;
	double g2[nbin];
	double r, gr, lngr, lngrbond, s2 = 0.0l, s2bond = 0.0l;
	double box = min(xbox, ybox);
	box = min(box, zbox);
	double del = box / (2.0l * nbin);
	nvtxRangePush("Entropy_Calculation");
	for (int i = 0; i < nbin; i++)
	{
		//      cout<<i+1<<" "<<h_g2[i]<<endl;
		rl = (i)*del;
		ru = rl + del;
		nideal = norm * (ru * ru * ru - rl * rl * rl);
		g2[i] = (double)h_g2[i] / ((double)nconf * (double)numatm * nideal);
		r = (i)*del;
		pairfile << (i + 0.5l) * del << " " << g2[i] << endl;
		if (r < 2.0l)
		{
			gr = 0.0l;
		}
		else
		{
			gr = g2[i];
		}
		if (gr < 1e-5)
		{
			lngr = 0.0l;
		}
		else
		{
			lngr = log(gr);
		}

		if (g2[i] < 1e-6)
		{
			lngrbond = 0.0l;
		}
		else
		{
			lngrbond = log(g2[i]);
		}
		s2 = s2 - 2.0l * pi * rho * ((gr * lngr) - gr + 1.0l) * del * r * r;
		s2bond = s2bond - 2.0l * pi * rho * ((g2[i] * lngrbond) - g2[i] + 1.0l) * del * r * r;
	}
	nvtxRangePop(); //Pop for Entropy Calculation
	stwo << "s2 value is " << s2 << endl;
	stwo << "s2bond value is " << s2bond << endl;

	cout << "#Freeing Host memory" << endl;
	delete[] h_x;
	delete[] h_y;
	delete[] h_z;
	delete[] h_g2;

	cout << "#Number of atoms processed: " << numatm << endl
		 << endl;
	cout << "#Number of confs processed: " << nconf << endl
		 << endl;
	return 0;
}

int round(float num)
{
	return num < 0 ? num - 0.5 : num + 0.5;
}

void pair_gpu(double *d_x, double *d_y, double *d_z,
			  std::atomic<int> *d_g2, int numatm, int nconf,
			  const double xbox, const double ybox, const double zbox, int d_bin)
{
	double cut;
	double box;
	box = min(xbox, ybox);
	box = min(box, zbox);

	double del = box / (2.0 * d_bin);
	cut = box * 0.5;

#ifndef USE_COUNTING_ITERATOR
	std::vector<unsigned int> indices(numatm * numatm);
	std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });
#endif

	printf("\n %d %d ", nconf, numatm);
	for (int frame = 0; frame < nconf; frame++)
	{
		printf("\n %d  ", frame);
#ifdef USE_COUNTING_ITERATOR
		// Todo : Use the right parallel execution policy and algorithm
		std::Fill parallel algorithm Here(Fill execution policy here, thrust::counting_iterator<unsigned int>(0u), thrust::counting_iterator<unsigned int>(numatm * numatm),
#else
		std::Fill parallel algorithm Here(execution policy here, indices.begin(), indices.end(),
#endif
										  [d_x, d_y, d_z, d_g2, numatm, frame, xbox, ybox, zbox, cut, del](unsigned int index) {
											  int id1 = index / numatm;
											  int id2 = index % numatm;

											  double dx = d_x[frame * numatm + id1] - d_x[frame * numatm + id2];
											  double dy = d_y[frame * numatm + id1] - d_y[frame * numatm + id2];
											  double dz = d_z[frame * numatm + id1] - d_z[frame * numatm + id2];

											  dx = dx - xbox * (round(dx / xbox));
											  dy = dy - ybox * (round(dy / ybox));
											  dz = dz - zbox * (round(dz / zbox));

											  double r = sqrtf(dx * dx + dy * dy + dz * dz);
											  if (r < cut)
											  {
												  int ig2 = (int)(r / del);
												  ++d_g2[ig2];
											  }
										  });
	}
}
