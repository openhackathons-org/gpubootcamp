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
#include <Kokkos_Core.hpp>
#include <nvToolsExt.h>

int l_round(float num);

typedef Kokkos::View<double *> view_type_double;
typedef Kokkos::View<unsigned long long int *> view_type_long;
typedef view_type_double::HostMirror host_view_type_double;
typedef view_type_long::HostMirror host_view_type_long;

void pair_gpu(view_type_double d_x, view_type_double d_y, view_type_double d_z,
			  view_type_long d_g2, int numatm, int nconf,
			  const double xbox, const double ybox, const double zbox,
			  int d_bin);

int main(int argc, char *argv[])
{
	Kokkos::initialize(argc, argv);
	{

		printf("Default  Kokkos execution space %s\n",
			   typeid(Kokkos::DefaultExecutionSpace).name());

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

		view_type_double x("x", nconf * numatm);
		view_type_double y("y", nconf * numatm);
		view_type_double z("z", nconf * numatm);
		view_type_long g2("g2", nbin);

		host_view_type_double h_x = Kokkos::create_mirror_view(x);
		host_view_type_double h_y = Kokkos::create_mirror_view(y);
		host_view_type_double h_z = Kokkos::create_mirror_view(z);
		host_view_type_long h_g2 = Kokkos::create_mirror_view(g2);

		/////////reading cordinates//////////////////////////////////////////////
		nvtxRangePush("Read_File");
		double ax[numatm], ay[numatm], az[numatm];
		for (int i = 0; i < nconf; i++)
		{
			dcdreadframe(ax, ay, az, infile, numatm, xbox, ybox, zbox);
			for (int j = 0; j < numatm; j++)
			{
				h_x(i * numatm + j) = ax[j];
				h_y(i * numatm + j) = ay[j];
				h_z(i * numatm + j) = az[j];
			}
		}
		for (int i = 0; i < nbin; i++)
			h_g2(0) = 0;

		nvtxRangePop(); //pop for Reading file
		cout << "Reading of input file is completed" << endl;

		nvtxRangePush("Pair_Calculation");
		Kokkos::deep_copy(x, h_x);
		Kokkos::deep_copy(y, h_y);
		Kokkos::deep_copy(z, h_z);
		Kokkos::deep_copy(g2, h_g2);
		//////////////////////////////////////////////////////////////////////////
		pair_gpu(x, y, z, g2, numatm, nconf, xbox, ybox, zbox, nbin);
		Kokkos::deep_copy(h_g2, g2);
		nvtxRangePop(); //Pop for Pair Calculation
		double pi = acos(-1.0l);
		double rho = (numatm) / (xbox * ybox * zbox);
		double norm = (4.0l * pi * rho) / 3.0l;
		double rl, ru, nideal;
		double t_g2[nbin];
		double r, gr, lngr, lngrbond, s2 = 0.0l, s2bond = 0.0l;
		double box = min(xbox, ybox);
		box = min(box, zbox);
		double del = box / (2.0l * nbin);
		nvtxRangePush("Entropy_Calculation");
		for (int i = 0; i < nbin; i++)
		{
			rl = (i)*del;
			ru = rl + del;
			nideal = norm * (ru * ru * ru - rl * rl * rl);
			t_g2[i] = (double)h_g2(i) / ((double)nconf * (double)numatm * nideal);
			r = (i)*del;
			pairfile << (i + 0.5l) * del << " " << t_g2[i] << endl;
			if (r < 2.0l)
			{
				gr = 0.0l;
			}
			else
			{
				gr = t_g2[i];
			}
			if (gr < 1e-5)
			{
				lngr = 0.0l;
			}
			else
			{
				lngr = log(gr);
			}

			if (t_g2[i] < 1e-6)
			{
				lngrbond = 0.0l;
			}
			else
			{
				lngrbond = log(t_g2[i]);
			}
			s2 = s2 - 2.0l * pi * rho * ((gr * lngr) - gr + 1.0l) * del * r * r;
			s2bond = s2bond - 2.0l * pi * rho * ((t_g2[i] * lngrbond) - t_g2[i] + 1.0l) * del * r * r;
		}
		nvtxRangePop(); //Pop for Entropy Calculation
		stwo << "s2 value is " << s2 << endl;
		stwo << "s2bond value is " << s2bond << endl;

		cout << "#Freeing Host memory" << endl;

		cout << "#Number of atoms processed: " << numatm << endl
			 << endl;
		cout << "#Number of confs processed: " << nconf << endl
			 << endl;

	} // Kokkos Initialize ends here
	Kokkos::finalize();
	return 0;
}
int l_round(float num)
{
	return num < 0 ? num - 0.5 : num + 0.5;
}

void pair_gpu(view_type_double d_x, view_type_double d_y, view_type_double d_z,
			  view_type_long d_g2, int numatm, int nconf,
			  const double xbox, const double ybox, const double zbox,
			  int d_bin)
{

	printf("\n %d %d ", nconf, numatm);
	for (int frame = 0; frame < nconf; frame++)
	{
		printf("\n %d  ", frame);
		Kokkos::parallel_for((numatm * numatm), KOKKOS_LAMBDA(const int index) {
			int id1 = index / numatm;
			int id2 = index % numatm;
			double r, cut, dx, dy, dz;
			int ig2;
			double box;
			int myround;
			float num;
			box = min(xbox, ybox);
			box = min(box, zbox);
			double del = box / (2.0 * d_bin);
			cut = box * 0.5;

			dx = d_x(frame * numatm + id1) - d_x(frame * numatm + id2);
			dy = d_y(frame * numatm + id1) - d_y(frame * numatm + id2);
			dz = d_z(frame * numatm + id1) - d_z(frame * numatm + id2);

			num = dx / xbox;
			myround = num < 0 ? num - 0.5 : num + 0.5;
			dx = dx - xbox * myround;

			num = dy / ybox;
			myround = num < 0 ? num - 0.5 : num + 0.5;
			dy = dy - ybox * myround;

			num = dz / zbox;
			myround = num < 0 ? num - 0.5 : num + 0.5;
			dz = dz - zbox * myround;
			r = sqrtf(dx * dx + dy * dy + dz * dz);
			if (r < cut)
			{
				ig2 = (int)(r / del);
				Kokkos::atomic_increment(&d_g2(ig2));
			}
		});
	}
}
