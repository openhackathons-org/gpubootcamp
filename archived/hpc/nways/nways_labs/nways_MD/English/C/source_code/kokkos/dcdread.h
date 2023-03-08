// Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
using namespace std;

void dcdreadhead(int *natom, int *nframes, std::istream &infile)
{

    infile.seekg(8, ios::beg);
    infile.read((char *)nframes, sizeof(int));
    infile.seekg(64 * 4, ios::cur);
    infile.read((char *)natom, sizeof(int));
    infile.seekg(1 * 8, ios::cur);
    return;
}

void dcdreadframe(double *x, double *y, double *z, std::istream &infile,
                  int natom, double &xbox, double &ybox, double &zbox)
{

    double d[6];
    for (int i = 0; i < 6; i++)
    {
        infile.read((char *)&d[i], sizeof(double));
    }
    xbox = d[0];
    ybox = d[2];
    zbox = d[5];
    float a, b, c;
    infile.seekg(1 * 8, ios::cur);
    for (int i = 0; i < natom; i++)
    {
        infile.read((char *)&a, sizeof(float));
        x[i] = a;
    }
    infile.seekg(1 * 8, ios::cur);
    for (int i = 0; i < natom; i++)
    {
        infile.read((char *)&b, sizeof(float));
        y[i] = b;
    }
    infile.seekg(1 * 8, ios::cur);
    for (int i = 0; i < natom; i++)
    {
        infile.read((char *)&c, sizeof(float));
        z[i] = c;
    }
    infile.seekg(1 * 8, ios::cur);

    return;
}
