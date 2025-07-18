# ifndef ELASTIC_ISO_CUH
# define ELASTIC_ISO_CUH

# include "modeling.cuh"

class Elastic_ISO : public Modeling
{    
    uintc * d_B = nullptr; float maxB; float minB;

    uintc * d_C13 = nullptr; float maxC13; float minC13;
    uintc * d_C55 = nullptr; float maxC55; float minC55;

    void set_specifications();

    void compute_eikonal();
    void compute_velocity();
    void compute_pressure();
};

__global__ void compute_velocity_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, float maxB, float minB, 
                                     float * damp1D, float * damp2D, float * wavelet, float dx, float dz, float dt, int tId, int tlag, int sIdx, 
                                     int sIdz, float * skw, int nxx, int nzz, int nb, int nt);

__global__ void compute_pressure_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, uintc * C55, uintc * C13, float maxC55, 
                                     float minC55, float maxC13, float minC13, int tId, int tlag, float dx, float dz, float dt, int nxx, int nzz);

# endif