# ifndef ELASTIC_ANI_CUH
# define ELASTIC_ANI_CUH

# include "modeling.cuh"

# define RSGR 2

class Elastic_ANI : public Modeling
{
    float * dwc = nullptr;

    uintc * d_B = nullptr; float maxB; float minB;

    uintc * d_C11 = nullptr; float maxC11; float minC11;
    uintc * d_C13 = nullptr; float maxC13; float minC13;
    uintc * d_C15 = nullptr; float maxC15; float minC15;
    uintc * d_C33 = nullptr; float maxC33; float minC33;
    uintc * d_C35 = nullptr; float maxC35; float minC35;
    uintc * d_C55 = nullptr; float maxC55; float minC55;

    void set_specifications();

    void compute_eikonal();
    void compute_velocity();
    void compute_pressure();
};

__global__ void get_quasi_slowness(float * T, float * S, float dx, float dz, int sIdx, int sIdz, int nxx, int nzz, 
                                   int nb, uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, 
                                   float minC11, float maxC11, float minC13, float maxC13, float minC15, float maxC15, 
                                   float minC33, float maxC33, float minC35, float maxC35, float minC55, float maxC55);

__global__ void compute_velocity_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, float minB, 
                                     float maxB, float * damp1D, float * damp2D, float * wavelet, float * dwc, float dx, float dz, 
                                     float dt, int tId, int tlag, int sIdx, int sIdz, int nxx, int nzz, int nb, int nt);

__global__ void compute_pressure_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                     uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, int tId, 
                                     int tlag, float dx, float dz, float dt, int nxx, int nzz, float minC11, float maxC11, 
                                     float minC13, float maxC13, float minC15, float maxC15, float minC33, float maxC33, 
                                     float minC35, float maxC35, float minC55, float maxC55);

# endif