# include "elastic_iso.cuh"

void Elastic_ISO::set_specifications()
{
    modeling_type = "elastic_iso";
    modeling_name = "Modeling type: Elastic isotropic solver";

    auto * Cij = new float[nPoints]();

    std::string vp_file = catch_parameter("vp_model_file", parameters);
    std::string ro_file = catch_parameter("ro_model_file", parameters);
    std::string Cijkl_folder = catch_parameter("Cijkl_folder", parameters);

    float * S = new float[matsize]();
    import_binary_float(vp_file, Cij, nPoints);
    expand_boundary(Cij, S);

    # pragma omp parallel for
    for (int index = 0; index < matsize; index++)
        S[index] = 1.0f / S[index];

    cudaMalloc((void**)&(d_S), matsize*sizeof(float));
    cudaMemcpy(d_S, S, matsize*sizeof(float), cudaMemcpyHostToDevice);
    delete[] S;

    auto * B = new float[matsize]();
    auto * uB = new uintc[matsize]();
    import_binary_float(ro_file, Cij, nPoints);
    expand_boundary(Cij, B);

    # pragma omp parallel for
    for (int index = 0; index < matsize; index++)
        B[index] = 1.0f / B[index];

    compression(B, uB, matsize, maxB, minB);    
    cudaMalloc((void**)&(d_B), matsize*sizeof(uintc));
    cudaMemcpy(d_B, uB, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] B;
    delete[] uB;

    auto * C13 = new float[matsize]();
    auto * uC13 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C13.bin", Cij, nPoints);
    expand_boundary(Cij, C13);
    compression(C13, uC13, matsize, maxC13, minC13);    
    cudaMalloc((void**)&(d_C13), matsize*sizeof(uintc));
    cudaMemcpy(d_C13, uC13, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C13;
    delete[] uC13;

    auto * C55 = new float[matsize]();
    auto * uC55 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C55.bin", Cij, nPoints);
    expand_boundary(Cij, C55);
    compression(C55, uC55, matsize, maxC55, minC55);    
    cudaMalloc((void**)&(d_C55), matsize*sizeof(uintc));
    cudaMemcpy(d_C55, uC55, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C55;
    delete[] uC55;
}

void Elastic_ISO::compute_eikonal()
{
    dim3 grid(1,1,1);
    dim3 block(MESHDIM+1,MESHDIM+1,1);

    time_set<<<nBlocks,NTHREADS>>>(d_T, matsize);
    time_init<<<grid,block>>>(d_T,d_S,sx,sz,dx,dz,sIdx,sIdz,nzz,nb);
    eikonal_solver();
}

void Elastic_ISO::compute_velocity()
{
    compute_velocity_ssg<<<nBlocks,NTHREADS>>>(d_Vx, d_Vz, d_Txx, d_Tzz, d_Txz, d_T, d_B, maxB, minB, d1D, d2D, 
                                               d_wavelet, dx, dz, dt, timeId, tlag, sIdx, sIdz, d_skw, nxx, nzz, nb, nt);
}

void Elastic_ISO::compute_pressure()
{
    compute_pressure_ssg<<<nBlocks,NTHREADS>>>(d_Vx, d_Vz, d_Txx, d_Tzz, d_Txz, d_P, d_T, d_C55, d_C13, maxC55, 
                                               minC55, maxC13, minC13, timeId, tlag, dx, dz, dt, nxx, nzz);    
}

__global__ void compute_velocity_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, float maxB, float minB, 
                                     float * damp1D, float * damp2D, float * wavelet, float dx, float dz, float dt, int tId, int tlag, int sIdx, 
                                     int sIdz, float * skw, int nxx, int nzz, int nb, int nt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float Bn, Bm;

    if ((index == 0) && (tId < nt))
    {
        for (int i = 0; i < KR; i++)
        {
            int zi = sIdz + i - 1;
            for (int j = 0; j < KR; j++)
            {
                int xi = sIdx + j - 1;

                Txx[zi + xi*nzz] += skw[i + j*KR]*wavelet[tId] / (dx*dz);
                Tzz[zi + xi*nzz] += skw[i + j*KR]*wavelet[tId] / (dx*dz);
            }
        }

        // Txx[sIdz + sIdx*nzz] += wavelet[tId] / (dx*dz);
        // Tzz[sIdz + sIdx*nzz] += wavelet[tId] / (dx*dz);
    }

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {
        Bn = (minB + (static_cast<float>(B[index]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

        if((i >= 3) && (i < nzz-4) && (j > 3) && (j < nxx-3)) 
        {
            float dTxx_dx = (FDM1*(Txx[i + (j-4)*nzz] - Txx[i + (j+3)*nzz]) +
                             FDM2*(Txx[i + (j+2)*nzz] - Txx[i + (j-3)*nzz]) +
                             FDM3*(Txx[i + (j-2)*nzz] - Txx[i + (j+1)*nzz]) +
                             FDM4*(Txx[i + j*nzz]     - Txx[i + (j-1)*nzz])) / dx;

            float dTxz_dz = (FDM1*(Txz[(i-3) + j*nzz] - Txz[(i+4) + j*nzz]) +
                             FDM2*(Txz[(i+3) + j*nzz] - Txz[(i-2) + j*nzz]) +
                             FDM3*(Txz[(i-1) + j*nzz] - Txz[(i+2) + j*nzz]) +
                             FDM4*(Txz[(i+1) + j*nzz] - Txz[i + j*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[i + (j+1)*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bx = 0.5f*(Bn + Bm);

            Vx[index] += dt*Bx*(dTxx_dx + dTxz_dz); 
        }

        if((i > 3) && (i < nzz-3) && (j >= 3) && (j < nxx-4)) 
        {
            float dTxz_dx = (FDM1*(Txz[i + (j-3)*nzz] - Txz[i + (j+4)*nzz]) +
                             FDM2*(Txz[i + (j+3)*nzz] - Txz[i + (j-2)*nzz]) +
                             FDM3*(Txz[i + (j-1)*nzz] - Txz[i + (j+2)*nzz]) +
                             FDM4*(Txz[i + (j+1)*nzz] - Txz[i + j*nzz])) / dx;

            float dTzz_dz = (FDM1*(Tzz[(i-4) + j*nzz] - Tzz[(i+3) + j*nzz]) +
                             FDM2*(Tzz[(i+2) + j*nzz] - Tzz[(i-3) + j*nzz]) +
                             FDM3*(Tzz[(i-2) + j*nzz] - Tzz[(i+1) + j*nzz]) +
                             FDM4*(Tzz[i + j*nzz]     - Tzz[(i-1) + j*nzz])) / dz;

            Bm = (minB + (static_cast<float>(B[(i+1) + j*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));

            float Bz = 0.5f*(Bn + Bm);

            Vz[index] += dt*Bz*(dTxz_dx + dTzz_dz); 
        }

    	float damper = get_boundary_damper(damp1D, damp2D, i, j, nxx, nzz, nb);

        Vx[index] *= damper;
        Vz[index] *= damper;

        Txx[index] *= damper;
        Tzz[index] *= damper;
        Txz[index] *= damper;
    }
}

__global__ void compute_pressure_ssg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, uintc * C44, uintc * C13, float maxC55, 
                                     float minC55, float maxC13, float minC13, int tId, int tlag, float dx, float dz, float dt, int nxx, int nzz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float c44_1, c44_2, c44_3, c44_4;

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4)) 
        {    
            float dVx_dx = (FDM1*(Vx[i + (j-3)*nzz] - Vx[i + (j+4)*nzz]) +
                            FDM2*(Vx[i + (j+3)*nzz] - Vx[i + (j-2)*nzz]) +
                            FDM3*(Vx[i + (j-1)*nzz] - Vx[i + (j+2)*nzz]) +
                            FDM4*(Vx[i + (j+1)*nzz] - Vx[i + j*nzz])) / dx;

            float dVz_dz = (FDM1*(Vz[(i-3) + j*nzz] - Vz[(i+4) + j*nzz]) +
                            FDM2*(Vz[(i+3) + j*nzz] - Vz[(i-2) + j*nzz]) +
                            FDM3*(Vz[(i-1) + j*nzz] - Vz[(i+2) + j*nzz]) +
                            FDM4*(Vz[(i+1) + j*nzz] - Vz[i + j*nzz])) / dz;
            
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c44 = (minC55 + (static_cast<float>(C44[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            Txx[index] += dt*((c13 + 2*c44)*dVx_dx + c13*dVz_dz);
            Tzz[index] += dt*((c13 + 2*c44)*dVz_dz + c13*dVx_dx);                    
        }

        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3)) 
        {
            float dVx_dz = (FDM1*(Vx[(i-4) + j*nzz] - Vx[(i+3) + j*nzz]) +
                            FDM2*(Vx[(i+2) + j*nzz] - Vx[(i-3) + j*nzz]) +
                            FDM3*(Vx[(i-2) + j*nzz] - Vx[(i+1) + j*nzz]) +
                            FDM4*(Vx[i + j*nzz]     - Vx[(i-1) + j*nzz])) / dz;

            float dVz_dx = (FDM1*(Vz[i + (j-4)*nzz] - Vz[i + (j+3)*nzz]) +
                            FDM2*(Vz[i + (j+2)*nzz] - Vz[i + (j-3)*nzz]) +
                            FDM3*(Vz[i + (j-2)*nzz] - Vz[i + (j+1)*nzz]) +
                            FDM4*(Vz[i + j*nzz]     - Vz[i + (j-1)*nzz])) / dx;

            c44_1 = (minC55 + (static_cast<float>(C44[(i+1) + (j+1)*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c44_2 = (minC55 + (static_cast<float>(C44[i + (j+1)*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c44_3 = (minC55 + (static_cast<float>(C44[(i+1) + j*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
            c44_4 = (minC55 + (static_cast<float>(C44[i + j*nzz]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            float Mxz = powf(0.25f*(1.0f/c44_1 + 1.0f/c44_2 + 1.0f/c44_3 + 1.0f/c44_4),-1.0f);

            Txz[index] += dt*Mxz*(dVx_dz + dVz_dx);
        }

        if ((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4))
            P[index] = 0.5f*(Txx[index] + Tzz[index]);
    }
}