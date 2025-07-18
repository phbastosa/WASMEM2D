# include "elastic_ani.cuh"

void Elastic_ANI::set_specifications()
{
    modeling_type = "elastic_ani";
    modeling_name = "Modeling type: Elastic anisotropic solver";

    int samples = 2*RSGR+1;
    int nKernel = samples*samples;
    float * kernel = new float[nKernel]();

    int index = 0;
    float sum = 0.0f;

    for (int z = -RSGR; z <= RSGR; z++)
    {
        for (int x = -RSGR; x <= RSGR; x++)
        {
            float r = sqrtf(x*x + z*z);

            kernel[index] = 1.0f/sqrtf(2.0f*M_PI)*expf(-0.5f*r*r);

            sum += kernel[index]; 
        
            ++index;
        }
    }

    for (index = 0; index < nKernel; index++) 
        kernel[index] /= sum;

    cudaMalloc((void**)&(dwc), nKernel*sizeof(float));
    cudaMemcpy(dwc, kernel, nKernel*sizeof(float), cudaMemcpyHostToDevice);
    delete[] kernel;

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
    delete[] uB;

    auto * C11 = new float[matsize]();
    auto * uC11 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C11.bin", Cij, nPoints);
    expand_boundary(Cij, C11);
    compression(C11, uC11, matsize, maxC11, minC11);        
    cudaMalloc((void**)&(d_C11), matsize*sizeof(uintc));
    cudaMemcpy(d_C11, uC11, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C11;
    delete[] uC11;

    auto * C13 = new float[matsize]();
    auto * uC13 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C13.bin", Cij, nPoints);
    expand_boundary(Cij, C13);
    compression(C13, uC13, matsize, maxC13, minC13);    
    cudaMalloc((void**)&(d_C13), matsize*sizeof(uintc));
    cudaMemcpy(d_C13, uC13, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C13;
    delete[] uC13;

    auto * C15 = new float[matsize]();
    auto * uC15 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C15.bin", Cij, nPoints);
    expand_boundary(Cij, C15);
    compression(C15, uC15, matsize, maxC15, minC15);    
    cudaMalloc((void**)&(d_C15), matsize*sizeof(uintc));
    cudaMemcpy(d_C15, uC15, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C15;
    delete[] uC15;

    auto * C33 = new float[matsize]();
    auto * uC33 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C33.bin", Cij, nPoints);
    expand_boundary(Cij, C33);
    compression(C33, uC33, matsize, maxC33, minC33);    
    cudaMalloc((void**)&(d_C33), matsize*sizeof(uintc));
    cudaMemcpy(d_C33, uC33, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C33;
    delete[] uC33;
    
    auto * C35 = new float[matsize]();
    auto * uC35 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C35.bin", Cij, nPoints);
    expand_boundary(Cij, C35);
    compression(C35, uC35, matsize, maxC35, minC35);    
    cudaMalloc((void**)&(d_C35), matsize*sizeof(uintc));
    cudaMemcpy(d_C35, uC35, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C35;
    delete[] uC35;

    auto * C55 = new float[matsize]();
    auto * uC55 = new uintc[matsize]();
    import_binary_float(Cijkl_folder + "C55.bin", Cij, nPoints);
    expand_boundary(Cij, C55);
    compression(C55, uC55, matsize, maxC55, minC55);    
    cudaMalloc((void**)&(d_C55), matsize*sizeof(uintc));
    cudaMemcpy(d_C55, uC55, matsize*sizeof(uintc), cudaMemcpyHostToDevice);
    delete[] C55;
    delete[] uC55;

    delete[] Cij;
}

void Elastic_ANI::compute_eikonal()
{
    dim3 grid(1,1,1);
    dim3 block(MESHDIM+1,MESHDIM+1,1);

    time_set<<<nBlocks,NTHREADS>>>(d_T, matsize);
    time_init<<<grid,block>>>(d_T,d_S,sx,sz,dx,dz,sIdx,sIdz,nzz,nb);
    eikonal_solver();

    get_quasi_slowness<<<nBlocks,NTHREADS>>>(d_T,d_S,dx,dz,sIdx,sIdz,nxx,nzz,nb,d_C11,d_C13,d_C15,d_C33,d_C35,d_C55,minC11,
                                             maxC11,minC13,maxC13,minC15,maxC15,minC33,maxC33,minC35,maxC35,minC55,maxC55);

    time_set<<<nBlocks,NTHREADS>>>(d_T, matsize);
    time_init<<<grid,block>>>(d_T,d_S,sx,sz,dx,dz,sIdx,sIdz,nzz,nb);
    eikonal_solver();

    cudaMemcpy(d_S, S, matsize * sizeof(float), cudaMemcpyHostToDevice);
}

void Elastic_ANI::compute_velocity()
{
    compute_velocity_rsg<<<nBlocks, NTHREADS>>>(d_Vx, d_Vz, d_Txx, d_Tzz, d_Txz, d_T, d_B, minB, maxB, d1D, d2D, 
                                                d_wavelet, dwc, dx, dz, dt, timeId, tlag, sIdx, sIdz, nxx, nzz, nb, nt);
}

void Elastic_ANI::compute_pressure()
{
    compute_pressure_rsg<<<nBlocks, NTHREADS>>>(d_Vx, d_Vz, d_Txx, d_Tzz, d_Txz, d_P, d_T, d_C11, d_C13, d_C15, d_C33, 
                                                d_C35, d_C55, timeId, tlag, dx, dz, dt, nxx, nzz, minC11, maxC11, minC13, 
                                                maxC13, minC15, maxC15, minC33, maxC33, minC35, maxC35, minC55, maxC55);    
}

__global__ void get_quasi_slowness(float * T, float * S, float dx, float dz, int sIdx, int sIdz, int nxx, int nzz, 
                                   int nb, uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, 
                                   float minC11, float maxC11, float minC13, float maxC13, float minC15, float maxC15, 
                                   float minC33, float maxC33, float minC35, float maxC35, float minC55, float maxC55)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    const int n = 2;
    const int v = 3;

    float p[n];
    float C[v*v];
    float Gv[n];

    if ((i >= nb) && (i < nzz-nb) && (j >= nb) && (j < nxx-nb))
    {
        if (!((i == sIdz) && (j == sIdx)))    
        {
            float dTz = 0.5f*(T[(i+1) + j*nzz] - T[(i-1) + j*nzz]) / dz;
            float dTx = 0.5f*(T[i + (j+1)*nzz] - T[i + (j-1)*nzz]) / dx;

            float norm = sqrtf(dTx*dTx + dTz*dTz);

            p[0] = dTx / norm;
            p[1] = dTz / norm;
            
            float c11 = (minC11 + (static_cast<float>(C11[index]) - 1.0f) * (maxC11 - minC11) / (COMPRESS - 1));
            float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
            float c15 = (minC15 + (static_cast<float>(C15[index]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));

            float c33 = (minC33 + (static_cast<float>(C33[index]) - 1.0f) * (maxC33 - minC33) / (COMPRESS - 1));
            float c35 = (minC35 + (static_cast<float>(C35[index]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));

            float c55 = (minC55 + (static_cast<float>(C55[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));

            C[0+0*v] = c11; C[0+1*v] = c13; C[0+2*v] = c15;
            C[1+0*v] = c13; C[1+1*v] = c33; C[1+2*v] = c35;
            C[2+0*v] = c15; C[2+1*v] = c35; C[2+2*v] = c55;

            float Ro = c33*S[index]*S[index];    
            
            for (int indp = 0; indp < v*v; indp++)
                C[indp] = C[indp] / Ro / Ro;

            float Gxx = C[0+0*v]*p[0]*p[0] + C[2+2*v]*p[1]*p[1] + 2.0f*C[0+2*v]*p[0]*p[1];
            float Gzz = C[2+2*v]*p[0]*p[0] + C[1+1*v]*p[1]*p[1] + 2.0f*C[1+2*v]*p[0]*p[1];
            float Gxz = C[0+2*v]*p[0]*p[0] + C[1+2*v]*p[1]*p[1] + (C[0+1*v] + C[2+2*v])*p[0]*p[1]; 
            
            float coeff1 = Gxx + Gzz;
            float coeff2 = Gxx - Gzz;
            
            float det = sqrtf((coeff2 * coeff2) / 4.0f + Gxz * Gxz);

            Gv[0] = coeff1 / 2.0 + det;
            Gv[1] = coeff1 / 2.0 - det;
            
            if (Gv[0] < Gv[1]) {float aux = Gv[0]; Gv[0] = Gv[1]; Gv[1] = aux;} 

            S[index] = 1.0f / sqrtf(Gv[0] * Ro);
        }
    }
}

__global__ void compute_velocity_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * T, uintc * B, float minB, 
                                     float maxB, float * damp1D, float * damp2D, float * wavelet, float * dwc, float dx, float dz, 
                                     float dt, int tId, int tlag, int sIdx, int sIdz, int nxx, int nzz, int nb, int nt)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    if ((index == 0) && (tId < nt))
    {
        int sId = 0;

        for (int si = -RSGR; si <= RSGR; si++)
        {
            for (int sj = -RSGR; sj <= RSGR; sj++)
            {
                Txx[(sIdz+si) + (sIdx+sj)*nzz] += dwc[sId]*wavelet[tId] / (dx*dz);
                Tzz[(sIdz+si) + (sIdx+sj)*nzz] += dwc[sId]*wavelet[tId] / (dx*dz);                            
                    
                ++sId;
            }
        }
    }

    float d1_Txx = 0.0f; float d2_Txx = 0.0f;
    float d1_Tzz = 0.0f; float d2_Tzz = 0.0f;
    float d1_Txz = 0.0f; float d2_Txz = 0.0f;
 
    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};
    
    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {
        if((i >= 3) && (i < nzz-4) && (j >= 3) && (j < nxx-4)) 
        {   
            # pragma unroll 4 
            for (int rsg = 0; rsg < 4; rsg++)
            {
                d1_Txx += FDM[rsg]*(Txx[(i+rsg+1) + (j+rsg+1)*nzz] - Txx[(i-rsg) + (j-rsg)*nzz]);
                d1_Tzz += FDM[rsg]*(Tzz[(i+rsg+1) + (j+rsg+1)*nzz] - Tzz[(i-rsg) + (j-rsg)*nzz]);
                d1_Txz += FDM[rsg]*(Txz[(i+rsg+1) + (j+rsg+1)*nzz] - Txz[(i-rsg) + (j-rsg)*nzz]);

                d2_Txx += FDM[rsg]*(Txx[(i-rsg) + (j+rsg+1)*nzz] - Txx[(i+rsg+1) + (j-rsg)*nzz]);
                d2_Tzz += FDM[rsg]*(Tzz[(i-rsg) + (j+rsg+1)*nzz] - Tzz[(i+rsg+1) + (j-rsg)*nzz]);
                d2_Txz += FDM[rsg]*(Txz[(i-rsg) + (j+rsg+1)*nzz] - Txz[(i+rsg+1) + (j-rsg)*nzz]);
            }
        }
    
        float dTxx_dx = 0.5f*(d1_Txx + d2_Txx) / dx;
        float dTxz_dx = 0.5f*(d1_Txz + d2_Txz) / dx;

        float dTxz_dz = 0.5f*(d1_Txz - d2_Txz) / dz;
        float dTzz_dz = 0.5f*(d1_Tzz - d2_Tzz) / dz;

        float B00 = (minB + (static_cast<float>(B[i + j*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));
        float B10 = (minB + (static_cast<float>(B[i + (j+1)*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));;
        float B01 = (minB + (static_cast<float>(B[(i+1) + j*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));;
        float B11 = (minB + (static_cast<float>(B[(i+1) + (j+1)*nzz]) - 1.0f) * (maxB - minB) / (COMPRESS - 1));;

        float Bxz = 0.25f*(B00 + B10 + B01 + B11);

        Vx[index] += dt*Bxz*(dTxx_dx + dTxz_dz); 
        Vz[index] += dt*Bxz*(dTxz_dx + dTzz_dz);    
        
    	float damper = get_boundary_damper(damp1D, damp2D, i, j, nxx, nzz, nb);

        Vx[index] *= damper;
        Vz[index] *= damper;

        Txx[index] *= damper;
        Tzz[index] *= damper;
        Txz[index] *= damper;
    }
}

__global__ void compute_pressure_rsg(float * Vx, float * Vz, float * Txx, float * Tzz, float * Txz, float * P, float * T, 
                                     uintc * C11, uintc * C13, uintc * C15, uintc * C33, uintc * C35, uintc * C55, int tId, 
                                     int tlag, float dx, float dz, float dt, int nxx, int nzz, float minC11, float maxC11, 
                                     float minC13, float maxC13, float minC15, float maxC15, float minC33, float maxC33, 
                                     float minC35, float maxC35, float minC55, float maxC55)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i = (int)(index % nzz);
    int j = (int)(index / nzz);

    float d1_Vx = 0.0f; float d2_Vx = 0.0f;
    float d1_Vz = 0.0f; float d2_Vz = 0.0f;

    float FDM[] = {FDM4, -FDM3, FDM2, -FDM1};

    if ((T[index] < (float)(tId + tlag)*dt) && (index < nxx*nzz))
    {
        if((i > 3) && (i < nzz-3) && (j > 3) && (j < nxx-3)) 
        {
            # pragma unroll 4
            for (int rsg = 0; rsg < 4; rsg++)
            {       
                d1_Vx += FDM[rsg]*(Vx[(i+rsg) + (j+rsg)*nzz] - Vx[(i-rsg-1) + (j-rsg-1)*nzz]);      
                d1_Vz += FDM[rsg]*(Vz[(i+rsg) + (j+rsg)*nzz] - Vz[(i-rsg-1) + (j-rsg-1)*nzz]);      
    
                d2_Vx += FDM[rsg]*(Vx[(i-rsg-1) + (j+rsg)*nzz] - Vx[(i+rsg) + (j-rsg-1)*nzz]);      
                d2_Vz += FDM[rsg]*(Vz[(i-rsg-1) + (j+rsg)*nzz] - Vz[(i+rsg) + (j-rsg-1)*nzz]);      
            }
        }
    
        float dVx_dx = 0.5f*(d1_Vx + d2_Vx) / dx;
        float dVz_dx = 0.5f*(d1_Vz + d2_Vz) / dx;
        
        float dVx_dz = 0.5f*(d1_Vx - d2_Vx) / dz;
        float dVz_dz = 0.5f*(d1_Vz - d2_Vz) / dz;

        float c11 = (minC11 + (static_cast<float>(C11[index]) - 1.0f) * (maxC11 - minC11) / (COMPRESS - 1));
        float c13 = (minC13 + (static_cast<float>(C13[index]) - 1.0f) * (maxC13 - minC13) / (COMPRESS - 1));
        float c15 = (minC15 + (static_cast<float>(C15[index]) - 1.0f) * (maxC15 - minC15) / (COMPRESS - 1));
        float c33 = (minC33 + (static_cast<float>(C33[index]) - 1.0f) * (maxC33 - minC33) / (COMPRESS - 1));
        float c35 = (minC35 + (static_cast<float>(C35[index]) - 1.0f) * (maxC35 - minC35) / (COMPRESS - 1));    
        float c55 = (minC55 + (static_cast<float>(C55[index]) - 1.0f) * (maxC55 - minC55) / (COMPRESS - 1));
                
        Txx[index] += dt*(c11*dVx_dx + c13*dVz_dz + c15*(dVx_dz + dVz_dx));
        Tzz[index] += dt*(c13*dVx_dx + c33*dVz_dz + c35*(dVx_dz + dVz_dx));
        Txz[index] += dt*(c15*dVx_dx + c35*dVz_dz + c55*(dVx_dz + dVz_dx));
    
        if ((i > 3) && (i < nzz-4) && (j > 3) && (j < nxx-4))
        {
            P[index] = 0.5f*(Txx[index] + Tzz[index]);
        }
    }
}