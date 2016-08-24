#include "kernels_common.h"

__global__ void mul_veff_with_phase_factors_gpu_kernel(int num_gvec_loc__,
                                                       cuDoubleComplex const* veff__, 
                                                       int const* gvec__, 
                                                       double const* atom_pos__, 
                                                       double* veff_a__)
{
    int igloc = blockDim.x * blockIdx.x + threadIdx.x;
    int ia = blockIdx.y;

    if (igloc < num_gvec_loc__)
    {
        int gvx = gvec__[array2D_offset(0, igloc, 3)];
        int gvy = gvec__[array2D_offset(1, igloc, 3)];
        int gvz = gvec__[array2D_offset(2, igloc, 3)];
        double ax = atom_pos__[array2D_offset(0, ia, 3)];
        double ay = atom_pos__[array2D_offset(1, ia, 3)];
        double az = atom_pos__[array2D_offset(2, ia, 3)];

        double p = twopi * (ax * gvx + ay * gvy + az * gvz);

        cuDoubleComplex z = cuConj(cuCmul(veff__[igloc], make_cuDoubleComplex(cos(p), sin(p))));
        veff_a__[array2D_offset(2 * igloc,     ia, 2 * num_gvec_loc__)] = z.x;
        veff_a__[array2D_offset(2 * igloc + 1, ia, 2 * num_gvec_loc__)] = z.y;
    }
}
 
extern "C" void mul_veff_with_phase_factors_gpu(int num_atoms__,
                                                int num_gvec_loc__, 
                                                cuDoubleComplex const* veff__, 
                                                int const* gvec__, 
                                                double const* atom_pos__,
                                                double* veff_a__,
                                                int stream_id__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_gvec_loc__, grid_t.x), num_atoms__);

    cudaStream_t stream = cuda_stream_by_id(stream_id__);

    mul_veff_with_phase_factors_gpu_kernel <<<grid_b, grid_t, 0, stream>>>
    (
        num_gvec_loc__,
        veff__,
        gvec__,
        atom_pos__,
        veff_a__
    );
}
