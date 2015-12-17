#include <sirius.h>
#include <thread>
#include <wave_functions.h>

using namespace sirius;

void test1(vector3d<int> const& dims__, double cutoff__, int num_bands__, std::vector<int> mpi_grid_dims__)
{
    Communicator comm(MPI_COMM_WORLD);
    MPI_grid mpi_grid(mpi_grid_dims__, comm);

    matrix3d<double> M;
    M(0, 0) = M(1, 1) = M(2, 2) = 1.0;
    
    FFT3D fft(dims__, Platform::max_num_threads(), mpi_grid.communicator(1 << 1), CPU);
    MEMORY_USAGE_INFO();

    Gvec gvec(vector3d<double>(0, 0, 0), M, cutoff__, fft.fft_grid(), fft.comm(), mpi_grid.communicator(1 << 0).size(), false, false);
    MEMORY_USAGE_INFO();

    Wave_functions psi_in(num_bands__, gvec, mpi_grid, true);
    Wave_functions psi_out(num_bands__, gvec, mpi_grid, true);
    
    for (int i = 0; i < num_bands__; i++)
    {
        for (int j = 0; j < psi_in.num_gvec_loc(); j++)
        {
            psi_in(j, i) = type_wrapper<double_complex>::random();
        }
    }
    MEMORY_USAGE_INFO();
    if (comm.rank() == 0)
    {
        printf("num_gvec_loc: %i\n", gvec.num_gvec(comm.rank()));
        printf("local size of wf: %f GB\n", sizeof(double_complex) * num_bands__ * gvec.num_gvec(comm.rank()) / double(1 << 30));
    }
    Timer t1("swap", comm);
    psi_in.swap_forward(0, num_bands__);
    MEMORY_USAGE_INFO();
    for (int i = 0; i < psi_in.spl_num_swapped().local_size(); i++)
    {
        std::memcpy(psi_out[i], psi_in[i], gvec.num_gvec_fft() * sizeof(double_complex));
    }
    psi_out.swap_backward(0, num_bands__);
    t1.stop();

    double diff = 0;
    for (int i = 0; i < num_bands__; i++)
    {
        for (int j = 0; j < psi_in.num_gvec_loc(); j++)
        {
            double d = std::abs(psi_in(j, i) - psi_out(j, i));
            diff += d;
        }
    }
    printf("diff: %18.12f\n", diff);

    comm.barrier();
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--dims=", "{vector3d<int>} FFT dimensions");
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--mpi_grid=", "{vector2d<int>} MPI grid");

    args.parse_args(argn, argv);
    if (argn == 1)
    {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        exit(0);
    }

    vector3d<int> dims = args.value< vector3d<int> >("dims");
    double cutoff = args.value<double>("cutoff", 1);
    int num_bands = args.value<int>("num_bands", 50);
    std::vector<int> mpi_grid = args.value< std::vector<int> >("mpi_grid", {1, 1});

    Platform::initialize(1);

    test1(dims, cutoff, num_bands, mpi_grid);
    
    Timer::print();

    Platform::finalize();
}
