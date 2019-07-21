#include <sirius.h>

using namespace sirius;

void init_wf(K_point* kp__, Wave_functions& phi__, int num_bands__, int num_mag_dims__)
{
    std::vector<double> tmp(0xFFFF);
    for (int i = 0; i < 0xFFFF; i++) {
        tmp[i] = utils::random<double>();
    }

    phi__.pw_coeffs(0).prime().zero();

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bands__; i++) {
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->idxgk(igk_loc);
            if (igk == 0) {
                phi__.pw_coeffs(0).prime(igk_loc, i) = 1.0;
            }
            if (igk == i + 1) {
                phi__.pw_coeffs(0).prime(igk_loc, i) = 0.5;
            }
            if (igk == i + 2) {
                phi__.pw_coeffs(0).prime(igk_loc, i) = 0.25;
            }
            if (igk == i + 3) {
                phi__.pw_coeffs(0).prime(igk_loc, i) = 0.125;
            }
            phi__.pw_coeffs(0).prime(igk_loc, i) += tmp[(igk + i) & 0xFFFF] * 1e-5;
        }
    }

    if (num_mag_dims__ == 3) {
        /* make pure spinor up- and dn- wave functions */
        phi__.copy_from(device_t::CPU, num_bands__, phi__, 0, 0, 1, num_bands__);
    }
}

void test_davidson(device_t pu__, double pw_cutoff__, double gk_cutoff__, int N__)
{
    utils::timer t1("test_davidson|setup");

    /* create simulation context */
    Simulation_context ctx(
        "{"
        "   \"parameters\" : {"
        "        \"electronic_structure_method\" : \"pseudopotential\""
        "    },"
        "   \"control\" : {"
        "       \"verification\" : 0"
        "    }"
        "}");

    /* add a new atom type to the unit cell */
    auto& atype = ctx.unit_cell().add_atom_type("Cu");
    /* set pseudo charge */
    atype.zn(11);
    /* set radial grid */
    atype.set_radial_grid(radial_grid_t::lin_exp, 1000, 0.0, 100.0, 6);
    /* cutoff at ~1 a.u. */
    int icut = atype.radial_grid().index_of(1.0);
    double rcut = atype.radial_grid(icut);
    /* create beta radial function */
    std::vector<double> beta(icut + 1);
    std::vector<double> beta1(icut + 1);
    for (int l = 0; l <= 2; l++) {
        for (int i = 0; i <= icut; i++) {
            double x = atype.radial_grid(i);
            beta[i] = utils::confined_polynomial(x, rcut, l, l + 1, 0);
            beta1[i] = utils::confined_polynomial(x, rcut, l, l + 2, 0);
        }
        /* add radial function for l */
        atype.add_beta_radial_function(l, beta);
        atype.add_beta_radial_function(l, beta1);
    }
    /* set local part of potential */
    std::vector<double> vloc(atype.radial_grid().num_points());
    for (int i = 0; i < atype.radial_grid().num_points(); i++) {
        //double x =  atype.radial_grid(i);
        vloc[i] = 0.0; //-atype.zn() / (std::exp(-x * (x + 1)) + x);
    }
    atype.local_potential(vloc);
    /* set Dion matrix */
    int nbf = atype.num_beta_radial_functions();
    matrix<double> dion(nbf, nbf);
    dion.zero();
    //for (int i = 0; i < nbf; i++) {
    //    dion(i, i) = -10.0;
    //}
    atype.d_mtrx_ion(dion);
    /* set atomic density */
    std::vector<double> arho(atype.radial_grid().num_points());
    for (int i = 0; i < atype.radial_grid().num_points(); i++) {
        double x = atype.radial_grid(i);
        arho[i] = 2 * atype.zn() * std::exp(-x * x) * x;
    }
    atype.ps_total_charge_density(arho);

    /* lattice constant */
    double a{5};
    /* set lattice vectors */
    ctx.unit_cell().set_lattice_vectors({{a * N__, 0, 0}, 
                                         {0, a * N__, 0}, 
                                         {0, 0, a * N__}});
    /* add atoms */
    double p = 1.0 / N__;
    for (int i = 0; i < N__; i++) {
        for (int j = 0; j < N__; j++) {
            for (int k = 0; k < N__; k++) {
                ctx.unit_cell().add_atom("Cu", {i * p, j * p, k * p});
            }
        }
    }

    /* initialize the context */
    ctx.set_verbosity(1);
    ctx.pw_cutoff(pw_cutoff__);
    ctx.gk_cutoff(gk_cutoff__);
    ctx.set_processing_unit(pu__);
    t1.stop();

    ctx.set_verbosity(1);
    ctx.iterative_solver_tolerance(1e-12);
    //ctx.set_iterative_solver_type("exact");
    ctx.initialize();

    Density rho(ctx);
    rho.initial_density();
    rho.zero();

    Potential pot(ctx);
    pot.generate(rho);
    pot.zero();


    for (int r = 0; r < 2; r++) {
        double vk[] = {0.1, 0.1, 0.1};
        K_point kp(ctx, vk, 1.0);
        kp.initialize();
        std::cout << "num_gkvec=" << kp.num_gkvec() << "\n";
        for (int i = 0; i < ctx.num_bands(); i++) {
            kp.band_occupancy(i, 0, 2);
        }
        init_wf(&kp, kp.spinor_wave_functions(), ctx.num_bands(), 0);



        Hamiltonian H(ctx, pot);
        H.prepare();
        Band(ctx).solve_pseudo_potential<double_complex>(kp, H);
        //for (int i = 0; i < ctx.num_bands(); i++) {
        //    std::cout << "energy[" << i << "]=" << kp.band_energy(i, 0) << "\n";
        //}
        std::vector<double> ekin(kp.num_gkvec());
        for (int i = 0; i < kp.num_gkvec(); i++) {
            ekin[i] = 0.5 * kp.gkvec().gkvec_cart<index_domain_t::global>(i).length2();
        }
        std::sort(ekin.begin(), ekin.end());

        for (int i = 0; i < ctx.num_bands(); i++) {
            printf("%20.16f %20.16f %20.16e\n", ekin[i], kp.band_energy(i, 0), std::abs(ekin[i] - kp.band_energy(i, 0)));
        }
    }

    //ctx.set_iterative_solver_type("davidson");
    //Band(ctx).solve_pseudo_potential<double_complex>(kp, H);

    //for (int i = 0; i < 1; i++) {

    //    double vk[] = {0.1 * i, 0.1 * i, 0.1 * i};
    //    K_point kp(ctx, vk, 1.0);
    //    kp.initialize();

    //Hamiltonian0 h0(ctx, pot);
    //auto hk = h0(kp);

    ////    init_wf(&kp, kp.spinor_wave_functions(), ctx.num_bands(), 0);

    //auto eval = davidson<double_complex>(hk, kp.spinor_wave_functions(), 0, 4, 40, 1e-12, 0, 1e-7, [](int i, int ispn){return 1.0;}, false);

    //std::vector<double> ekin(kp.num_gkvec());
    //for (int i = 0; i < kp.num_gkvec(); i++) {
    //    ekin[i] = 0.5 * kp.gkvec().gkvec_cart<index_domain_t::global>(i).length2();
    //}
    //std::sort(ekin.begin(), ekin.end());

    //for (int i = 0; i < ctx.num_bands(); i++) {
    //    printf("%20.16f %20.16f %20.16e\n", ekin[i], eval[i], std::abs(ekin[i] - eval[i]));
    //}

    //}

    //for (int i = 0; i < 1; i++) {

    //    double vk[] = {0.1 * i, 0.1 * i, 0.1 * i};
    //    K_point kp(ctx, vk, 1.0);
    //    kp.initialize();
    //    for (int i = 0; i < ctx.num_bands(); i++) {
    //        kp.band_occupancy(i, 0, 2);
    //    }
    //    init_wf(&kp, kp.spinor_wave_functions(), ctx.num_bands(), 0);

    //    Hamiltonian H(ctx, pot);
    //    H.prepare();
    //    Band(ctx).solve_pseudo_potential<double_complex>(kp, H);
    //}

    //for (int i = 0; i < 1; i++) {

    //    double vk[] = {0.1 * i, 0.1 * i, 0.1 * i};
    //    K_point kp(ctx, vk, 1.0);
    //    kp.initialize();
    //    for (int i = 0; i < ctx.num_bands(); i++) {
    //        kp.band_occupancy(i, 0, 2);
    //    }
    //    init_wf(&kp, kp.spinor_wave_functions(), ctx.num_bands(), 0);

    //    Hamiltonian H(ctx, pot);
    //    H.prepare();
    //    Band(ctx).solve_pseudo_potential<double_complex>(kp, H);
    //}



}

int main(int argn, char** argv)
{
    cmd_args args(argn, argv, {{"device=", "(string) CPU or GPU"},
                               {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
                               {"gk_cutoff=", "(double) plane-wave cutoff for wave-functions"},
                               {"N=", "(int) cell multiplicity"},
                               {"mpi_grid=", "(int[2]) dimensions of the MPI grid for band diagonalization"}
                              });

    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    auto pu        = get_device_t(args.value<std::string>("device", "CPU"));
    auto pw_cutoff = args.value<double>("pw_cutoff", 30);
    auto gk_cutoff = args.value<double>("gk_cutoff", 10);
    auto N         = args.value<int>("N", 1);
    auto mpi_grid  = args.value<std::vector<int>>("mpi_grid", {1, 1});

    sirius::initialize(1);
    test_davidson(pu, pw_cutoff, gk_cutoff, N);
    int rank = Communicator::world().rank();
    sirius::finalize();
    if (!rank) {
        utils::timer::print();
        utils::timer::print_tree();
    }
}
