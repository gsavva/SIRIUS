#ifdef __GPU
extern "C" void compute_chebyshev_polynomial_gpu(int num_gkvec,
                                                 int n,
                                                 double c,
                                                 double r,
                                                 cuDoubleComplex* phi0,
                                                 cuDoubleComplex* phi1,
                                                 cuDoubleComplex* phi2);
#endif

inline void Band::diag_fv_full_potential_exact(K_point* kp, Periodic_function<double>* effective_potential) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_fv_full_potential_exact");

    if (kp->num_ranks() > 1 && !gen_evp_solver().parallel()) {
        TERMINATE("eigen-value solver is not parallel");
    }

    int ngklo = kp->gklo_basis_size();
    int bs = ctx_.cyclic_block_size();
    dmatrix<double_complex> h(ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> o(ngklo, ngklo, ctx_.blacs_grid(), bs, bs);
    
    /* setup Hamiltonian and overlap */
    switch (ctx_.processing_unit()) {
        case CPU: {
            set_fv_h_o<CPU, electronic_structure_method_t::full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
        #ifdef __GPU
        case GPU: {
            set_fv_h_o<GPU, electronic_structure_method_t::full_potential_lapwlo>(kp, effective_potential, h, o);
            break;
        }
        #endif
        default: {
            TERMINATE("wrong processing unit");
        }
    }

    // TODO: move debug code to a separate function
    #if (__VERIFICATION > 0)
    if (!gen_evp_solver()->parallel())
    {
        Utils::check_hermitian("h", h.panel());
        Utils::check_hermitian("o", o.panel());
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    auto z1 = h.checksum();
    auto z2 = o.checksum();
    DUMP("checksum(h): %18.10f %18.10f", std::real(z1), std::imag(z1));
    DUMP("checksum(o): %18.10f %18.10f", std::real(z2), std::imag(z2));
    #endif

    #ifdef __PRINT_OBJECT_HASH
    DUMP("hash(h): %16llX", h.panel().hash());
    DUMP("hash(o): %16llX", o.panel().hash());
    #endif

    assert(kp->gklo_basis_size() > ctx_.num_fv_states());
    
    std::vector<double> eval(ctx_.num_fv_states());
    
    runtime::Timer t("sirius::Band::diag_fv_full_potential|genevp");
    
    if (gen_evp_solver().solve(kp->gklo_basis_size(), ctx_.num_fv_states(), h.at<CPU>(), h.ld(), o.at<CPU>(), o.ld(), 
                               &eval[0], kp->fv_eigen_vectors().prime().at<CPU>(), kp->fv_eigen_vectors().prime().ld(),
                               kp->gklo_basis_size_row(), kp->gklo_basis_size_col()))

    {
        TERMINATE("error in generalized eigen-value problem");
    }
    kp->set_fv_eigen_values(&eval[0]);

    //== wave_functions phi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
    //==                    [this](int ia) {return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states());
    //== wave_functions hphi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
    //==                    [this](int ia) {return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states());
    //== wave_functions ophi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
    //==                    [this](int ia) {return unit_cell_.atom(ia).mt_lo_basis_size(); }, ctx_.num_fv_states());
    //== 
    //== for (int i = 0; i < ctx_.num_fv_states(); i++) {
    //==     std::memcpy(phi.pw_coeffs().prime().at<CPU>(0, i),
    //==                 kp->fv_eigen_vectors().prime().at<CPU>(0, i),
    //==                 kp->num_gkvec() * sizeof(double_complex));

    //==     std::memcpy(phi.mt_coeffs().prime().at<CPU>(0, i),
    //==                 kp->fv_eigen_vectors().prime().at<CPU>(kp->num_gkvec(), i),
    //==                 unit_cell_.mt_lo_basis_size() * sizeof(double_complex));

    //== }
    //== apply_fv_h_o(kp, effective_potential, 0, ctx_.num_fv_states(), phi, hphi, ophi);

    //== matrix<double_complex> ovlp(ctx_.num_fv_states(), ctx_.num_fv_states());
    //== matrix<double_complex> hmlt(ctx_.num_fv_states(), ctx_.num_fv_states());

    //== inner(phi, 0, ctx_.num_fv_states(), hphi, 0, ctx_.num_fv_states(), hmlt, 0, 0, kp->comm(), ctx_.processing_unit());
    //== inner(phi, 0, ctx_.num_fv_states(), ophi, 0, ctx_.num_fv_states(), ovlp, 0, 0, kp->comm(), ctx_.processing_unit());

    //== for (int i = 0; i < ctx_.num_fv_states(); i++) {
    //==     for (int j = 0; j < ctx_.num_fv_states(); j++) {
    //==         double_complex z = (i == j) ? ovlp(i, j) - 1.0 : ovlp(i, j);
    //==         double_complex z1 = (i == j) ? hmlt(i, j) - eval[i] : hmlt(i, j);
    //==         if (std::abs(z) > 1e-10) {
    //==             printf("ovlp(%i, %i) = %f %f\n", i, j, z.real(), z.imag());
    //==         }
    //==         if (std::abs(z1) > 1e-10) {
    //==             printf("hmlt(%i, %i) = %f %f\n", i, j, z1.real(), z1.imag());
    //==         }
    //==     }
    //== }
    //== STOP();
}

template <typename T>
inline void Band::diag_pseudo_potential_exact(K_point* kp__,
                                              int ispn__,
                                              Hloc_operator& h_op__,
                                              D_operator<T>& d_op__,
                                              Q_operator<T>& q_op__) const
{
    PROFILE();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions(ispn__);

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();     

    int ngk = kp__->num_gkvec();

    wave_functions  phi(ctx_, kp__->comm(), kp__->gkvec(), ngk);
    wave_functions hphi(ctx_, kp__->comm(), kp__->gkvec(), ngk);
    wave_functions ophi(ctx_, kp__->comm(), kp__->gkvec(), ngk);
    
    std::vector<double> eval(ngk);

    phi.pw_coeffs().prime().zero();
    for (int i = 0; i < ngk; i++) phi.pw_coeffs().prime(i, i) = complex_one;

    apply_h_o(kp__, ispn__, 0, ngk, phi, hphi, ophi, h_op__, d_op__, q_op__);
        
    //Utils::check_hermitian("h", hphi.coeffs(), ngk);
    //Utils::check_hermitian("o", ophi.coeffs(), ngk);

    #ifdef __PRINT_OBJECT_CHECKSUM
    auto z1 = hphi.pw_coeffs().prime().checksum();
    auto z2 = ophi.pw_coeffs().prime().checksum();
    printf("checksum(h): %18.10f %18.10f\n", z1.real(), z1.imag());
    printf("checksum(o): %18.10f %18.10f\n", z2.real(), z2.imag());
    #endif

    if (gen_evp_solver().solve(ngk, num_bands,
                               hphi.pw_coeffs().prime().at<CPU>(),
                               hphi.pw_coeffs().prime().ld(),
                               ophi.pw_coeffs().prime().at<CPU>(),
                               ophi.pw_coeffs().prime().ld(), 
                               &eval[0],
                               psi.pw_coeffs().prime().at<CPU>(),
                               psi.pw_coeffs().prime().ld())) {
        TERMINATE("error in evp solve");
    }

    for (int j = 0; j < ctx_.num_fv_states(); j++) {
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
    }
}

inline void Band::diag_fv_full_potential_davidson(K_point* kp, Periodic_function<double>* effective_potential) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_fv_full_potential_davidson");

    auto h_diag = get_h_diag(kp, effective_potential->f_pw(0).real(), ctx_.step_function().theta_pw(0).real());
    auto o_diag = get_o_diag(kp, ctx_.step_function().theta_pw(0).real());

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto& itso = ctx_.iterative_solver_input_section();

    /* short notation for target wave-functions */
    auto& psi = kp->fv_eigen_vectors_slab();

    bool converge_by_energy = (itso.converge_by_energy_ == 1);
    
    assert(num_bands * 2 < kp->num_gkvec()); // iterative subspace size can't be smaller than this

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp->num_gkvec());

    /* allocate wave-functions */
    wave_functions  phi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_phi);
    wave_functions hphi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_phi);
    wave_functions ophi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_phi);
    wave_functions hpsi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_bands);
    wave_functions opsi(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
                        [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_bands);

    /* residuals */
    wave_functions res(ctx_, kp->comm(), kp->gkvec(), unit_cell_.num_atoms(),
                       [this](int ia){return unit_cell_.atom(ia).mt_lo_basis_size();}, num_bands);

    //auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    int bs = ctx_.cyclic_block_size();

    dmatrix<double_complex> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> hmlt_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<double_complex> ovlp_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) {
        eval[i] = kp->band_energy(i);
    }
    std::vector<double> eval_old(num_bands);
    
    /* trial basis functions */
    phi.copy_from(psi, 0, num_bands);
    
    /* current subspace size */
    int N = 0;

    /* number of newly added basis functions */
    int n = num_bands;

    #if (__VERBOSITY > 2)
    if (kp->comm().rank() == 0) {
        DUMP("iterative solver tolerance: %18.12f", ctx_.iterative_solver_tolerance());
    }
    #endif

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #ifdef __GPU
    gpu_mem = cuda_get_free_mem() >> 20;
    printf("[rank%04i at line %i of file %s] CUDA free memory: %i Mb\n", mpi_comm_world().rank(), __LINE__, __FILE__, gpu_mem);
    #endif
    #endif
    
    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_fv_h_o(kp, effective_potential, N, n, phi, hphi, ophi);
        
        orthogonalize(N, n, phi, hphi, ophi, ovlp, res);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx(N, n, phi, hphi, hmlt, hmlt_old);

        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve standard eigen-value problem with the size N */
        diag_subspace_mtrx(N, num_bands, hmlt, evec, eval);

        #if (__VERBOSITY > 2)
        if (kp->comm().rank() == 0) {
            DUMP("step: %i, current subspace size: %i, maximum subspace size: %i", k, N, num_phi);
            for (int i = 0; i < num_bands; i++) DUMP("eval[%i]=%20.16f, diff=%20.16f", i, eval[i], std::abs(eval[i] - eval_old[i]));
        }
        #endif

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1) {
            /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
            n = residuals(kp, 0, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag);
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {   
            runtime::Timer t1("sirius::Band::diag_pseudo_potential_davidson|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            transform<double_complex>(phi, 0, N, evec, 0, 0, psi, 0, num_bands);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1)) {
                break;
            }
            else { /* otherwise, set Psi as a new trial basis */
                #if (__VERBOSITY > 2)
                if (kp->comm().rank() == 0) {
                    DUMP("subspace size limit reached");
                }
                #endif
                hmlt_old.zero();
                for (int i = 0; i < num_bands; i++) {
                    hmlt_old.set(i, i, eval[i]);
                }

                /* need to compute all hpsi and opsi states (not only unconverged) */
                if (converge_by_energy) {
                    transform<double_complex>({&hphi, &ophi}, 0, N, evec, 0, 0, {&hpsi, &opsi}, 0, num_bands);
                }
 
                /* update basis functions */
                phi.copy_from(psi, 0, num_bands);
                /* update hphi and ophi */
                hphi.copy_from(hpsi, 0, num_bands);
                ophi.copy_from(opsi, 0, num_bands);
                /* number of basis functions that we already have */
                N = num_bands;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        phi.copy_from(res, 0, n, N);
    }

    kp->set_fv_eigen_values(&eval[0]);
    kp->comm().barrier();
}

template <typename T>
inline void Band::diag_pseudo_potential_davidson(K_point* kp__,
                                                 int ispn__,
                                                 Hloc_operator& h_op__,
                                                 D_operator<T>& d_op__,
                                                 Q_operator<T>& q_op__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_davidson");

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #ifdef __GPU
    size_t gpu_mem = cuda_get_free_mem() >> 20;
    printf("[rank%04i at line %i of file %s] CUDA free memory: %i Mb\n", mpi_comm_world().rank(), __LINE__, __FILE__, gpu_mem);
    #endif
    #endif

    /* get diagonal elements for preconditioning */
    auto h_diag = get_h_diag(kp__, ispn__, h_op__.v0(ispn__), d_op__);
    auto o_diag = get_o_diag(kp__, q_op__);

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    auto& itso = ctx_.iterative_solver_input_section();

    /* short notation for target wave-functions */
    auto& psi = kp__->spinor_wave_functions(ispn__);

    bool converge_by_energy = (itso.converge_by_energy_ == 1);
    
    assert(num_bands * 2 < kp__->num_gkvec()); // iterative subspace size can't be smaller than this

    /* number of auxiliary basis functions */
    int num_phi = std::min(itso.subspace_size_ * num_bands, kp__->num_gkvec());

    /* allocate wave-functions */
    wave_functions  phi(ctx_, kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions hphi(ctx_, kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions ophi(ctx_, kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions hpsi(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    wave_functions opsi(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    /* residuals */
    wave_functions res(ctx_, kp__->comm(), kp__->gkvec(), num_bands);

    //auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    #ifdef __GPU
    if (gen_evp_solver_->type() == ev_magma) {
        TERMINATE("remember to pin memory in dmatrix");
    }
    #endif

    int bs = ctx_.cyclic_block_size();

    dmatrix<T> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> hmlt_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);
    dmatrix<T> ovlp_old(num_phi, num_phi, ctx_.blacs_grid(), bs, bs);

    std::vector<double> eval(num_bands);
    for (int i = 0; i < num_bands; i++) {
        eval[i] = kp__->band_energy(i);
    }
    std::vector<double> eval_old(num_bands);
    
    kp__->beta_projectors().prepare();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        psi.pw_coeffs().allocate_on_device();
        psi.pw_coeffs().copy_to_device(0, num_bands);

        phi.pw_coeffs().allocate_on_device();
        res.pw_coeffs().allocate_on_device();

        hphi.pw_coeffs().allocate_on_device();
        ophi.pw_coeffs().allocate_on_device();

        hpsi.pw_coeffs().allocate_on_device();
        opsi.pw_coeffs().allocate_on_device();

        evec.allocate(memory_t::device);
        ovlp.allocate(memory_t::device);
        hmlt.allocate(memory_t::device);
    }
    #endif

    /* trial basis functions */
    phi.copy_from(psi, 0, num_bands);

    /* current subspace size */
    int N{0};

    /* number of newly added basis functions */
    int n = num_bands;

    #if (__VERBOSITY > 2)
    if (kp__->comm().rank() == 0) {
        DUMP("iterative solver tolerance: %18.12f", ctx_.iterative_solver_tolerance());
    }
    #endif

    #ifdef __PRINT_MEMORY_USAGE
    MEMORY_USAGE_INFO();
    #ifdef __GPU
    gpu_mem = cuda_get_free_mem() >> 20;
    printf("[rank%04i at line %i of file %s] CUDA free memory: %i Mb\n", mpi_comm_world().rank(), __LINE__, __FILE__, gpu_mem);
    #endif
    #endif
    
    /* start iterative diagonalization */
    for (int k = 0; k < itso.num_steps_; k++) {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o<T>(kp__, ispn__, N, n, phi, hphi, ophi, h_op__, d_op__, q_op__);
        
        orthogonalize<T>(N, n, phi, hphi, ophi, ovlp, res);

        /* setup eigen-value problem
         * N is the number of previous basis functions
         * n is the number of new basis functions */
        set_subspace_mtrx(N, n, phi, hphi, hmlt, hmlt_old);

        /* increase size of the variation space */
        N += n;

        eval_old = eval;

        /* solve standard eigen-value problem with the size N */
        diag_subspace_mtrx(N, num_bands, hmlt, evec, eval);
        
        #if (__VERBOSITY > 2)
        if (kp__->comm().rank() == 0) {
            DUMP("step: %i, current subspace size: %i, maximum subspace size: %i", k, N, num_phi);
            for (int i = 0; i < num_bands; i++) {
                DUMP("eval[%i]=%20.16f, diff=%20.16f", i, eval[i], std::abs(eval[i] - eval_old[i]));
            }
        }
        #endif

        /* check if occupied bands have converged */
        bool occ_band_converged = true;
        for (int i = 0; i < num_bands; i++) {
            if (kp__->band_occupancy(i + ispn__ * ctx_.num_fv_states()) > 1e-2 &&
                std::abs(eval_old[i] - eval[i]) > ctx_.iterative_solver_tolerance()) {
                occ_band_converged = false;
            }
        }

        /* don't compute residuals on last iteration */
        if (k != itso.num_steps_ - 1 && !occ_band_converged) {
            /* get new preconditionined residuals, and also hpsi and opsi as a by-product */
            n = residuals<T>(kp__, ispn__, N, num_bands, eval, eval_old, evec, hphi, ophi, hpsi, opsi, res, h_diag, o_diag);
        }

        /* check if we run out of variational space or eigen-vectors are converged or it's a last iteration */
        if (N + n > num_phi || n <= itso.min_num_res_ || k == (itso.num_steps_ - 1) || occ_band_converged) {   
            runtime::Timer t1("sirius::Band::diag_pseudo_potential_davidson|update_phi");
            /* recompute wave-functions */
            /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
            transform<T>(phi, 0, N, evec, 0, 0, psi, 0, num_bands);

            /* exit the loop if the eigen-vectors are converged or this is a last iteration */
            if (n <= itso.min_num_res_ || k == (itso.num_steps_ - 1) || occ_band_converged) {
                break;
            }
            else { /* otherwise, set Psi as a new trial basis */
                #if (__VERBOSITY > 2)
                if (kp__->comm().rank() == 0) {
                    DUMP("subspace size limit reached");
                }
                #endif
                hmlt_old.zero();
                for (int i = 0; i < num_bands; i++) {
                    hmlt_old.set(i, i, eval[i]);
                }

                /* need to compute all hpsi and opsi states (not only unconverged) */
                if (converge_by_energy) {
                    transform<T>({&hphi, &ophi}, 0, N, evec, 0, 0, {&hpsi, &opsi}, 0, num_bands);
                }
 
                /* update basis functions */
                phi.copy_from(psi, 0, num_bands);
                /* update hphi and ophi */
                hphi.copy_from(hpsi, 0, num_bands);
                ophi.copy_from(opsi, 0, num_bands);
                /* number of basis functions that we already have */
                N = num_bands;
            }
        }
        /* expand variational subspace with new basis vectors obtatined from residuals */
        phi.copy_from(res, 0, n, N);
    }

    kp__->beta_projectors().dismiss();

    for (int j = 0; j < ctx_.num_fv_states(); j++) {
        kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
    }

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        psi.pw_coeffs().copy_to_host(0, num_bands);
        psi.pw_coeffs().deallocate_on_device();
    }
    #endif
    kp__->comm().barrier();
}

template <typename T>
inline void Band::diag_pseudo_potential_chebyshev(K_point* kp__,
                                                  int ispn__,
                                                  Hloc_operator& h_op__,
                                                  D_operator<T>& d_op__,
                                                  Q_operator<T>& q_op__,
                                                  P_operator<T>& p_op__) const
{
    PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_chebyshev");

//==     auto pu = ctx_.processing_unit();
//== 
//==     /* short notation for number of target wave-functions */
//==     int num_bands = ctx_.num_fv_states();
//== 
//==     auto& itso = ctx_.iterative_solver_input_section();
//== 
//==     /* short notation for target wave-functions */
//==     auto& psi = kp__->spinor_wave_functions<false>(ispn__);
//== 
//== //== 
//== //==     //auto& beta_pw_panel = kp__->beta_pw_panel();
//== //==     //dmatrix<double_complex> S(unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->blacs_grid());
//== //==     //linalg<CPU>::gemm(2, 0, unit_cell_.mt_basis_size(), unit_cell_.mt_basis_size(), kp__->num_gkvec(), complex_one,
//== //==     //                  beta_pw_panel, beta_pw_panel, complex_zero, S);
//== //==     //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//== //==     //{
//== //==     //    auto type = unit_cell_.atom(ia)->type();
//== //==     //    int nbf = type->mt_basis_size();
//== //==     //    int ofs = unit_cell_.atom(ia)->offset_lo();
//== //==     //    matrix<double_complex> qinv(nbf, nbf);
//== //==     //    type->uspp().q_mtrx >> qinv;
//== //==     //    linalg<CPU>::geinv(nbf, qinv);
//== //==     //    for (int i = 0; i < nbf; i++)
//== //==     //    {
//== //==     //        for (int j = 0; j < nbf; j++) S.add(ofs + j, ofs + i, qinv(j, i));
//== //==     //    }
//== //==     //}
//== //==     //linalg<CPU>::geinv(unit_cell_.mt_basis_size(), S);
//== //== 
//== //== 
//==     /* maximum order of Chebyshev polynomial*/
//==     int order = itso.num_steps_ + 2;
//== 
//==     std::vector< Wave_functions<false>* > phi(order);
//==     for (int i = 0; i < order; i++) {
//==         phi[i] = new Wave_functions<false>(kp__->num_gkvec_loc(), num_bands, pu);
//==     }
//== 
//==     Wave_functions<false> hphi(kp__->num_gkvec_loc(), num_bands, pu);
//== 
//==     /* trial basis functions */
//==     phi[0]->copy_from(psi, 0, num_bands);
//== 
//==     /* apply Hamiltonian to the basis functions */
//==     apply_h<T>(kp__, ispn__, 0, num_bands, *phi[0], hphi, h_op__, d_op__);
//== 
//==     /* compute Rayleight quotients */
//==     std::vector<double> e0(num_bands, 0.0);
//==     if (pu == CPU) {
//==         #pragma omp parallel for schedule(static)
//==         for (int i = 0; i < num_bands; i++) {
//==             for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//==                 e0[i] += std::real(std::conj((*phi[0])(igk, i)) * hphi(igk, i));
//==             }
//==         }
//==     }
//==     kp__->comm().allreduce(e0);
//== 
//==     //== if (parameters_.processing_unit() == GPU)
//==     //== {
//==     //==     #ifdef __GPU
//==     //==     mdarray<double, 1> e0_loc(kp__->spl_fv_states().local_size());
//==     //==     e0_loc.allocate_on_device();
//==     //==     e0_loc.zero_on_device();
//== 
//==     //==     compute_inner_product_gpu(kp__->num_gkvec_row(),
//==     //==                               (int)kp__->spl_fv_states().local_size(),
//==     //==                               phi[0].at<GPU>(),
//==     //==                               hphi.at<GPU>(),
//==     //==                               e0_loc.at<GPU>());
//==     //==     e0_loc.copy_to_host();
//==     //==     for (int iloc = 0; iloc < (int)kp__->spl_fv_states().local_size(); iloc++)
//==     //==     {
//==     //==         int i = kp__->spl_fv_states(iloc);
//==     //==         e0[i] = e0_loc(iloc);
//==     //==     }
//==     //==     #endif
//==     //== }
//==     //== 
//== 
//==     /* estimate low and upper bounds of the Chebyshev filter */
//==     double lambda0 = -1e10;
//==     //double emin = 1e100;
//==     for (int i = 0; i < num_bands; i++)
//==     {
//==         lambda0 = std::max(lambda0, e0[i]);
//==         //emin = std::min(emin, e0[i]);
//==     }
//==     double lambda1 = 0.5 * std::pow(ctx_.gk_cutoff(), 2);
//== 
//==     double r = (lambda1 - lambda0) / 2.0;
//==     double c = (lambda1 + lambda0) / 2.0;
//== 
//==     auto apply_p = [kp__, &p_op__, num_bands](Wave_functions<false>& phi, Wave_functions<false>& op_phi) {
//==         op_phi.copy_from(phi, 0, num_bands);
//==         //for (int i = 0; i < kp__->beta_projectors().num_beta_chunks(); i++) {
//==         //    kp__->beta_projectors().generate(i);
//== 
//==         //    kp__->beta_projectors().inner<T>(i, phi, 0, num_bands);
//== 
//==         //    p_op__.apply(i, 0, op_phi, 0, num_bands);
//==         //}
//==     };
//== 
//==     apply_p(hphi, *phi[1]);
//==     
//==     /* compute \psi_1 = (S^{-1}H\psi_0 - c\psi_0) / r */
//==     if (pu == CPU) {
//==         #pragma omp parallel for schedule(static)
//==         for (int i = 0; i < num_bands; i++) {
//==             for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//==                 (*phi[1])(igk, i) = ((*phi[1])(igk, i) - (*phi[0])(igk, i) * c) / r;
//==             }
//==         }
//==     }
//== //==     //if (parameters_.processing_unit() == GPU)
//== //==     //{
//== //==     //    #ifdef __GPU
//== //==     //    compute_chebyshev_polynomial_gpu(kp__->num_gkvec_row(), (int)kp__->spl_fv_states().local_size(), c, r,
//== //==     //                                     phi[0].at<GPU>(), phi[1].at<GPU>(), NULL);
//== //==     //    phi[1].panel().copy_to_host();
//== //==     //    #endif
//== //==     //}
//== //== 
//== 
//==     /* compute higher polynomial orders */
//==     for (int k = 2; k < order; k++) {
//== 
//==         apply_h<T>(kp__, ispn__, 0, num_bands, *phi[k - 1], hphi, h_op__, d_op__);
//== 
//==         apply_p(hphi, *phi[k]);
//== 
//==         if (pu == CPU) {
//==             #pragma omp parallel for schedule(static)
//==             for (int i = 0; i < num_bands; i++) {
//==                 for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
//==                     (*phi[k])(igk, i) = ((*phi[k])(igk, i) - c * (*phi[k - 1])(igk, i)) * 2.0 / r - (*phi[k - 2])(igk, i);
//==                 }
//==             }
//==         }
//==         //== if (parameters_.processing_unit() == GPU)
//==         //== {
//==         //==     #ifdef __GPU
//==         //==     compute_chebyshev_polynomial_gpu(kp__->num_gkvec(), num_bands, c, r,
//==         //==                                      phi[k - 2].at<GPU>(), phi[k - 1].at<GPU>(), phi[k].at<GPU>());
//==         //==     phi[k].copy_to_host();
//==         //==     #endif
//==         //== }
//==     }
//== 
//==     /* allocate Hamiltonian and overlap */
//==     matrix<T> hmlt(num_bands, num_bands);
//==     matrix<T> ovlp(num_bands, num_bands);
//==     matrix<T> evec(num_bands, num_bands);
//==     matrix<T> hmlt_old;
//==     matrix<T> ovlp_old;
//== 
//==     int bs = ctx_.cyclic_block_size();
//== 
//==     dmatrix<T> hmlt_dist;
//==     dmatrix<T> ovlp_dist;
//==     dmatrix<T> evec_dist;
//==     if (kp__->comm().size() == 1) {
//==         hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         evec_dist = dmatrix<T>(&evec(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==     } else {
//==         hmlt_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         ovlp_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==         evec_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
//==     }
//== 
//==     std::vector<double> eval(num_bands);
//== 
//==     /* apply Hamiltonian and overlap operators to the new basis functions */
//==     apply_h_o<T>(kp__, ispn__, 0, num_bands, *phi[order - 1], hphi, *phi[0], h_op__, d_op__, q_op__);
//==     
//==     //orthogonalize<T>(kp__, N, n, phi, hphi, ophi, ovlp);
//== 
//==     /* setup eigen-value problem */
//==     set_h_o<T>(kp__, 0, num_bands, *phi[order - 1], hphi, *phi[0], hmlt, ovlp, hmlt_old, ovlp_old);
//== 
//==     /* solve generalized eigen-value problem with the size N */
//==     diag_h_o<T>(kp__, num_bands, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
//== 
//==     /* recompute wave-functions */
//==     /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
//==     psi.transform_from<T>(*phi[order - 1], num_bands, evec, num_bands);
//== 
//==     for (int j = 0; j < ctx_.num_fv_states(); j++) {
//==         kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
//==     }
//== 
//==     for (int i = 0; i < order; i++) {
//==         delete phi[i];
//==     }
}

template <typename T>
inline T 
inner_local(K_point* kp__,
            wave_functions& a,
            int ia,
            wave_functions& b,
            int ib);

template<>
inline double 
inner_local<double>(K_point* kp__,
                    wave_functions& a,
                    int ia,
                    wave_functions& b,
                    int ib)
{
    double result{0};
    double* a_tmp = reinterpret_cast<double*>(&a.pw_coeffs().prime(0, ia));
    double* b_tmp = reinterpret_cast<double*>(&b.pw_coeffs().prime(0, ib));
    for (int igk = 0; igk < 2 * kp__->num_gkvec_loc(); igk++) {
        result += a_tmp[igk] * b_tmp[igk];
    }

    if (kp__->comm().rank() == 0) {
        result = 2 * result - a_tmp[0] * b_tmp[0];
    } else {
        result *= 2;
    }

    return result;
}

template<>
inline double_complex 
inner_local<double_complex>(K_point* kp__,
                            wave_functions& a,
                            int ia,
                            wave_functions& b,
                            int ib)
{
    double_complex result{0, 0};
    for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
        result += std::conj(a.pw_coeffs().prime(igk, ia)) * b.pw_coeffs().prime(igk, ib);
    }
    return result;
}

template <typename T>
inline void Band::diag_pseudo_potential_rmm_diis(K_point* kp__,
                                                 int ispn__,
                                                 Hloc_operator& h_op__,
                                                 D_operator<T>& d_op__,
                                                 Q_operator<T>& q_op__) const

{
    STOP();
    //== auto& itso = ctx_.iterative_solver_input_section();
    //== double tol = ctx_.iterative_solver_tolerance();

    //== if (tol > 1e-4) {
    //==     diag_pseudo_potential_davidson(kp__, ispn__, h_op__, d_op__, q_op__);
    //==     return;
    //== }

    //== PROFILE_WITH_TIMER("sirius::Band::diag_pseudo_potential_rmm_diis");

    //== /* get diagonal elements for preconditioning */
    //== auto h_diag = get_h_diag(kp__, ispn__, h_op__.v0(ispn__), d_op__);
    //== auto o_diag = get_o_diag(kp__, q_op__);

    //== /* short notation for number of target wave-functions */
    //== int num_bands = ctx_.num_fv_states();

    //== auto pu = ctx_.processing_unit();

    //== /* short notation for target wave-functions */
    //== auto& psi = kp__->spinor_wave_functions(ispn__);

    //== int niter = itso.num_steps_;

    //== Eigenproblem_lapack evp_solver(2 * linalg_base::dlamch('S'));

    //== std::vector< wave_functions* > phi(niter);
    //== std::vector< wave_functions* > res(niter);
    //== std::vector< wave_functions* > ophi(niter);
    //== std::vector< wave_functions* > hphi(niter);

    //== for (int i = 0; i < niter; i++) {
    //==     phi[i]  = new wave_functions(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    //==     res[i]  = new wave_functions(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    //==     hphi[i] = new wave_functions(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    //==     ophi[i] = new wave_functions(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    //== }

    //== wave_functions  phi_tmp(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    //== wave_functions hphi_tmp(ctx_, kp__->comm(), kp__->gkvec(), num_bands);
    //== wave_functions ophi_tmp(ctx_, kp__->comm(), kp__->gkvec(), num_bands);

    //== auto mem_type = (gen_evp_solver_->type() == ev_magma) ? memory_t::host_pinned : memory_t::host;

    //== /* allocate Hamiltonian and overlap */
    //== matrix<T> hmlt(num_bands, num_bands, mem_type);
    //== matrix<T> ovlp(num_bands, num_bands, mem_type);
    //== matrix<T> hmlt_old;
    //== matrix<T> ovlp_old;

    //== //#ifdef __GPU
    //== //if (gen_evp_solver_->type() == ev_magma) {
    //== //    hmlt.pin_memory();
    //== //    ovlp.pin_memory();
    //== //}
    //== //#endif

    //== matrix<T> evec(num_bands, num_bands);

    //== int bs = ctx_.cyclic_block_size();

    //== dmatrix<T> hmlt_dist;
    //== dmatrix<T> ovlp_dist;
    //== dmatrix<T> evec_dist;
    //== if (kp__->comm().size() == 1) {
    //==     hmlt_dist = dmatrix<T>(&hmlt(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    //==     ovlp_dist = dmatrix<T>(&ovlp(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    //==     evec_dist = dmatrix<T>(&evec(0, 0), num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    //== } else {
    //==     hmlt_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    //==     ovlp_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    //==     evec_dist = dmatrix<T>(num_bands, num_bands, ctx_.blacs_grid(), bs, bs);
    //== }

    //== std::vector<double> eval(num_bands);
    //== for (int i = 0; i < num_bands; i++) {
    //==     eval[i] = kp__->band_energy(i);
    //== }
    //== std::vector<double> eval_old(num_bands);

    //== /* trial basis functions */
    //== phi[0]->copy_from(psi, 0, num_bands);

    //== std::vector<int> last(num_bands, 0);
    //== std::vector<bool> conv_band(num_bands, false);
    //== std::vector<double> res_norm(num_bands);
    //== std::vector<double> res_norm_start(num_bands);
    //== std::vector<double> lambda(num_bands, 0);
    //== 
    //== auto update_res = [kp__, num_bands, &phi, &res, &hphi, &ophi, &last, &conv_band]
    //==                   (std::vector<double>& res_norm__, std::vector<double>& eval__) -> void
    //== {
    //==     runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|res");
    //==     std::vector<double> e_tmp(num_bands, 0), d_tmp(num_bands, 0);

    //==     #pragma omp parallel for
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             e_tmp[i] = std::real(inner_local<T>(kp__, *phi[last[i]], i, *hphi[last[i]], i));
    //==             d_tmp[i] = std::real(inner_local<T>(kp__, *phi[last[i]], i, *ophi[last[i]], i));
    //==         }
    //==     }
    //==     kp__->comm().allreduce(e_tmp);
    //==     kp__->comm().allreduce(d_tmp);
    //==     
    //==     res_norm__ = std::vector<double>(num_bands, 0);
    //==     #pragma omp parallel for
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             eval__[i] = e_tmp[i] / d_tmp[i];

    //==             /* compute residual r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} */
    //==             for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
    //==                 (*res[last[i]]).pw_coeffs().prime(igk, i) = (*hphi[last[i]]).pw_coeffs().prime(igk, i) - eval__[i] * (*ophi[last[i]]).pw_coeffs().prime(igk, i);
    //==             }
    //==             res_norm__[i] = std::real(inner_local<T>(kp__, *res[last[i]], i, *res[last[i]], i));
    //==         }
    //==     }
    //==     kp__->comm().allreduce(res_norm__);
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             res_norm__[i] = std::sqrt(res_norm__[i]);
    //==         }
    //==     }
    //== };

    //== auto apply_h_o = [this, kp__, num_bands, &phi, &phi_tmp, &hphi, &hphi_tmp, &ophi, &ophi_tmp, &conv_band, &last,
    //==                   &h_op__, &d_op__, &q_op__, ispn__]() -> int
    //== {
    //==     runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|h_o");
    //==     int n{0};
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             std::memcpy(&phi_tmp(0, n), &(*phi[last[i]])(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
    //==             n++;
    //==         }
    //==     }

    //==     if (n == 0) {
    //==         return 0;
    //==     }
    //==     
    //==     /* apply Hamiltonian and overlap operators to the initial basis functions */
    //==     this->apply_h_o<T>(kp__, ispn__, 0, n, phi_tmp, hphi_tmp, ophi_tmp, h_op__, d_op__, q_op__);

    //==     n = 0;
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             std::memcpy(&(*hphi[last[i]])(0, i), &hphi_tmp(0, n), kp__->num_gkvec_loc() * sizeof(double_complex));
    //==             std::memcpy(&(*ophi[last[i]])(0, i), &ophi_tmp(0, n), kp__->num_gkvec_loc() * sizeof(double_complex));
    //==             n++;
    //==         }
    //==     }
    //==     return n;
    //== };

    //== auto apply_preconditioner = [kp__, num_bands, &h_diag, &o_diag, &eval, &conv_band]
    //==                             (std::vector<double> lambda,
    //==                              Wave_functions<false>& res__,
    //==                              double alpha,
    //==                              Wave_functions<false>& kres__) -> void
    //== {
    //==     runtime::Timer t("sirius::Band::diag_pseudo_potential_rmm_diis|pre");
    //==     #pragma omp parallel for
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
    //==                 double p = h_diag[igk] - eval[i] * o_diag[igk];

    //==                 p *= 2; // QE formula is in Ry; here we convert to Ha
    //==                 p = 0.25 * (1 + p + std::sqrt(1 + (p - 1) * (p - 1)));
    //==                 kres__(igk, i) = alpha * kres__(igk, i) + lambda[i] * res__(igk, i) / p;
    //==             }
    //==         }

    //==         //== double Ekin = 0;
    //==         //== double norm = 0;
    //==         //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
    //==         //== {
    //==         //==     Ekin += 0.5 * std::pow(std::abs(res__(igk, i)), 2) * std::pow(kp__->gkvec_cart(igk).length(), 2);
    //==         //==     norm += std::pow(std::abs(res__(igk, i)), 2);
    //==         //== }
    //==         //== Ekin /= norm;
    //==         //== for (int igk = 0; igk < kp__->num_gkvec(); igk++)
    //==         //== {
    //==         //==     double x = std::pow(kp__->gkvec_cart(igk).length(), 2) / 3 / Ekin;
    //==         //==     kres__(igk, i) = alpha * kres__(igk, i) + lambda[i] * res__(igk, i) * 
    //==         //==         (4.0 / 3 / Ekin) * (27 + 18 * x + 12 * x * x + 8 * x * x * x) / (27 + 18 * x + 12 * x * x + 8 * x * x * x + 16 * x * x * x * x);
    //==         //== }
    //==     }
    //== };

    //== /* apply Hamiltonian and overlap operators to the initial basis functions */
    //== this->apply_h_o<T>(kp__, ispn__, 0, num_bands, *phi[0], *hphi[0], *ophi[0], h_op__, d_op__, q_op__);
    //== 
    //== /* compute initial residuals */
    //== update_res(res_norm_start, eval);

    //== bool conv = true;
    //== for (int i = 0; i < num_bands; i++) {
    //==     if (kp__->band_occupancy(i) > 1e-2 && res_norm_start[i] > itso.residual_tolerance_) {
    //==         conv = false;
    //==     }
    //== }
    //== if (conv) {
    //==     DUMP("all bands are converged at stage#0");
    //==     return;
    //== }

    //== last = std::vector<int>(num_bands, 1);
    //== 
    //== /* apply preconditioner to the initial residuals */
    //== apply_preconditioner(std::vector<double>(num_bands, 1), *res[0], 0.0, *phi[1]);
    //== 
    //== /* apply H and O to the preconditioned residuals */
    //== apply_h_o();

    //== /* estimate lambda */
    //== std::vector<double> f1(num_bands, 0);
    //== std::vector<double> f2(num_bands, 0);
    //== std::vector<double> f3(num_bands, 0);
    //== std::vector<double> f4(num_bands, 0);

    //== #pragma omp parallel for
    //== for (int i = 0; i < num_bands; i++) {
    //==     if (!conv_band[i]) {
    //==         f1[i] = std::real(inner_local<T>(kp__, *phi[1], i, *ophi[1], i));     //  <KR_i | OKR_i>
    //==         f2[i] = std::real(inner_local<T>(kp__, *phi[0], i, *ophi[1], i)) * 2; // <phi_i | OKR_i>
    //==         f3[i] = std::real(inner_local<T>(kp__, *phi[1], i, *hphi[1], i));     //  <KR_i | HKR_i>
    //==         f4[i] = std::real(inner_local<T>(kp__, *phi[0], i, *hphi[1], i)) * 2; // <phi_i | HKR_i>
    //==     }
    //== }
    //== kp__->comm().allreduce(f1);
    //== kp__->comm().allreduce(f2);
    //== kp__->comm().allreduce(f3);
    //== kp__->comm().allreduce(f4);

    //== #pragma omp parallel for
    //== for (int i = 0; i < num_bands; i++) {
    //==     if (!conv_band[i]) {
    //==         double a = f1[i] * f4[i] - f2[i] * f3[i];
    //==         double b = f3[i] - eval[i] * f1[i];
    //==         double c = eval[i] * f2[i] - f4[i];

    //==         lambda[i] = (b - std::sqrt(b * b - 4.0 * a * c)) / 2.0 / a;
    //==         if (std::abs(lambda[i]) > 2.0) {
    //==             lambda[i] = 2.0 * Utils::sign(lambda[i]);
    //==         }
    //==         if (std::abs(lambda[i]) < 0.5) {
    //==             lambda[i] = 0.5 * Utils::sign(lambda[i]);
    //==         }
    //==         
    //==         /* construct new basis functions */
    //==         for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
    //==              (*phi[1])(igk, i) =  (*phi[0])(igk, i) + lambda[i] *  (*phi[1])(igk, i);
    //==             (*hphi[1])(igk, i) = (*hphi[0])(igk, i) + lambda[i] * (*hphi[1])(igk, i);
    //==             (*ophi[1])(igk, i) = (*ophi[0])(igk, i) + lambda[i] * (*ophi[1])(igk, i);
    //==         }
    //==     }
    //== }
    //== /* compute new residuals */
    //== update_res(res_norm, eval);
    //== /* check which bands have converged */
    //== for (int i = 0; i < num_bands; i++) {
    //==     if (kp__->band_occupancy(i) <= 1e-2 || res_norm[i] < itso.residual_tolerance_) {
    //==         conv_band[i] = true;
    //==     }
    //== }

    //== mdarray<T, 3> A(niter, niter, num_bands);
    //== mdarray<T, 3> B(niter, niter, num_bands);
    //== mdarray<T, 2> V(niter, num_bands);
    //== std::vector<double> ev(niter);

    //== for (int iter = 2; iter < niter; iter++) {
    //==     runtime::Timer t1("sirius::Band::diag_pseudo_potential_rmm_diis|AB");
    //==     A.zero();
    //==     B.zero();
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             for (int i1 = 0; i1 < iter; i1++) {
    //==                 for (int i2 = 0; i2 < iter; i2++) {
    //==                     A(i1, i2, i) = inner_local<T>(kp__, *res[i1], i, *res[i2], i);
    //==                     B(i1, i2, i) = inner_local<T>(kp__, *phi[i1], i, *ophi[i2], i);
    //==                 }
    //==             }
    //==         }
    //==     }
    //==     kp__->comm().allreduce(A.template at<CPU>(), (int)A.size());
    //==     kp__->comm().allreduce(B.template at<CPU>(), (int)B.size());
    //==     t1.stop();

    //==     runtime::Timer t2("sirius::Band::diag_pseudo_potential_rmm_diis|phi");
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             if (evp_solver.solve(iter, 1, &A(0, 0, i), A.ld(), &B(0, 0, i), B.ld(), &ev[0], &V(0, i), V.ld()) == 0) {
    //==                 std::memset(&(*phi[iter])(0, i), 0, kp__->num_gkvec_loc() * sizeof(double_complex));
    //==                 std::memset(&(*res[iter])(0, i), 0, kp__->num_gkvec_loc() * sizeof(double_complex));
    //==                 for (int i1 = 0; i1 < iter; i1++) {
    //==                     for (int igk = 0; igk < kp__->num_gkvec_loc(); igk++) {
    //==                         (*phi[iter])(igk, i) += (*phi[i1])(igk, i) * V(i1, i);
    //==                         (*res[iter])(igk, i) += (*res[i1])(igk, i) * V(i1, i);
    //==                     }
    //==                 }
    //==                 last[i] = iter;
    //==             } else {
    //==                 conv_band[i] = true;
    //==             }
    //==         }
    //==     }
    //==     t2.stop();
    //==     
    //==     apply_preconditioner(lambda, *res[iter], 1.0, *phi[iter]);

    //==     apply_h_o();

    //==     eval_old = eval;

    //==     update_res(res_norm, eval);
    //==     
    //==     for (int i = 0; i < num_bands; i++) {
    //==         if (!conv_band[i]) {
    //==             if (kp__->band_occupancy(i) <= 1e-2) {
    //==                 conv_band[i] = true;
    //==             }
    //==             if (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol) {
    //==                 conv_band[i] = true;
    //==             }
    //==             if (kp__->band_occupancy(i) > 1e-2 && res_norm[i] < itso.residual_tolerance_) {
    //==                 conv_band[i] = true;
    //==             }
    //==             //if (kp__->band_occupancy(i) <= 1e-2 ||
    //==             //    res_norm[i] / res_norm_start[i] < 0.7 ||
    //==             //    (kp__->band_occupancy(i) > 1e-2 && std::abs(eval[i] - eval_old[i]) < tol)) {
    //==             //    conv_band[i] = true;
    //==             //}
    //==         }
    //==     }
    //==     if (std::all_of(conv_band.begin(), conv_band.end(), [](bool e){return e;})) {
    //==         std::cout << "early exit from the diis loop" << std::endl;
    //==         break;
    //==     }
    //== }

    //== #pragma omp parallel for
    //== for (int i = 0; i < num_bands; i++) {
    //==     std::memcpy(&phi_tmp(0, i),  &(*phi[last[i]])(0, i),  kp__->num_gkvec_loc() * sizeof(double_complex));
    //==     std::memcpy(&hphi_tmp(0, i), &(*hphi[last[i]])(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
    //==     std::memcpy(&ophi_tmp(0, i), &(*ophi[last[i]])(0, i), kp__->num_gkvec_loc() * sizeof(double_complex));
    //== }

    //== if (typeid(T) == typeid(double)) {
    //==     orthogonalize<T>(kp__, 0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, ovlp);
    //== }

    //== set_h_o<T>(kp__, 0, num_bands, phi_tmp, hphi_tmp, ophi_tmp, hmlt, ovlp, hmlt_old, ovlp_old);
    //== 
    //== /* solve generalized eigen-value problem with the size N */
    //== diag_h_o<T>(kp__, num_bands, num_bands, hmlt, ovlp, evec, hmlt_dist, ovlp_dist, evec_dist, eval);
    //== 
    //== /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
    //== psi.transform_from<T>(phi_tmp, num_bands, evec, num_bands);
    //== 
    //== for (int j = 0; j < ctx_.num_fv_states(); j++) {
    //==     kp__->band_energy(j + ispn__ * ctx_.num_fv_states()) = eval[j];
    //== }

    //== for (int i = 0; i < niter; i++) {
    //==     delete phi[i];
    //==     delete res[i];
    //==     delete hphi[i];
    //==     delete ophi[i];
    //== }
}

