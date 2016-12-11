inline void Band::initialize_subspace(K_set& kset__,
                                      Potential& potential__) const
{
    PROFILE("sirius::Band::initialize_subspace");

    int nq = 20;
    int lmax = 1;
    /* this is the regular grid in reciprocal space in the range [0, |G+k|_max ] */
    Radial_grid qgrid(linear_grid, nq, 0, ctx_.gk_cutoff());

    /* interpolate I_{\alpha,n}(q) = <j_{l_n}(q*x) | wf_{n,l_n}(x) > with splines */
    std::vector<std::vector<Spline<double>>> rad_int(unit_cell_.num_atom_types());
    
    /* spherical Bessel functions jl(qx) for atom types */
    mdarray<Spherical_Bessel_functions, 2> jl(nq, unit_cell_.num_atom_types());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        /* create jl(qx) */
        #pragma omp parallel for
        for (int iq = 0; iq < nq; iq++) {
            jl(iq, iat) = Spherical_Bessel_functions(lmax, atom_type.radial_grid(), qgrid[iq]);
        }

        //rad_int[iat].resize(atom_type.pp_desc().atomic_pseudo_wfs_.size());
        rad_int[iat].resize(lmax + 1);
        /* loop over all pseudo wave-functions */
        //for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
        for (int l = 0; l <= lmax; l++) {
            //rad_int[iat][i] = Spline<double>(qgrid);
            rad_int[iat][l] = Spline<double>(qgrid);
            
            /* interpolate atomic_pseudo_wfs(r) */
            Spline<double> wf(atom_type.radial_grid());
            for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                //wf[ir] = atom_type.pp_desc().atomic_pseudo_wfs_[i].second[ir];
                double x = atom_type.radial_grid(ir);
                wf[ir] = std::exp(-atom_type.zn() * x) * std::pow(x, l);
            }
            wf.interpolate();
            double norm = inner(wf, wf, 2);
            
            //int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
            #pragma omp parallel for
            for (int iq = 0; iq < nq; iq++) {
                //rad_int[iat][i][iq] = sirius::inner(jl(iq, iat)[l], wf, 1);
                rad_int[iat][l][iq] = inner(jl(iq, iat)[l], wf, 2) / std::sqrt(norm);
            }

            //rad_int[iat][i].interpolate();
            rad_int[iat][l].interpolate();
        }
    }

    /* get the total number of atomic-centered orbitals */
    int N{0};
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        int n = Utils::lmmax(lmax);
        //int n{0};
        //for (auto& wf: atom_type.pp_desc().atomic_pseudo_wfs_) {
        //    n += (2 * wf.first + 1);
        //}
        N += atom_type.num_atoms() * n;
    }

    if (ctx_.comm().rank() == 0 && ctx_.control().verbosity_ >= 2) {
        printf("number of atomic orbitals: %i\n", N);
    }

    for (int ikloc = 0; ikloc < kset__.spl_num_kpoints().local_size(); ikloc++) {
        int ik = kset__.spl_num_kpoints(ikloc);
        auto kp = kset__[ik];
        
        if (ctx_.gamma_point()) {
            initialize_subspace<double>(kp, potential__.effective_potential(),
                                        potential__.effective_magnetic_field(), N, lmax, rad_int);
        } else {
            initialize_subspace<double_complex>(kp, potential__.effective_potential(),
                                                potential__.effective_magnetic_field(), N, lmax, rad_int);
        }
    }

    /* reset the energies for the iterative solver to do at least two steps */
    for (int ik = 0; ik < kset__.num_kpoints(); ik++) {
        for (int i = 0; i < ctx_.num_bands(); i++) {
            kset__[ik]->band_energy(i) = 0;
            kset__[ik]->band_occupancy(i) = ctx_.max_occupancy();
        }
    }
}

template <typename T>
inline void Band::initialize_subspace(K_point* kp__,
                                      Periodic_function<double>* effective_potential__,
                                      Periodic_function<double>* effective_magnetic_field__[3],
                                      int num_ao__,
                                      int lmax__,
                                      std::vector<std::vector<Spline<double>>> const& rad_int__) const
{
    PROFILE("sirius::Band::initialize_subspace|kp");

    /* number of basis functions */
    int num_phi = std::max(num_ao__, ctx_.num_fv_states());

    wave_functions phi(ctx_.processing_unit(), kp__->comm(), kp__->gkvec(), num_phi);

    #pragma omp parallel
    {
        std::vector<double> gkvec_rlm(Utils::lmmax(lmax__));
        /* fill first N functions with atomic orbitals */
        #pragma omp for
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->gkvec().gvec_offset(kp__->comm().rank()) + igk_loc;
            /* vs = {r, theta, phi} */
            auto vs = SHT::spherical_coordinates(kp__->gkvec().gkvec_cart(igk));
            int idx_gk = static_cast<int>((vs[0] / ctx_.gk_cutoff()) * (rad_int__[0][0].num_points() - 1));
            double dgk = vs[0] - rad_int__[0][0].radial_grid()[idx_gk];
            /* compute real spherical harmonics for G+k vector */
            SHT::spherical_harmonics(lmax__, vs[1], vs[2], &gkvec_rlm[0]);

            int n{0};
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                double phase = twopi * (kp__->gkvec().gkvec(igk) * unit_cell_.atom(ia).position());
                double_complex phase_factor = std::exp(double_complex(0.0, -phase));

                auto& atom_type = unit_cell_.atom(ia).type();
                //for (size_t i = 0; i < atom_type.pp_desc().atomic_pseudo_wfs_.size(); i++) {
                for (int l = 0; l <= lmax__; l++) {
                    //int l = atom_type.pp_desc().atomic_pseudo_wfs_[i].first;
                    for (int m = -l; m <= l; m++) {
                        int lm = Utils::lm_by_l_m(l, m);
                        double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                        //phi.pw_coeffs().prime(igk_loc, n++) = z * phase_factor * gkvec_rlm[lm] * rad_int__[atom_type.id()][i](vs[0]);
                        phi.pw_coeffs().prime(igk_loc, n++) = z * phase_factor * gkvec_rlm[lm] * rad_int__[atom_type.id()][l](idx_gk, dgk);
                    }
                }
            }
        }
    }
    
    assert(kp__->num_gkvec() > num_phi + 10);
    double norm = std::sqrt(1.0 / kp__->num_gkvec()); 
    std::vector<double_complex> v(kp__->num_gkvec());
    for (int i = 0; i < num_phi - num_ao__; i++) {
        std::generate(v.begin(), v.end(), []{return type_wrapper<double_complex>::random();});
        v[0]     = 0.0;
        v[i + 1] = 1.0 / norm;
        v[i + 2] = 0.5 / norm;
        v[i + 3] = 0.25 / norm;
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++) {
            /* global index of G+k vector */
            int igk = kp__->gkvec().gvec_offset(kp__->comm().rank()) + igk_loc;
            phi.pw_coeffs().prime(igk_loc, num_ao__ + i) = v[igk] * norm;
        }
    }

    /* short notation for number of target wave-functions */
    int num_bands = ctx_.num_fv_states();

    Hloc_operator hloc(ctx_.fft_coarse(), kp__->gkvec_vloc(), ctx_.mpi_grid_fft_vloc().communicator(1 << 1), 
                       ctx_.num_mag_dims(), ctx_.gvec_coarse(), effective_potential__, effective_magnetic_field__);

    ctx_.fft_coarse().prepare(kp__->gkvec().partition());
    
    D_operator<T> d_op(ctx_, kp__->beta_projectors());
    Q_operator<T> q_op(ctx_, kp__->beta_projectors());

    /* allocate wave-functions */
    wave_functions hphi(ctx_.processing_unit(), kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions ophi(ctx_.processing_unit(), kp__->comm(), kp__->gkvec(), num_phi);
    wave_functions wf_tmp(ctx_.processing_unit(), kp__->comm(), kp__->gkvec(), num_phi);

    int bs = ctx_.cyclic_block_size();
    auto mem_type = (std_evp_solver().type() == ev_magma) ? memory_t::host_pinned : memory_t::host;
    dmatrix<T> hmlt(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> evec(num_phi, num_phi, ctx_.blacs_grid(), bs, bs, mem_type);
    dmatrix<T> hmlt_old;

    std::vector<double> eval(num_bands);
    
    kp__->beta_projectors().prepare();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        phi.allocate_on_device();
        phi.copy_to_device(0, num_phi);
        hphi.allocate_on_device();
        ophi.allocate_on_device();
        wf_tmp.allocate_on_device();
        evec.allocate(memory_t::device);
        hmlt.allocate(memory_t::device);
    }
    #endif
    
    if (ctx_.control().print_checksum_) {
        auto cs = phi.checksum(0, num_phi);
        DUMP("checksum(phi): %18.10f %18.10f", cs.real(), cs.imag());
    }

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
    {
        /* apply Hamiltonian and overlap operators to the new basis functions */
        apply_h_o<T>(kp__, ispn, 0, num_phi, phi, hphi, ophi, hloc, d_op, q_op);
        
        /* do some checks */
        if (ctx_.control().verification_ >= 1) {
            set_subspace_mtrx<T>(0, num_phi, phi, ophi, hmlt, hmlt_old);
            double max_diff = Utils::check_hermitian(hmlt, num_phi);
            if (max_diff > 1e-12) {
                std::stringstream s;
                s << "overlap matrix is not hermitian, max_err = " << max_diff;
                TERMINATE(s);
            }
            std::vector<double> eo(num_phi);
            if (std_evp_solver().solve(num_phi, num_phi, hmlt.template at<CPU>(), hmlt.ld(),
                                       eo.data(), evec.template at<CPU>(), evec.ld(),
                                       hmlt.num_rows_local(), hmlt.num_cols_local())) {
                std::stringstream s;
                s << "error in diagonalziation";
                TERMINATE(s);
            }
            if (kp__->comm().rank() == 0) {
                printf("[verification] minimum eiegen-value of the overlap matrix: %18.12f\n", eo[0]);
            }
            if (eo[0] < 0) {
                TERMINATE("overlap matrix is not positively defined");
            }
        }
        
        orthogonalize<T>(0, num_phi, phi, hphi, ophi, hmlt, wf_tmp);

        /* setup eigen-value problem */
        set_subspace_mtrx<T>(0, num_phi, phi, hphi, hmlt, hmlt_old);

        /* solve generalized eigen-value problem with the size N */
        if (std_evp_solver().solve(num_phi, num_bands, hmlt.template at<CPU>(), hmlt.ld(),
                                   eval.data(), evec.template at<CPU>(), evec.ld(),
                                   hmlt.num_rows_local(), hmlt.num_cols_local())) {
            std::stringstream s;
            s << "error in diagonalziation";
            TERMINATE(s);
        }

        if (ctx_.control().print_checksum_) {
            auto cs = evec.checksum();
            kp__->comm().allreduce(&cs, 1);
            DUMP("checksum(evec): %18.10f", std::abs(cs));
            double cs1{0};
            for (int i = 0; i < num_bands; i++) {
                cs1 += eval[i];
            }
            DUMP("checksum(eval): %18.10f", cs1);
        }

        if (ctx_.control().verbosity_ >= 3 && kp__->comm().rank() == 0) {
            for (int i = 0; i < num_bands; i++) {
                DUMP("eval[%i]=%20.16f", i, eval[i]);
            }
        }
        
        /* compute wave-functions */
        /* \Psi_{i} = \sum_{mu} \phi_{mu} * Z_{mu, i} */
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            kp__->spinor_wave_functions(ispn).pw_coeffs().allocate_on_device();
        }
        #endif

        transform<T>(phi, 0, num_phi, evec, 0, 0, kp__->spinor_wave_functions(ispn), 0, num_bands);

        if (ctx_.control().print_checksum_) {
            auto cs = kp__->spinor_wave_functions(ispn).checksum(0, num_bands);
            DUMP("checksum(spinor_wave_functions): %18.10f %18.10f", cs.real(), cs.imag());
        }

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            kp__->spinor_wave_functions(ispn).pw_coeffs().copy_to_host(0, num_bands);
            kp__->spinor_wave_functions(ispn).pw_coeffs().deallocate_on_device();
        }
        #endif

        for (int j = 0; j < ctx_.num_fv_states(); j++) {
            kp__->band_energy(j + ispn * ctx_.num_fv_states()) = eval[j];
        }
    }

    kp__->beta_projectors().dismiss();

    ctx_.fft_coarse().dismiss();
}
