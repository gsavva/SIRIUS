#include "md_extrapolation.hpp"
#include "context/simulation_context.hpp"
#include "core/env/env.hpp"
#include "core/la/eigenproblem.hpp"
#include "core/la/linalg_base.hpp"
#include "core/rte/rte.hpp"
#include "core/wf/wave_functions.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "k_point/k_point.hpp"
#include "k_point/k_point_set.hpp"
#include "core/memory.hpp"
#include "potential/potential.hpp"
#include <stdexcept>
#include "core/env/env.hpp"

namespace sirius {

namespace md {

/** Aligns subspace between two wave-functions.
 *  R = arg min_R' || Ψᵒ R' -  Ψⁱ ||
 *  O = <Ψᵒ|S|Ψⁱ>
 *  R = (O O*)^(-1/2) O = U Vt
 *  where O = U s Vt, is the singular value decomposition of O
 *
 *  Returns: Ψᵒ ← Ψᵒ R
 */
template <class T>
void
subspace_alignment(Simulation_context& ctx, wf::Wave_functions<T>& wf_out, wf::Wave_functions<T> const& wf_in,
                   K_point<T>& kp, Hamiltonian0<T>& H0)
{
    auto Hk               = H0(kp);
    const int num_spinors = (ctx.num_mag_dims() == 1) ? 2 : 1;
    const bool nc_mag     = ctx.num_mag_dims() == 3;
    auto num_wf           = wf::num_bands(wf_in.num_wf());
    auto num_mag_dims     = wf::num_mag_dims(ctx.num_mag_dims());
    int n                 = wf_in.num_wf();

    // psi_tmp <- wf_in
    wf::Wave_functions<T> psi_tmp(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
    for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
        wf::copy(memory_t::host, wf_out, wf::spin_index(ispin_step), wf::band_range(n), psi_tmp,
                 wf::spin_index(ispin_step), wf::band_range(n));
    }
    // compute <wf_out|S|wf_in>
    auto sphi       = std::make_shared<wf::Wave_functions<T>>(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
    auto proc_mem_t = ctx.processing_unit_memory_t();
    std::array<la::dmatrix<std::complex<T>>, 2> ovlp;
    {
        auto sphi_guard   = sphi->memory_guard(proc_mem_t, wf::copy_to::host);
        auto wf_in_guard  = wf_in.memory_guard(proc_mem_t, wf::copy_to::device);
        auto wf_out_guard = wf_out.memory_guard(proc_mem_t, wf::copy_to::device);

        for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
            ovlp[ispin_step] = la::dmatrix<std::complex<T>>(num_wf, num_wf, memory_t::host);
            auto br          = wf::band_range(0, num_wf);
            auto sr          = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispin_step);
            if (ctx.gamma_point()) {
                Hk.template apply_s<double>(sr, br, wf_in, *sphi);
            } else {
                Hk.template apply_s<std::complex<double>>(sr, br, wf_in, *sphi);
            }
            /*   compute overlap <wf_out|S|wf_in>   */
            wf::inner(ctx.spla_context(), proc_mem_t, sr, wf_out, br, *sphi, br, ovlp[ispin_step], 0, 0);
            // compute SVD
            la::dmatrix<std::complex<T>> U(n, n);
            la::dmatrix<std::complex<T>> Vt(n, n);
            mdarray<double, 1> s({n}, "singular values");
            auto la = la::lib_t::lapack;
            la::wrap(la).gesvd('A', 'A', ovlp[ispin_step], s, U, Vt);
            // rotation matrix, to align wf_out
            la::dmatrix<std::complex<T>> Rot(n, n);
            auto ptr_one  = &la::constant<std::complex<T>>::one();
            auto ptr_zero = &la::constant<std::complex<T>>::zero();
            la::wrap(la::lib_t::blas)
                    .gemm('N', 'N', n, n, n, ptr_one, U.at(memory_t::host), U.ld(), Vt.at(memory_t::host), Vt.ld(),
                          ptr_zero, Rot.at(memory_t::host), Rot.ld());
            // call spla to transform wfc
            wf::transform(ctx.spla_context(), memory_t::host, Rot, 0, 0,               // irow0, jcol0
                          1.0, psi_tmp, wf::spin_index(ispin_step), wf::band_range(n), // input
                          0.0,                                                         // beta
                          wf_out, wf::spin_index(ispin_step), wf::band_range(n)        // output
            );
        }
    }
}

LinearWfcExtrapolation::LinearWfcExtrapolation()
{
    if (env::skip_wfct_extrapolation()) {
        this->skip_ = true;
    }
}

void
LinearWfcExtrapolation::push_back_history(const K_point_set& kset__, const Density& density__,
                                          const Potential& potential__)
{
    /*
      copy
      - plane-wave coefficients
      - band energies
      into internal data structures
     */

    if (this->skip_) {
        return;
    }

    int nbnd  = kset__.ctx().num_bands();
    auto& ctx = kset__.ctx();
    kp_map<std::shared_ptr<wf::Wave_functions<double>>> wfc_k;
    kp_map<mdarray<double, 2>> e_k;
    auto num_wf       = wf::num_bands(nbnd);
    auto num_mag_dims = wf::num_mag_dims(ctx.num_mag_dims());

    for (auto it : kset__.spl_num_kpoints()) {

        // wf::Wave_functions<double>
        auto& kp = *kset__.get<double>(it.i);
        auto wf_tmp =
                std::make_shared<wf::Wave_functions<double>>(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);
        // const auto spin_up = wf::spin_index(0);
        const auto& wfc = kp.spinor_wave_functions();
        int num_sc      = wfc.num_sc();
        mdarray<double, 2> ek_loc({2, nbnd});
        for (int i = 0; i < num_sc; ++i) {
            wf::copy(memory_t::host, kp.spinor_wave_functions(), wf::spin_index(i), wf::band_range(nbnd), *wf_tmp,
                     wf::spin_index(i), wf::band_range(nbnd));

            for (int ie = 0; ie < nbnd; ++ie) {
                ek_loc(i, ie) = kp.band_energy(ie, i);
            }
        }
        e_k[it.i]   = std::move(ek_loc);
        wfc_k[it.i] = wf_tmp;
    }

    wfc_.push_front(wfc_k);
    band_energies_.push_front(std::move(e_k));

    if (wfc_.size() > 2) {
        wfc_.pop_back();
        band_energies_.pop_back();
    }
}

void
LinearWfcExtrapolation::extrapolate(K_point_set& kset__, Density& density__, Potential& potential__) const
{
    auto& ctx = kset__.ctx();

    if (wfc_.size() < 2 || this->skip_) {
        // skip extrapolation, but regenerate density with updated ionic positions
        density__.generate<double>(kset__, ctx.use_symmetry(), true /* add core */, true /* transform to rg */);
        potential__.generate(density__, ctx.use_symmetry(), true);
        return;
    }

    if (wfc_.size() != 2) {
        throw std::runtime_error("expected size =2");
    }

    std::stringstream ss;
    ss << "extrapolate";
    ctx.message(1, __func__, ss);
    /* H0 */
    auto H0 = Hamiltonian0<double>(potential__, false);

    /* true if this is a non-collinear case */
    const bool nc_mag = ctx.num_mag_dims() == 3;
    if (nc_mag) {
        RTE_THROW("non-collinear case not implemented");
    }

    const int num_spinors = (ctx.num_mag_dims() == 1) ? 2 : 1;

    // Ψ⁽ⁿ⁺¹⁾ = Löwdin(2 Ψ⁽ⁿ⁾ - Ψ⁽ⁿ⁻¹⁾)
    for (auto it : kset__.spl_num_kpoints()) {
        auto& kp          = *kset__.get<double>(it.i);
        auto& wfc         = kp.spinor_wave_functions();
        int num_sc        = wfc.num_sc(); // number of spin components
        auto num_wf       = wf::num_bands(wfc.num_wf());
        auto num_mag_dims = wf::num_mag_dims(ctx.num_mag_dims());

        wf::Wave_functions<double> psi_tilde(kp.gkvec_sptr(), num_mag_dims, num_wf, memory_t::host);

        auto& wfc_prev = wfc_.back().at(it.i);
        subspace_alignment(ctx, wfc, *wfc_prev, kp, H0);

        for (int ispn = 0; ispn < num_sc; ++ispn) {
            auto sp             = wf::spin_index(ispn);
            auto* psi_tilde_ptr = psi_tilde.pw_coeffs(sp).host_data();
            auto* psi_ptr       = wfc.pw_coeffs(sp).host_data();
            auto* psi_prev_ptr  = wfc_prev->pw_coeffs(sp).host_data();
#pragma omp parallel for
            for (auto i = 0ul; i < wfc.pw_coeffs(sp).size(); ++i) {
                *(psi_tilde_ptr + i) = 2.0 * (*(psi_ptr + i)) - (*(psi_prev_ptr + i));
            }
        }
        /* Löwdin orthogonalization */
        // orthogonalize
        auto Hk         = H0(kp);
        auto proc_mem_t = ctx.processing_unit_memory_t();
        std::array<la::dmatrix<std::complex<double>>, 2> ovlp_spinor;
        // compute S|psi~>
        {
            auto sphi = std::make_shared<wf::Wave_functions<double>>(kp.gkvec_sptr(), num_mag_dims, num_wf, proc_mem_t);
            auto phi_guard = psi_tilde.memory_guard(proc_mem_t, wf::copy_to::device);

            for (auto ispin_step = 0; ispin_step < num_spinors; ++ispin_step) {
                ovlp_spinor[ispin_step] = la::dmatrix<std::complex<double>>(num_wf, num_wf, memory_t::host);
                auto br                 = wf::band_range(0, num_wf);
                auto sr                 = nc_mag ? wf::spin_range(0, 2) : wf::spin_range(ispin_step);
                if (ctx.gamma_point()) {
                    Hk.apply_s<double>(sr, br, psi_tilde, *sphi);
                } else {
                    Hk.apply_s<std::complex<double>>(sr, br, psi_tilde, *sphi);
                }
                /*   compute overlap <psi~|S|psi~>   */
                wf::inner(kset__.ctx().spla_context(), proc_mem_t, sr, psi_tilde, br, *sphi, br,
                          ovlp_spinor[ispin_step], 0, 0);
            }
        }
        la::lib_t la{la::lib_t::blas};

        for (int ispn = 0; ispn < num_sc; ++ispn) {
            int ovlp_index = nc_mag ? 0 : ispn;
            /*   compute eig(<psi|S|psi>)   */
            la::dmatrix<std::complex<double>> Z(num_wf, num_wf);
            la::Eigensolver_lapack lapack_ev;
            std::vector<double> eval(num_wf);
            lapack_ev.solve(num_wf, ovlp_spinor[ovlp_index], eval.data(), Z);

            std::vector<std::complex<double>> d(num_wf);
            for (int i = 0; i < num_wf; ++i) {
                d[i] = 1 / sqrt(eval[i]);
            }

            /*   Zi2 <- U * diag(1/sqrt(eval))   */
            la::dmatrix<std::complex<double>> Zi2(num_wf, num_wf);
            la::wrap(la).dgmm('r', num_wf, num_wf, Z.at(memory_t::host), Z.ld(), d.data(), 1, Zi2.at(memory_t::host),
                              Zi2.ld());

            /*   R <- (Zi2 * U.H) = U * diag(1/sqrt(eval)) * U.H   */
            la::dmatrix<std::complex<double>> R(num_wf, num_wf);
            auto ptr_one  = &la::constant<std::complex<double>>::one();
            auto ptr_zero = &la::constant<std::complex<double>>::zero();
            la::wrap(la).gemm('N', 'C', num_wf, num_wf, num_wf, ptr_one, Zi2.at(memory_t::host), Zi2.ld(),
                              Z.at(memory_t::host), Z.ld(), ptr_zero, R.at(memory_t::host), R.ld());
            /*   orthogonalize current wfc: to kset.wfc <- psi_tilde * R   */
            const std::complex<double>* psi_tilde_ptr =
                    psi_tilde.at(memory_t::host, 0, wf::spin_index(ispn), wf::band_index(0));
            int num_gkvec_loc = kp.num_gkvec_loc();
            auto wf_i_ptr     = wfc.at(memory_t::host, 0, wf::spin_index(ispn), wf::band_index(0));
            la::wrap(la).gemm('N', 'N', num_gkvec_loc, num_wf, num_wf, ptr_one, psi_tilde_ptr, psi_tilde.ld(),
                              R.at(memory_t::host), R.ld(), ptr_zero, wf_i_ptr, num_gkvec_loc);
        }
    }

    /*   extrapolate band energies: Ɛ⁽ⁿ⁺¹⁾ = 2 Ɛ⁽ⁿ⁾ - Ɛ⁽ⁿ⁻¹⁾   */
    for (auto it : kset__.spl_num_kpoints()) {
        auto& kp   = *kset__.get<double>(it.i);
        int nbnd   = kset__.ctx().num_bands();
        int num_sc = kset__.ctx().num_spins();
        for (int ispn = 0; ispn < num_sc; ++ispn) {
            for (int j = 0; j < nbnd; ++j) {
                double enew = 2 * kp.band_energy(j, ispn) - band_energies_.back().at(it.i)(ispn, j);
                kp.band_energy(j, ispn, enew);
            }
        }
    }
    // compute occupation numbers from new band energies
    kset__.sync_band<double, sync_band_t::energy>();
    kset__.find_band_occupancies<double>();

    // generate density
    density__.generate<double>(kset__, ctx.use_symmetry(), true /* add core */, true /* transform to rg */);
    // generate potential
    potential__.generate(density__, ctx.use_symmetry(), true);
}

} // namespace md

} // namespace sirius