#include "context/simulation_context.hpp"
#include "core/wf/wave_functions.hpp"
#include "hamiltonian/hamiltonian.hpp"
#include "k_point/k_point_set.hpp"
#include "core/memory.hpp"
#include "potential/potential.hpp"

namespace sirius {
namespace md {

class MDExtrapolation
{
  public:
    virtual void
    extrapolate(K_point_set&, Density&, Potential&) const = 0;
    virtual void
    push_back_history(const K_point_set&, const Density&, const Potential&) = 0;
};

template <class T>
using kp_map = std::map<kp_index_t::global, T>;

class LinearWfcExtrapolation : MDExtrapolation
{
  public:
    LinearWfcExtrapolation();
    /// store plane-wave and band energies of the current time-step
    void
    push_back_history(const K_point_set& kset__, const Density& density__, const Potential& potential__) override;
    /// extrapolate wave-functions and band-energies (occupation numbers), generate new density and potential
    void
    extrapolate(K_point_set& kset__, Density& density__, Potential& potential__) const override;

  private:
    /// plane-wave coefficients
    std::list<kp_map<std::shared_ptr<wf::Wave_functions<double>>>> wfc_;
    /// store band energies as complex, they need to be transformed into a
    /// pseudo-Hamiltonian matrix, which is complex
    using e_vec_t = mdarray<std::complex<double>, 1>;
    /// spin up/dn band energy
    using e_vec_kp_t = std::array<e_vec_t, 2>;
    std::list<kp_map<e_vec_kp_t>> band_energies_;
    /// skip extrapolation
    bool skip_{false};
};

} // namespace md
} // namespace sirius
