// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file simulation_parameters.cpp
 *
 *  \brief Contains implementation of sirius::Simulation_parameters class.
 */

#include "simulation_parameters.hpp"
#include "mpi/communicator.hpp"

/// Json dictionary containing the options given by the interface.
#include "context/runtime_options_json.hpp"
#include "context/input_schema.hpp"

namespace sirius {

/// Compose JSON dictionary with default parameters based on input schema.
/** Traverse the JSON schema and add nodes with default parameters to the output dictionary. The nodes without
 *  default parameters are ignored. Still, user has a possibility to add the missing nodes later by providing a
 *  corresponding input JSON dictionary. See compose_json() function. */
void compose_default_json(nlohmann::json const& schema__, nlohmann::json& output__)
{
    for (auto it: schema__.items()) {
        auto key = it.key();
        /* this is a final node with the description of the data type */
        if (it.value().contains("type") && it.value()["type"] != "object") {
            /* check if default parameter is present */
            if (it.value().contains("default")) {
                output__[key] = it.value()["default"];
            }
        } else { /* otherwise continue to traverse the shcema */
            if (!output__.contains(key)) {
                output__[key] = nlohmann::json{};
            }
            if (it.value().contains("properties")) {
                compose_default_json(it.value()["properties"], output__[key]);
            }
        }
    }
}

/// Append the input dictionary to the existing dictionary.
/** Use JSON schema to traverse the existing dictionary and add on top the values from the input dictionary. In this
 *  way we can add missing nodes which were not defined in the existing dictionary. */
void compose_json(nlohmann::json const& schema__, nlohmann::json const& in__, nlohmann::json& inout__)
{
    for (auto it: schema__.items()) {
        auto key = it.key();
        /* this is a final node with the description of the data type */
        if (it.value().contains("type") && it.value()["type"] != "object") {
            if (in__.contains(key)) {
                /* copy the new input */
                inout__[key] = in__[key];
            }
        } else { /* otherwise continue to traverse the shcema */
            if (it.value().contains("properties")) {
                compose_json(it.value()["properties"], in__.contains(key) ? in__[key] : nlohmann::json{}, inout__[key]);
            } else if (in__.contains(key)) {
                inout__[key] = in__[key];
            } else {
                inout__[key] = nlohmann::json();
            }
        }
    }
}

Config::Config()
{
    /* initialize JSON dictionary with default parameters */
    compose_default_json(sirius::input_schema["properties"], this->dict_);
}

void Config::import(nlohmann::json const& in__)
{
    /* overwrite the parameters by the values from the input dictionary */
    compose_json(sirius::input_schema["properties"], in__, this->dict_);
}

/// Get all possible options for initializing sirius. It is a json dictionary.
json const& get_options_dictionary()
{
    if (all_options_dictionary_.size() == 0) {
        throw std::runtime_error("Dictionary not initialized\n");
    }
    return all_options_dictionary_;
}

void Simulation_parameters::import(std::string const& str__)
{
    auto json = utils::read_json_from_file_or_string(str__);
    import(json);
}

void Simulation_parameters::import(json const& dict__)
{
    cfg_.import(dict__);
    /* read unit cell */
    unit_cell_input_.read(dict__);
    /* read parameters of iterative solver */
    iterative_solver_input_.read(dict__);
    /* read controls */
    control_input_.read(dict__);
    /* read parameters */
    parameters_input_.read(dict__);
    /* read settings */
    //settings_input_.read(dict__);
    /* read hubbard parameters */
    hubbard_input_.read(dict__);
    /* read nlcg parameters */
    nlcg_input_.read(dict__);
}

void Simulation_parameters::import(cmd_args const& args__) // TODO: somehow redesign to use json_pointer
{
    control_input_.processing_unit_ = args__.value("control.processing_unit", control_input_.processing_unit_);
    control_input_.mpi_grid_dims_   = args__.value("control.mpi_grid_dims", control_input_.mpi_grid_dims_);
    control_input_.std_evp_solver_name_ =
        args__.value("control.std_evp_solver_name", control_input_.std_evp_solver_name_);
    control_input_.gen_evp_solver_name_ =
        args__.value("control.gen_evp_solver_name", control_input_.gen_evp_solver_name_);
    control_input_.fft_mode_     = args__.value("control.fft_mode", control_input_.fft_mode_);
    control_input_.memory_usage_ = args__.value("control.memory_usage", control_input_.memory_usage_);
    control_input_.verbosity_    = args__.value("control.verbosity", control_input_.verbosity_);
    control_input_.verification_ = args__.value("control.verification", control_input_.verification_);

    parameters_input_.ngridk_      = args__.value("parameters.ngridk", parameters_input_.ngridk_);
    parameters_input_.gamma_point_ = args__.value("parameters.gamma_point", parameters_input_.gamma_point_);
    parameters_input_.pw_cutoff_   = args__.value("parameters.pw_cutoff", parameters_input_.pw_cutoff_);

    iterative_solver_input_.early_restart_ = args__.value("iterative_solver.early_restart", iterative_solver_input_.early_restart_);
    //mixer_input_.beta_ = args__.value("mixer.beta", mixer_input_.beta_);
    //mixer_input_.type_ = args__.value("mixer.type", mixer_input_.type_);
}

void Simulation_parameters::core_relativity(std::string name__)
{
    parameters_input_.core_relativity_ = name__;
    core_relativity_ = get_relativity_t(name__);
}

void Simulation_parameters::valence_relativity(std::string name__)
{
    parameters_input_.valence_relativity_ = name__;
    valence_relativity_ = get_relativity_t(name__);
}

void Simulation_parameters::processing_unit(std::string name__)
{
    /* set the default value */
    if (name__ == "") {
        if (acc::num_devices() > 0) {
            name__ = "gpu";
        } else {
            name__ = "cpu";
        }
    }
    control_input_.processing_unit_ = name__;
    processing_unit_ = get_device_t(name__);
}

void Simulation_parameters::smearing(std::string name__)
{
    parameters_input_.smearing_ = name__;
    smearing_ = smearing::get_smearing_t(name__);
}

void Simulation_parameters::print_options() const
{
    json const& dict = get_options_dictionary();

    if (Communicator::world().rank() == 0) {
        std::printf("The SIRIUS library or the mini apps can be initialized through the interface\n");
        std::printf("using the API directly or through a json dictionary. The following contains\n");
        std::printf("a description of all the runtime options, that can be used directly to\n");
        std::printf("initialize SIRIUS.\n");

        for (auto& el : dict.items()) {
            std::cout << "============================================================================\n";
            std::cout << "                                                                              ";
            std::cout << "                      section : " << el.key() << "                             \n";
            std::cout << "                                                                            \n";
            std::cout << "============================================================================\n";

            for (size_t s = 0; s < dict[el.key()].size(); s++) {
                std::cout << "name of the option : " << dict[el.key()][s]["name"].get<std::string>() << std::endl;
                std::cout << "description : " << dict[el.key()][s]["description"].get<std::string>() << std::endl;
                if (dict[el.key()][s].count("possible_values")) {
                    const auto& v = dict[el.key()][s]["description"].get<std::vector<std::string>>();
                    std::cout << "possible values : " << v[0];
                    for (size_t st = 1; st < v.size(); st++)
                        std::cout << " " << v[st];
                }
                std::cout << "default value : " << dict[el.key()]["default_values"].get<std::string>() << std::endl;
            }
        }
    }
    Communicator::world().barrier();
}

void Simulation_parameters::electronic_structure_method(std::string name__)
{
    parameters_input_.electronic_structure_method_ = name__;

    std::map<std::string, electronic_structure_method_t> m = {
        {"full_potential_lapwlo", electronic_structure_method_t::full_potential_lapwlo},
        {"pseudopotential", electronic_structure_method_t::pseudopotential}};

    if (m.count(name__) == 0) {
        std::stringstream s;
        s << "wrong type of electronic structure method: " << name__;
        TERMINATE(s);
    }
    electronic_structure_method_ = m[name__];
}
} // namespace sirius
