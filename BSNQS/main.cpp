#include <iostream>
#include <ctime>
#include "src/library.h"


int main(int argc, char* argv[]){

  //Prepares the simulation
  auto options = setOptions(argc, argv);  //Load parameters
  WaveFunction* Psi;  // 𝜳(𝒗,𝜙,𝜶)
  SpinHamiltonian* H;  // ℍ̂

  //Defines the wavefunction
  if(options["ansatz"] == "JasNN"){  //Nearest-neighbors Jastrow Quantum State

    if(options["filename"] != "None")
      Psi = new JasNN(options["filename"], std::stoi(options["if_phi_neq_zero"]));
    else
      Psi = new JasNN(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]));

  }
  else if(options["ansatz"] == "RBM"){  //Neural Network Quantum State

    if(options["filename"] != "None")
      Psi = new RBM(options["filename"]);
    else
      Psi = new RBM(std::stoi(options["n_visible"]), std::stoi(options["density"]));

  }
  else if(options["ansatz"] == "uRBM"){  //Shadow (quasi)-uRBM Quantum State

    if(options["filename"] != "None")
      Psi = new quasi_uRBM(options["filename"], std::stoi(options["if_phi_neq_zero"]));
    else
      Psi = new quasi_uRBM(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]));

  }
  else if(options["ansatz"] == "BS_NNQS"){  //Baeriswyl-Shadow Quantum State

    if(options["filename"] != "None")
      Psi = new BS_NNQS(options["filename"], std::stoi(options["if_phi_neq_zero"]));
    else
      Psi = new BS_NNQS(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]));

  }
  else{

    std::cerr << "##InputError: type of variational wave function not allowed." << std::endl;
    std::cerr << "  Failed to construct the wave function ansatz." << std::endl;
    std::abort();

  }

  //Defines the Quantum Hamiltonian
  if(options["model"] == "Ising1d"){
    H = new Ising1d(std::stoi(options["n_visible"]),    //TFI model in 𝐝 = 𝟏 with 𝑃𝐵𝐶
                    std::stod(options["h_field"]),
                    std::stod(options["J"]));
  }
  else if(options["model"] == "Heisenberg1d"){  //Heisenberg model in 𝐝 = 𝟏 with 𝑃𝐵𝐶
    H = new Heisenberg1d(std::stoi(options["n_visible"]),
                         std::stod(options["h_field"]),
                         std::stod(options["Jx"]),
                         std::stod(options["Jy"]),
                         std::stod(options["Jz"]));
  }
  else{

    std::cerr << "##InputError: Quantum Hamiltonian not allowed." << std::endl;
    std::cerr << "  Failed to construct the model." << std::endl;
    std::abort();

  }

  //Defines the VMC sampler and the output files
  VMC_Sampler sampler(*Psi, *H);

  if(std::stoi(options["if_write_move_statistics"]))
    sampler.setFile_Move_Statistics(options["file_info"]);
  if(std::stoi(options["if_write_MCMC_config"]))
    sampler.setFile_MCMC_Config(options["file_info"]);
  if(std::stoi(options["if_write_final_config"]))
    sampler.setFile_final_Config(options["file_info"]);
  if(std::stoi(options["if_write_opt_obs"]))
    sampler.setFile_opt_Obs(options["file_info"]);
  if(std::stoi(options["if_write_block_obs"]))
    sampler.setFile_block_Obs(options["file_info"]);
  if(std::stoi(options["if_write_opt_params"]))
    sampler.setFile_opt_Params(options["file_info"]);
  if(std::stoi(options["if_write_all_params"]))
    sampler.setFile_all_Params(options["file_info"]);
  if(std::stoi(options["if_write_qgt_matrix"]))
    sampler.setFile_QGT_matrix(options["file_info"]);
  if(std::stoi(options["if_write_qgt_cond"]))
    sampler.setFile_QGT_cond(options["file_info"]);
  if(std::stoi(options["if_write_qgt_eigen"]))
    sampler.setFile_QGT_eigen(options["file_info"]);

  //Sets the simulation parameters
  if(std::stoi(options["if_hidden_off"]))
    sampler.ShutDownHidden(std::stoi(options["orientation_bias"]));
  if(std::stod(options["if_QGT_reg"]) != 0.0){

    if(std::stod(options["if_QGT_reg"]) == 1.0)
      sampler.setQGTReg();
    else
      sampler.setQGTReg(std::stod(options["if_QGT_reg"]));

  }
  if(std::stoi(options["_if_extra_hidden_sum"]))
    sampler.setExtraHiddenSum(std::stoi(options["N_extra_sum"]), std::stoi(options["N_blks_hidden"]));

  //Chooses the appropriate algorithm
  std::cout << "\n#Start the Variational Monte Carlo optimization algorithm." << std::endl;
  if(std::stoi(options["if_vmc"])){

    options["n_optimization_steps"] = "1";
    std::cout << " Quantum properties calculation via a simple Monte Carlo (𝑴𝑪)." << std::endl;

  }
  else if(std::stoi(options["if_imaginary_time"])){

    sampler.setImagTimeDyn(std::stod(options["delta"]));
    std::cout << " Searching the Ground State via imaginary Time-Dependent Variational Monte Carlo (𝒊𝑻𝑫𝑽𝑴𝑪)." << std::endl;

  }
  else if(std::stoi(options["if_real_time"])){

      if(options["model"] == "Ising1d")
        H -> Quench(std::stod(options["h_quench"]));
      else if(options["model"] == "Heisenberg1d")
        H -> Quench(std::stod(options["Jz_quench"]));
      sampler.setRealTimeDyn(std::stod(options["delta"]));
      std::cout << " Performing the dynamics via Time-Dependent Variational Monte Carlo (𝑻𝑫𝑽𝑴𝑪)." << std::endl;

  }
  else{

    std::cerr << "##InputError: it is mandatory to make a choice regarding the type of variational optimization!" << std::endl;
    std::cerr << "   Failed to choose the appropriate algorithm." << std::endl;
    std::abort();

  }

  sampler.setStepParameters(std::stoi(options["n_sweeps"]), std::stoi(options["n_blks"]),
                            std::stoi(options["n_equilibration_steps"]), std::stoi(options["n_bunch"]),
                            std::stoi(options["bunch_dimension"]), std::stod(options["p_equal_site"]),
                            std::stod(options["p_visible_nn"]), std::stod(options["p_hidden_nn"]));
  
  /*##############################################*/
  /*  PERFORMS THE CHOSEN OPTIMIZATION ALGORTIHM  */
  /*##############################################*/
  std::clock_t t_start = std::clock();  //Start to measure CPU time

  //Initial thermalization phase
  sampler.Init_Config();  //Random initial configuration at time 0
  std::cout << " Thermalization ... ";
  std::flush(std::cout);
  for(unsigned int init_eq_step = 0; init_eq_step < std::stoi(options["n_init_eq_steps"]); init_eq_step++){

    //Moves configuration without measurements
    sampler.Make_Sweep();

  }
  std::cout << "done" << std::endl;

  //Time evolution of the variational quantum state
  for(unsigned int tdvmc_step = 0; tdvmc_step < std::stoi(options["n_optimization_steps"]); tdvmc_step++){

    //Print options
    if(std::stoi(options["if_restart"]) == 1)
      sampler.setRestartFromConfig();
    if(!std::stoi(options["if_real_time"])){

      std::cout << " 𝝉 = " << std::setprecision(2) << std::fixed << std::stod(options["delta"]) * (tdvmc_step + 1) << " ... ";
      std::flush(std::cout);

    }
    else{

      std::cout << " 𝒕 = " << std::setprecision(2) << std::fixed << std::stod(options["delta"]) * (tdvmc_step + 1) << " ... ";
      std::flush(std::cout);

    }

    //𝐕𝐌𝐂 step
    sampler.VMC_Step();

    //Updates variational parameters
    if(options["integrator"] == "Euler"){  //Euler ODE method

      sampler.Euler();

    }
    else if(options["integrator"] == "Heun"){  //Heun ODE method

      sampler.Heun();

    }
    else if(options["integrator"] == "RK4"){  //4th order Runge Kutta ODE method

      sampler.RK4();

    }
    else{

      std::cerr << "##InputError: choosen ODE integrator not available!" << std::endl;
      std::cerr << "  Failed to choose the appropriate ODE integration method." << std::endl;
      std::abort();

    }

    //Prints results
    sampler.Write_Move_Statistics(tdvmc_step);
    sampler.Write_final_Config(tdvmc_step);
    sampler.Write_Quantum_properties(tdvmc_step);
    sampler.Write_all_Params(tdvmc_step);

    //Prepares all stuff for the next VMC
    sampler.Reset_Moves_Statistics();
    sampler.Reset();

    std::cout << "done" << std::endl;

  }

  std::clock_t t_end = std::clock();  //Ends to measure CPU time
  long double time = (t_end - t_start)*1.0 / CLOCKS_PER_SEC;  //CPU time in seconds

  if(!std::stoi(options["if_vmc"]))
    std::cout << " Quantum State optimized." << std::endl;
  std::cout << " CPU time for the optimization of the Quantum State: \t" << std::setprecision(4) << std::fixed << time << " (s)." << std::endl;

  sampler.Write_opt_Params();
  sampler.CloseFile();
  std::cout << std::endl;
  delete Psi;
  delete H;

  return 0;


}
