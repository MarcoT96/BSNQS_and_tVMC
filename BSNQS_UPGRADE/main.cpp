#include <iostream>
#include <mpi.h>
#include <ctime>
#include "src/library.h"


int main(int argc, char* argv[]){

  //Initializes the MPI execution environment
  MPI_Init(&argc,&argv);

  int n_procs;  //Number of process units involved in the communicator
  int rank;  //Rank of the calling process in the communicator
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);  //Returns the size of the group associated to a communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //Determines the rank of the calling process in the communicator

  //Prepares the simulation
  auto options = setOptions(argc, argv, rank);  //Load parameters
  WaveFunction* Psi;  // 𝜳(𝒗,𝜙,𝜶)
  SpinHamiltonian* H;  // ℍ̂

  //Defines the wavefunction
  if(options["ansatz"] == "JasNN"){  //Nearest-neighbors Jastrow Quantum State

    if(options["filename"] != "None") Psi = new JasNN(options["filename"], std::stoi(options["if_phi_neq_zero"]), rank);
    else Psi = new JasNN(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]), rank);

  }
  else if(options["ansatz"] == "RBM"){  //Neural Network Quantum State

    if(options["filename"] != "None") Psi = new RBM(options["filename"], std::stoi(options["if_phi_neq_zero"]), rank);
    else Psi = new RBM(std::stoi(options["n_visible"]), std::stoi(options["density"]), std::stoi(options["if_phi_neq_zero"]), rank);

  }
  else if(options["ansatz"] == "uRBM"){  //Shadow (quasi)-uRBM Quantum State

    if(options["filename"] != "None") Psi = new quasi_uRBM(options["filename"], std::stoi(options["if_phi_neq_zero"]), rank);
    else Psi = new quasi_uRBM(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]), rank);

  }
  else if(options["ansatz"] == "BSWF"){  //Baeriswyl-Shadow Quantum State

    if(options["filename"] != "None") Psi = new Baeriswyl_Shadow(options["filename"], std::stoi(options["if_phi_neq_zero"]), rank);
    else Psi = new Baeriswyl_Shadow(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]), rank);

  }
  else if(options["ansatz"] == "SWF_NNN"){  //Next-Nearest-Neighbors Shadow Quantum State

    if(options["filename"] != "None") Psi = new SWF_NNN(options["filename"], std::stoi(options["if_phi_neq_zero"]), rank);
    else Psi = new SWF_NNN(std::stoi(options["n_visible"]), std::stoi(options["if_phi_neq_zero"]), rank);

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
                    rank,
                    std::stod(options["J"]));
  }
  else if(options["model"] == "Heisenberg1d"){  //Heisenberg model in 𝐝 = 𝟏 with 𝑃𝐵𝐶
    H = new Heisenberg1d(std::stoi(options["n_visible"]),
                         rank,
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
  VMC_Sampler sampler(*Psi, *H, rank);

  if(std::stoi(options["if_write_move_statistics"])) sampler.setFile_Move_Statistics(options["file_info"], rank);
  if(std::stoi(options["if_write_MCMC_config"])) sampler.setFile_MCMC_Config(options["file_info"], rank);
  if(std::stoi(options["if_write_final_config"])) sampler.setFile_final_Config(options["file_info"], MPI_COMM_WORLD);
  sampler.setFile_opt_Energy(options["file_info"], rank);
  sampler.setFile_block_Energy(options["file_info"], rank);
  if(std::stoi(options["if_write_opt_obs"])) sampler.setFile_opt_Obs(options["file_info"], rank);
  if(std::stoi(options["if_write_block_obs"])) sampler.setFile_block_Obs(options["file_info"], rank);
  if(std::stoi(options["if_write_opt_params"])) sampler.setFile_opt_Params(options["file_info"], rank);
  if(std::stoi(options["if_write_all_params"])) sampler.setFile_all_Params(options["file_info"], rank);
  if(std::stoi(options["if_write_qgt_matrix"])) sampler.setFile_QGT_matrix(options["file_info"], rank);
  if(std::stoi(options["if_write_qgt_cond"])) sampler.setFile_QGT_cond(options["file_info"], rank);
  if(std::stoi(options["if_write_qgt_eigen"])) sampler.setFile_QGT_eigen(options["file_info"], rank);

  //Sets the simulation parameters
  if(std::stoi(options["if_hidden_off"])) sampler.ShutDownHidden();
  if(std::stod(options["if_QGT_reg"]) != 0.0){

    sampler.choose_reg_method(std::stoi(options["QGT_reg_method"]));
    if(std::stod(options["if_QGT_reg"]) == 1.0) sampler.setQGTReg();
    else sampler.setQGTReg(std::stod(options["if_QGT_reg"]));

  }
  if(std::stoi(options["_if_extra_hidden_sum"])) sampler.setExtraHiddenSum(std::stoi(options["N_extra_sum"]), std::stoi(options["N_blks_hidden"]));

  //Chooses the appropriate algorithm
  if(rank == 0) std::cout << "\n#Start the Variational Monte Carlo algorithm." << std::endl;
  if(std::stoi(options["if_vmc"])){

    options["n_optimization_steps"] = "1";
    if(rank == 0) std::cout << " Quantum properties calculation via a simple Variational Monte Carlo (𝐕𝐌𝐂)." << std::endl;

  }
  else if(std::stoi(options["if_imaginary_time"])){

    sampler.setImagTimeDyn(std::stod(options["delta"]));
    if(rank == 0) std::cout << " Searching the Ground State via imaginary time-dependent Variational Monte Carlo (𝑖-𝐭𝐕𝐌𝐂)." << std::endl;

  }
  else if(std::stoi(options["if_real_time"])){

      if(options["model"] == "Ising1d") H -> Quench(std::stod(options["h_quench"]));
      else if(options["model"] == "Heisenberg1d") H -> Quench(std::stod(options["Jz_quench"]));
      sampler.setRealTimeDyn(std::stod(options["delta"]));
      if(rank == 0) std::cout << " Performing the quantum dynamics via time-dependent Variational Monte Carlo (𝐭𝐕𝐌𝐂)." << std::endl;

  }
  else{

    std::cerr << "##InputError: it is mandatory to make a choice regarding the type of variational optimization!" << std::endl;
    std::cerr << "   Failed to choose the appropriate algorithm." << std::endl;
    std::abort();

  }

  sampler.setStepParameters(std::stoi(options["n_sweeps"]), std::stoi(options["n_blks"]),
                            std::stoi(options["n_equilibration_steps"]), std::stoi(options["n_bunch"]),
                            std::stoi(options["bunch_dimension"]), std::stod(options["p_equal_site"]),
                            std::stod(options["p_visible_nn"]), std::stod(options["p_hidden_nn"]),
                            rank);

  /*##############################################*/
  /*  PERFORMS THE CHOSEN OPTIMIZATION ALGORTIHM  */
  /*##############################################*/
  std::clock_t t_start = std::clock();  //Start to measure CPU time

  //Initial thermalization phase
  sampler.Init_Config();  //Random initial configuration at time 0
  if(rank == 0){

    std::cout << " Thermalization ... ";
    std::flush(std::cout);

  }
  for(unsigned int init_eq_step = 0; init_eq_step < std::stoi(options["n_init_eq_steps"]); init_eq_step++){  //Loop over equilibration time

    //Moves configuration without measurements
    sampler.Make_Sweep();

  }
  if(rank == 0) std::cout << "done" << std::endl;

  //Time evolution of the variational quantum state
  for(unsigned int tdvmc_step = 0; tdvmc_step < std::stoi(options["n_optimization_steps"]); tdvmc_step++){  //Loop over time

    //Print options
    if(std::stoi(options["if_restart"]) == 1) sampler.setRestartFromConfig();
    if(std::stoi(options["if_real_time"]) == false && rank == 0){

      std::cout << " 𝝉 = " << std::setprecision(4) << std::fixed << std::stod(options["delta"]) * (tdvmc_step + 1) << " ... ";
      std::flush(std::cout);

    }
    else if(std::stoi(options["if_real_time"]) == true && rank == 0){

      std::cout << " 𝒕 = " << std::setprecision(4) << std::fixed << std::stod(options["delta"]) * (tdvmc_step + 1) << " ... ";
      std::flush(std::cout);

    }

    //𝐕𝐌𝐂 step
    sampler.VMC_Step(MPI_COMM_WORLD);

    //Updates variational parameters
    if(options["integrator"] == "Euler") sampler.Euler(MPI_COMM_WORLD);     //Euler ODE method
    else if(options["integrator"] == "Heun") sampler.Heun(MPI_COMM_WORLD);  //Heun ODE method
    else if(options["integrator"] == "RK4") sampler.RK4(MPI_COMM_WORLD);    //4th order Runge Kutta ODE method
    else{

      std::cerr << "##InputError: choosen ODE integrator not available!" << std::endl;
      std::cerr << "  Failed to choose the appropriate ODE integration method." << std::endl;
      std::abort();

    }

    //Prints results
    sampler.Write_Move_Statistics(tdvmc_step, MPI_COMM_WORLD);
    sampler.Write_final_Config(tdvmc_step);
    sampler.Write_Quantum_properties(tdvmc_step, rank);
    sampler.Write_all_Params(tdvmc_step, rank);

    //Prepares all stuff for the next VMC
    sampler.Reset_Moves_Statistics();
    sampler.Reset();

    if(rank == 0) std::cout << "done" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

  }
  if(std::stoi(options["if_vmc"]) == false && rank == 0) std::cout << " Quantum State evolved." << std::endl << std::endl;

  //Computational time
  if(rank == 0) std::cout << " Total sampling per time  --> " << std::scientific << std::setprecision(2) << std::stoi(options["n_sweeps"]) * double(n_procs) << std::endl;
  if(rank == 0) std::cout << " CPU times for the evolution of the Quantum State: \n";
  std::clock_t t_end = std::clock();  //Ends to measure CPU time
  long double time = (t_end - t_start)*1.0 / CLOCKS_PER_SEC;  //CPU time in seconds
  int hours = time/3600;  //CPU h
  int min = (time - (double)(3600*hours))/60;  //CPU min
  int sec = time - (double)(3600*hours) - (double)(60*min);  //CPU s
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "\tNode " << rank << " --> " << hours << "h " << min << "min " << sec << "s" << std::endl;

  //Remaining output files management
  sampler.Write_opt_Params(rank);
  sampler.CloseFile(rank);
  sampler.Finalize(rank);
  if(rank == 0) std::cout << std::endl;
  delete Psi;
  delete H;

  //Terminates MPI execution environment
  MPI_Finalize();

  return 0;


}
