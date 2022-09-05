/****************************************************************
******************** All rights reserved ************************
*****************************************************************
    _/      _/_/_/  _/_/_/  Laboratorio di Calcolo Parallelo e
   _/      _/      _/  _/  di Simulazioni di Materia Condensata
  _/      _/      _/_/_/  c/o Sezione Struttura della Materia
 _/      _/      _/      Dipartimento di Fisica
_/_/_/  _/_/_/  _/      Universita' degli Studi di Milano
                       Professor Davide E. Galli
                      Doctor Christian Apostoli
                     Code written by Marco Tesoro
*****************************************************************
*****************************************************************/


#include <iostream>
#include <mpi.h>
#include <ctime>
#include "src/library.h"
#include <chrono>
#include <thread>


int main(int argc, char* argv[]){

  //Initializes the MPI execution environment
  MPI_Init(&argc,&argv);

  int n_PROCESS;  //Number of process units involved in the communicator
  int rank;  //Rank of the calling process in the communicator
  MPI_Comm_size(MPI_COMM_WORLD, &n_PROCESS);  //Returns the size of the group associated to a communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //Determines the rank of the calling process in the communicator

  //Prepares the simulation
  auto options = setOptions(argc, argv, rank);  //Load parameters
  WaveFunction* Psi;  // 𝜳(𝒗,𝜙,𝜶)
  SpinHamiltonian* H;  // Ĥ
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  //Defines the variational wave function
  if(options["ansatz"] == "JWF"){  //Nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏

    if(options["filename"] != "None") Psi = new JWF(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new JWF(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "JWF_inhom"){  //Inhomogeneous nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏

    if(options["filename"] != "None") Psi = new JWF_inhom(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new JWF_inhom(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "LRHJas"){  //Long-range homogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏

    if(options["filename"] != "None") Psi = new LRHJas(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new LRHJas(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "JasNQS"){  //Jastrow neural network quantum state

    if(options["filename"] != "None") Psi = new JasNQS(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new JasNQS(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "RBM"){  //Restricted Boltzmann Machine Neural Network Quantum State

    if(options["filename"] != "None") Psi = new RBM(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new RBM(std::stoi(options["n_REAL"]), std::stoi(options["SHADOWS_DENSITY"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "BSWF"){  //Baeriswyl-Shadow 𝒜𝓃𝓈𝒶𝓉𝓏

    if(options["filename"] != "None") Psi = new BSWF(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new BSWF(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "NNN_BSWF"){  //Next-Nearest-Neighbors Baeriswyl-Shadow 𝒜𝓃𝓈𝒶𝓉𝓏

    if(options["filename"] != "None") Psi = new NNN_BSWF(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new NNN_BSWF(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else if(options["ansatz"] == "uRBM"){  //Shadow (quasi)-uRBM 𝒜𝓃𝓈𝒶𝓉𝓏

    if(options["filename"] != "None") Psi = new quasi_uRBM(options["filename"], std::stoi(options["if_GLOBAL_PHASE"]), rank);
    else Psi = new quasi_uRBM(std::stoi(options["n_REAL"]), std::stoi(options["if_GLOBAL_PHASE"]), std::stoi(options["if_ZERO_IMAGINARY_PART"]), rank);

  }
  else{

    std::cerr << "##InputError: type of variational wave function not allowed." << std::endl;
    std::cerr << "  Failed to construct the wave function 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  //Defines the Quantum Hamiltonian
  if(options["model"] == "Ising1d"){  //TFI model in 𝐝 = 𝟏 with 𝑃𝐵𝐶𝑠
    H = new Ising1d(std::stoi(options["n_REAL"]),
                    std::stod(options["h_field"]),
                    rank,
                    std::stod(options["J"]));
  }
  else if(options["model"] == "Heisenberg1d"){  //Heisenberg model in 𝐝 = 𝟏 with 𝑃𝐵𝐶𝑠
    H = new Heisenberg1d(std::stoi(options["n_REAL"]),
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
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  //Defines the VMC sampler and the output files
  VMC_Sampler sampler(*Psi, *H, rank);
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  if(std::stoi(options["if_write_MOVE_STATISTICS"])) sampler.setFile_Move_Statistics(options["output_file_info"], rank);
  if(std::stoi(options["if_write_MCMC_CONFIG"])) sampler.setFile_MCMC_Config(options["output_file_info"], rank);
  if(std::stoi(options["if_write_FINAL_CONFIG"])) sampler.setFile_final_Config(options["output_file_info"], MPI_COMM_WORLD);
  if(std::stoi(options["if_measure_ENERGY"])) sampler.setFile_Energy(options["output_file_info"], std::stoi(options["if_write_ENERGY_ALL"]), rank);
  if(std::stoi(options["if_measure_BLOCK_ENERGY"])) sampler.setFile_block_Energy(options["output_file_info"], rank);
  if(std::stoi(options["if_measure_NON-DIAGONAL_OBS"])) sampler.setFile_non_Diagonal_Obs(options["output_file_info"], rank);
  if(std::stoi(options["if_measure_BLOCK_NON-DIAGONAL_OBS"])) sampler.setFile_block_non_Diagonal_Obs(options["output_file_info"], rank);
  if(std::stoi(options["if_measure_DIAGONAL_OBS"])) sampler.setFile_Diagonal_Obs(options["output_file_info"], rank);
  if(std::stoi(options["if_measure_BLOCK_DIAGONAL_OBS"])) sampler.setFile_block_Diagonal_Obs(options["output_file_info"], rank);
  if(std::stoi(options["if_write_OPT_VQS"])) sampler.setFile_opt_VQS(options["output_file_info"], rank);
  if(std::stoi(options["if_write_VQS_EVOLUTION"])) sampler.setFile_VQS_evolution(options["output_file_info"], rank);
  if(std::stoi(options["if_write_QGT"])) sampler.setFile_QGT(options["output_file_info"], rank);
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  //Sets the simulation parameters
  if(std::stoi(options["if_shadows_off"])) sampler.ShutDownShadows();
  sampler.choose_regularization_method(std::stoi(options["QGT_REGULARIZATION_METHOD"]),
                                       std::stod(options["QGT_REGULARIZATION_CONTROL_PARAMETER_VALUE_1"]),
                                       std::stod(options["QGT_REGULARIZATION_CONTROL_PARAMETER_VALUE_2"]));
  if(std::stoi(options["if_write_QGT_CONDITION_NUMBER"])) sampler.setFile_QGT_condition_number(options["output_file_info"], rank);
  if(std::stoi(options["if_write_QGT_EIGENVALUES"])) sampler.setFile_QGT_eigenvalues(options["output_file_info"], rank);
  if(std::stoi(options["if_extra_shadow_sum"])) sampler.setExtraShadowSum(std::stoi(options["n_extra_sum"]), std::stoi(options["n_shadow_blks"]));

  //Chooses the appropriate algorithm
  if(rank == 0) std::cout << "\n#Start the Variational Monte Carlo algorithm." << std::endl;
  if(std::stoi(options["if_VMC"])){

    options["NUMBER_OF_TIME-STEPS"] = "1";
    if(rank == 0) std::cout << " Quantum properties calculation via a simple Variational Monte Carlo (𝐕𝐌𝐂)." << std::endl;

  }
  else if(std::stoi(options["if_IMAGINARY-TIME_DYNAMICS"])){

    sampler.setImagTimeDyn(std::stod(options["INTEGRATION_TIME-STEP"]));
    if(rank == 0){

      std::cout << " Search the ground state via imaginary time-dependent Variational Monte Carlo (𝑖-𝐭𝐕𝐌𝐂)." << std::endl;
      std::cout << " Equations of motion for the variational parameters numerically integrated via the " << options["integrator"] << " method." << std::endl;

    }

  }
  else if(std::stoi(options["if_REAL-TIME_DYNAMICS"])){

    sampler.setRealTimeDyn(std::stod(options["INTEGRATION_TIME-STEP"]));
    if(rank == 0){

      std::cout << " Performing the quantum dynamics via time-dependent Variational Monte Carlo (𝐭𝐕𝐌𝐂)." << std::endl;
      if(options["model"] == "Ising1d")
        std::cout << " Quantum quench at 𝒕 = 𝟢 towards 𝙝 = " << std::stod(options["h_field"]) << "." << std::endl;
      else if(options["model"] == "Heisenberg1d"){}
        std::cout << " Equations of motion for the variational parameters numerically integrated via the " << options["integrator"] << " method." << std::endl;

    }

  }
  else{

    std::cerr << "##InputError: it is mandatory to make a choice regarding the type of the variational algorithm!" << std::endl;
    std::cerr << "   Failed to choose the appropriate algorithm." << std::endl;
    std::abort();

  }

  sampler.setStepParameters(std::stoi(options["n_sweeps"]), std::stoi(options["n_blks"]),
                            std::stoi(options["n_equilibration_steps"]), std::stoi(options["n_bunches"]),
                            std::stoi(options["bunch_dimension"]), std::stod(options["p_equal_site"]),
                            std::stod(options["p_real_nn"]), std::stod(options["p_shadow_nn"]),
                            std::stod(options["p_global_ket_flip"]), std::stod(options["p_global_bra_flip"]),
                            rank);
  //std::this_thread::sleep_for(std::chrono::milliseconds(2000));

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
  for(int init_eq_step = 0; init_eq_step < std::stoi(options["n_init_eq_steps"]); init_eq_step++){  //Loop over equilibration time

    //Moves configuration without measurements
    sampler.Make_Sweep();

  }
  if(rank == 0) std::cout << "done" << std::endl;

  //Time evolution of the variational quantum state
  for(int tvmc_step = 0; tvmc_step < std::stoi(options["NUMBER_OF_TIME-STEPS"]); tvmc_step++){  //Loop over time

    //Print options
    if(std::stoi(options["if_restart_from_config"]) == 1) sampler.setRestartFromConfig();
    if(std::stoi(options["if_REAL-TIME_DYNAMICS"]) == false && rank == 0){

      std::cout << " 𝝉 = " << std::setprecision(4) << std::fixed << std::stod(options["INTEGRATION_TIME-STEP"]) * (tvmc_step + 1) << " ... ";
      std::flush(std::cout);

    }
    else if(std::stoi(options["if_REAL-TIME_DYNAMICS"]) == true && rank == 0){

      std::cout << " 𝒕 = " << std::setprecision(6) << std::fixed << std::stod(options["INTEGRATION_TIME-STEP"]) * (tvmc_step + 1) << " ... ";
      std::flush(std::cout);

    }

    //𝐭𝐕𝐌𝐂 step
    sampler.tVMC_Step(MPI_COMM_WORLD, tvmc_step);

    //Updates variational parameters
    if(options["integrator"] == "Euler") sampler.Euler(MPI_COMM_WORLD, tvmc_step);  //Euler ODE method
    else if(options["integrator"] == "Heun") sampler.Heun(MPI_COMM_WORLD, tvmc_step);  //Heun ODE method
    else if(options["integrator"] == "RK4") sampler.RK4(MPI_COMM_WORLD, tvmc_step);  //4th order Runge Kutta ODE method
    else{

      std::cerr << "##InputError: choosen ODE integrator not available!" << std::endl;
      std::cerr << "  Failed to choose the appropriate ODE integration method." << std::endl;
      std::abort();

    }

    //Prints results
    sampler.write_Move_Statistics(tvmc_step, MPI_COMM_WORLD);
    sampler.write_final_Config(tvmc_step, rank);
    sampler.write_Quantum_properties(tvmc_step, rank);
    sampler.write_VQS_evolution(tvmc_step, rank);

    //Prepares all stuff for the next 𝐭𝐕𝐌𝐂 step
    sampler.Reset_Moves_Statistics();
    sampler.Reset();

    if(rank == 0) std::cout << "done" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

  }
  if(std::stoi(options["if_VMC"]) == false && rank == 0) std::cout << " Quantum state evolved." << std::endl << std::endl;

  //Computational time
  if(rank == 0) std::cout << " Total sampling per time  --> " << std::scientific << std::setprecision(2) << std::stoi(options["n_sweeps"]) * double(n_PROCESS) << std::endl;
  if(rank == 0) std::cout << " CPU times for the evolution of the quantum state: \n";
  std::clock_t t_end = std::clock();  //Ends to measure CPU time
  long double time = (t_end - t_start)*1.0 / CLOCKS_PER_SEC;  //CPU time in seconds
  int hours = time/3600;  //CPU h
  int min = (time - (double)(3600*hours))/60;  //CPU min
  int sec = time - (double)(3600*hours) - (double)(60*min);  //CPU s
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "\tNode " << rank << " --> " << hours << "h " << min << "min " << sec << "s" << std::endl;

  //Remaining output files management
  sampler.write_opt_VQS(rank);
  sampler.CloseFile(rank);
  sampler.Finalize(rank);
  if(rank == 0) std::cout << std::endl;
  delete Psi;
  delete H;

  //Terminates MPI execution environment
  MPI_Finalize();

  return 0;


}


/****************************************************************
******************** All rights reserved ************************
*****************************************************************
    _/      _/_/_/  _/_/_/  Laboratorio di Calcolo Parallelo e
   _/      _/      _/  _/  di Simulazioni di Materia Condensata
  _/      _/      _/_/_/  c/o Sezione Struttura della Materia
 _/      _/      _/      Dipartimento di Fisica
_/_/_/  _/_/_/  _/      Universita' degli Studi di Milano
                       Professor Davide E. Galli
                      Doctor Christian Apostoli
                     Code written by Marco Tesoro
*****************************************************************
*****************************************************************/
