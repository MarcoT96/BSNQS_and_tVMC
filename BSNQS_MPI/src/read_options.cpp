#ifndef __READ_OPTIONS__
#define __READ_OPTIONS__


/*************************************************************************************************/
/*********************************** Reading Options Interface ***********************************/
/*************************************************************************************************/
/*

  Functionality for the input management, both from file and command line, for the
  Variational Quantum State sampler program.

*/
/*************************************************************************************************/


#include <iostream>  // <-- std::cout, std::endl, etc…
#include <cstdlib>  // <-- std::abort(), std::exit()
#include <fstream>  // <-- std::ifstream
#include <getopt.h>  // <-- long_options, required_argument


void PrintHeader(int rank){

  if(rank == 0){

    std::cout << std::endl;
    std::cout << "#################### All rights reserved ########################" << std::endl;
    std::cout << "#################################################################" << std::endl;
    std::cout << "    _/      _/_/_/  _/_/_/  Laboratorio di Calcolo Parallelo e" << std::endl;
    std::cout << "   _/      _/      _/  _/  di Simulazioni di Materia Condensata" << std::endl;
    std::cout << "  _/      _/      _/_/_/  c/o Sezione Struttura della Materia" << std::endl;
    std::cout << " _/      _/      _/      Dipartimento di Fisica" << std::endl;
    std::cout << "_/_/_/  _/_/_/  _/      Universita' degli Studi di Milano" << std::endl;
    std::cout << "                       Professor Davide E. Galli" << std::endl;
    std::cout << "                      Doctor Christian Apostoli" << std::endl;
    std::cout << "                     Code written by Marco Tesoro" << std::endl;
    std::cout << "#################################################################" << std::endl;
    std::cout << "#################################################################" << std::endl;

  }

}


void PrintInfo(int rank){  //Print informations for command line instructions

  if(rank == 0){

    std::cout << "\nTo execute the main code: mpiexec -np N ./main.exe OPTIONS" << std::endl;
    std::cout << "  ================" << std::endl;
    std::cout << "  Allowed OPTIONS" << std::endl;
    std::cout << "  ================" << std::endl << std::endl;

    std::cout << "\t • N" << std::endl;
    std::cout << "\t\t * Number of processes involved in the parallel execution of the code with openMPI" << std::endl;

    std::cout << "\t • --filename=..." << std::endl;
    std::cout << "\t\t * Name of the file containing initial variational parameters for the Variational Quantum State" <<std::endl;
    std::cout << "\t\t * Choose the file in the appropriate quantum model directory and sub-directory GroundState/ or Dynamics/" << std::endl;
    std::cout << "\t\t * Default filename is 'None'" << std::endl;

    std::cout << "\t • --model=..." << std::endl;
    std::cout << "\t\t * Name of the Quantum model to be simulated" <<std::endl;
    std::cout << "\t\t * Quantum models available at the moment:" << std::endl;
    std::cout << "\t\t\t ** Transverse Field Ising Model in 1d (enter Ising1d)" << std::endl;
    std::cout << "\t\t\t ** Heisenberg Model in 1d (enter Heisenberg1d)" << std::endl;
    std::cout << "\t\t * Default model is the 1d Ising model" << std::endl << std::endl;

    std::cout << "\t • --ansatz=..." << std::endl;
    std::cout << "\t\t * Name of the type of variational ansatz to be used in the VMC" <<std::endl;
    std::cout << "\t\t * Variational wavefunction available at the moment:" << std::endl;
    std::cout << "\t\t\t ** Nearest-neighbors Jastrow (enter JasNN)" << std::endl;
    std::cout << "\t\t\t ** Restricted Boltzmann Machine Neural Network (enter RBM)" << std::endl;
    std::cout << "\t\t\t ** (quasi)-unRestricted Boltzmann Machine Shadow Ansatz in 𝗱 = 𝟏 (enter uRBM)" << std::endl;
    std::cout << "\t\t\t ** Baeriswyl-Shadow Ansatz in 𝗱 = 𝟏 (enter BS_NNQS)" << std::endl;
    std::cout << "\t\t * Default ansatz is the nearest-neighbors Jastrow wave function" << std::endl << std::endl;

    std::cout << "\t • --input=..." << std::endl;
    std::cout << "\t\t * Name of the file containing the input simulation hyper-parameters" <<std::endl;
    std::cout << "\t\t * Choose the file in the appropriate quantum model directory" << std::endl << std::endl;

    std::cout << "\t • --integrator=..." << std::endl;
    std::cout << "\t\t * Name of the ODE integrator to be used to solving the variational parameters equations of motion" <<std::endl;
    std::cout << "\t\t * ODE integrators available at the moment:" << std::endl;
    std::cout << "\t\t\t ** Euler method (enter Euler)" << std::endl;
    std::cout << "\t\t\t ** Heun method (enter Heun)" << std::endl;
    std::cout << "\t\t\t ** Fourth order Runge Kutta method (enter RK4)" << std::endl;
    std::cout << "\t\t * Default integrator is the Euler method" << std::endl << std::endl;

  }

}


std::map <std::string, std::string> setOptions(int argc, char* argv[], int rank) {  //Prepare all the stuff for the simulation

  //Information
  PrintHeader(rank);
  if(argc == 1) {

    PrintInfo(rank);
    std::exit(0);

  }
  if(rank == 0){

    std::cout << std::endl;
    std::cout << "#Prepare all stuff for the simulation" << std::endl;

  }

  std::map <std::string, std::string> options;
  while(true){

    //Manages command line options
    static struct option long_options[] = {

      /*#############################################################################*/
      //  These options don’t set a flag.
      //  We distinguish them by their indices.
      //  We have
      //
      //    {Name of the command line option, required argument, flag, option index}.
      /*#############################################################################*/
      {"filename", required_argument, 0, 'a'},
      {"model", required_argument, 0, 'b'},
      {"ansatz", required_argument, 0, 'c'},
      {"input", required_argument, 0, 'd'},
      {"integrator", required_argument, 0, 'e'},
      {0, 0, 0, 0}

    };
    int option_index = 0;  //getopt_long stores the option index here
    int c = getopt_long(argc, argv, "a:b:c:d:e:", long_options, &option_index);
    if (c == -1)  //Detect the end of the options
      break;
    switch(c){

      case 'a':
        options["filename"] = optarg;
        if(std::string(optarg) != "None")
          if(rank == 0) std::cout << " Initial wave function variational parameters uploaded from file " << std::string(optarg) << std::endl;
        break;

      case 'b':
        options["model"] = optarg;
        if(rank == 0) std::cout << " Quantum model to be simulated: " << std::string(optarg) << std::endl;
        break;

      case 'c':
        options["ansatz"] = optarg;
        if(rank == 0) std::cout << " Type of variational wave function ansatz: " << std::string(optarg) << std::endl;
        break;

      case 'd':
        options["input"] = optarg;
        if(rank == 0) std::cout << " Read simulation parameters from the file " << std::string(optarg) << std::endl;
        break;

      case 'e':
        options["integrator"] = optarg;
        if(rank == 0) std::cout << " ODE integrator for the TDVMC equations of motion: " << std::string(optarg) << std::endl;
        break;

      case '?':  //Unrecognize option
        PrintInfo(rank);
        if(rank == 0) std::cout << std::endl;
        break;

      default:  //Default case
        std::exit(0);

    }

  }

  //Sets default values for command line options
  if(options.count("filename")==0) options["filename"] = "None";
  if(options.count("model")==0) options["model"] = "Ising1d";
  if(options.count("ansatz")==0) options["ansatz"] = "JasNN";
  if(options.count("input")==0){

    if(rank == 0)
      std::cerr << " ##InputError: Option for the input file must be specified with the option --input=INPUTFILE" << std::endl << std::endl;
    std::exit(0);

  }
  if(options.count("integrator")==0) options["integrator"] = "Euler";

  //Manages the remaining options from input file
  std::ifstream in(options["input"]);
  if(!in.good()){

    if(rank == 0){

      std::cerr << " ##Error: opening file " << options["input"] << " failed." << std::endl;
      std::cerr << "   File not found." << std::endl;

    }
    std::abort();

  }
  char* string_away = new char[60];
  if(in.is_open()){

    in >> string_away >> options["n_visible"];
    in >> string_away >> options["density"];
    in >> string_away >> options["if_phi_neq_zero"];
    if(options["model"]=="Ising1d"){
      in >> string_away >> options["J"];
      in >> string_away >> options["h_field"];
      in >> string_away >> options["h_quench"];
    }
    else if(options["model"]=="Heisenberg1d"){
      in >> string_away >> options["h_field"];
      in >> string_away >> options["Jz_quench"];
      in >> string_away >> options["Jx"];
      in >> string_away >> options["Jy"];
      in >> string_away >> options["Jz"];
    }
    in >> string_away >> options["if_vmc"];
    in >> string_away >> options["if_imaginary_time"];
    in >> string_away >> options["if_real_time"];
    in >> string_away >> options["if_QGT_reg"];
    in >> string_away >> options["delta"];
    in >> string_away >> options["n_optimization_steps"];
    in >> string_away >> options["n_blks"];
    in >> string_away >> options["n_sweeps"];
    in >> string_away >> options["n_init_eq_steps"];
    in >> string_away >> options["n_equilibration_steps"];
    in >> string_away >> options["n_bunch"];
    in >> string_away >> options["bunch_dimension"];
    in >> string_away >> options["p_equal_site"];
    in >> string_away >> options["p_visible_nn"];
    in >> string_away >> options["p_hidden_nn"];
    in >> string_away >> options["if_restart"];
    in >> string_away >> options["if_hidden_off"];
    in >> string_away >> options["orientation_bias"];
    in >> string_away >> options["_if_extra_hidden_sum"];
    in >> string_away >> options["N_extra_sum"];
    in >> string_away >> options["N_blks_hidden"];
    in >> string_away >> options["file_info"];
    in >> string_away >> options["if_write_move_statistics"];
    in >> string_away >> options["if_write_MCMC_config"];
    in >> string_away >> options["if_write_final_config"];
    in >> string_away >> options["if_write_opt_obs"];
    in >> string_away >> options["if_write_block_obs"];
    in >> string_away >> options["if_write_opt_params"];
    in >> string_away >> options["if_write_all_params"];
    in >> string_away >> options["if_write_qgt_matrix"];
    in >> string_away >> options["if_write_qgt_cond"];
    in >> string_away >> options["if_write_qgt_eigen"];

  }
  else{

    std::cerr << " ##PROBLEM: Unable to open the parameters file." << std::endl;
    std::cerr << "   File " << options["input"] << " is not open." << std::endl;
    std::abort();

  }

  //End of the reading
  if(rank == 0) std::cout << " Parameters loaded" << std::endl << std::endl;
  return options;

}


#endif
