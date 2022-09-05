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


#ifndef __READ_OPTIONS__
#define __READ_OPTIONS__


/*************************************************************************************************/
/************************************ 𝑹𝒆𝒂𝒅𝒊𝒏𝒈 𝑶𝒑𝒕𝒊𝒐𝒏𝒔 𝑰𝒏𝒕𝒆𝒓𝒇𝒂𝒄𝒆 *************************************/
/*************************************************************************************************/
/*

  Functionality for the input management, both from file and command line, for the
  Variational Quantum State sampler program.

*/
/*************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
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

    std::cout << "\n** To execute the main code: mpirun -np N ./main.exe OPTIONS **" << std::endl;
    std::cout << "  ================" << std::endl;
    std::cout << "  Allowed OPTIONS" << std::endl;
    std::cout << "  ================" << std::endl << std::endl;

    std::cout << "\t • N" << std::endl;
    std::cout << "\t\t * Number of processes involved in the parallel execution of the code with MPICH." << std::endl << std::endl;

    std::cout << "\t • --input=..." << std::endl;
    std::cout << "\t\t * Name of the file containing the input simulation hyper-parameters." <<std::endl;
    std::cout << "\t\t * Remember to enter the full path for the input file." << std::endl << std::endl;

    std::cout << "\t • --filename=..." << std::endl;
    std::cout << "\t\t * Name of the file containing initial conditions for the variational parameters defining the Variational Quantum State." <<std::endl;
    std::cout << "\t\t * Remember to enter the full path for the input file." << std::endl;
    std::cout << "\t\t * Default filename is 'None'." << std::endl << std::endl;

    std::cout << "\t • --model=..." << std::endl;
    std::cout << "\t\t * Name of the Quantum model to be simulated." <<std::endl;
    std::cout << "\t\t * Quantum models available at the moment:" << std::endl;
    std::cout << "\t\t\t ** Transverse Field Ising Model in 𝗱 = 𝟏 (enter Ising1d);" << std::endl;
    std::cout << "\t\t\t ** Heisenberg Model in 𝗱 = 𝟏 (enter Heisenberg1d);" << std::endl;
    std::cout << "\t\t * Default model is the quantum Ising model in 𝗱 = 𝟏." << std::endl << std::endl;

    std::cout << "\t • --ansatz=..." << std::endl;
    std::cout << "\t\t * Name of the type of variational 𝒜𝓃𝓈𝒶𝓉𝓏 to be used in the VMC algorithm." <<std::endl;
    std::cout << "\t\t * Variational wavefunction available at the moment:" << std::endl;
    std::cout << "\t\t\t ** Nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 (enter JWF);" << std::endl;
    std::cout << "\t\t\t ** Nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 with inhomogeneous entangling parameters (enter JWF_inhom);" << std::endl;
    std::cout << "\t\t\t ** Long-range homogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 (enter LRHJas);" << std::endl;
    std::cout << "\t\t\t ** Jastrow Neural Network Quantum state (enter JasNQS);" << std::endl;
    std::cout << "\t\t\t ** Restricted Boltzmann Machine Neural Network (enter RBM);" << std::endl;
    std::cout << "\t\t\t ** Baeriswyl-Shadow 𝒜𝓃𝓈𝒶𝓉𝓏 in 𝗱 = 𝟏 (enter BSWF);" << std::endl;
    std::cout << "\t\t\t ** Baeriswyl-Shadow 𝒜𝓃𝓈𝒶𝓉𝓏 with next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌 correlations in 𝗱 = 𝟏 (enter NNN_BSWF);" << std::endl;
    std::cout << "\t\t\t ** (quasi)-unRestricted Boltzmann Machine 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 in 𝗱 = 𝟏 (enter uRBM);" << std::endl;
    std::cout << "\t\t * Default 𝒜𝓃𝓈𝒶𝓉𝓏 is the nearest-neighbors Jastrow wave function." << std::endl << std::endl;

    std::cout << "\t • --integrator=..." << std::endl;
    std::cout << "\t\t * Name of the ODE integrator to be used to solving the variational parameters equations of motion." <<std::endl;
    std::cout << "\t\t * ODE integrators available at the moment:" << std::endl;
    std::cout << "\t\t\t ** Euler method (enter Euler);" << std::endl;
    std::cout << "\t\t\t ** Heun method (enter Heun);" << std::endl;
    std::cout << "\t\t\t ** Fourth order Runge Kutta method (enter RK4);" << std::endl;
    std::cout << "\t\t * Default integrator is the Euler method." << std::endl << std::endl;

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
    std::cout << "#Prepare all stuff for the simulation." << std::endl;

  }

  //Captures command-line inputs
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
      {"input", required_argument, 0, 'a'},
      {"filename", required_argument, 0, 'b'},
      {"model", required_argument, 0, 'c'},
      {"ansatz", required_argument, 0, 'd'},
      {"integrator", required_argument, 0, 'e'},
      {0, 0, 0, 0}

    };
    int option_index = 0;  //getopt_long stores the option index here
    int c = getopt_long(argc, argv, "a:b:c:d:e:", long_options, &option_index);
    if (c == -1)  //Detect the end of the options
      break;
    switch(c){

      case 'a':
        options["input"] = optarg;
        if(rank == 0) std::cout << " Read simulation parameters from the file " << std::string(optarg) << "." << std::endl;
        break;

      case 'b':
        options["filename"] = optarg;
        if(std::string(optarg) != "None")
          if(rank == 0) std::cout << " Initial condition for the wave function variational parameters uploaded from file " << std::string(optarg) << "." << std::endl;
        break;

      case 'c':
        options["model"] = optarg;
        //if(rank == 0) std::cout << " Quantum model to be simulated: " << std::string(optarg) << "." << std::endl;
        break;

      case 'd':
        options["ansatz"] = optarg;
        //if(rank == 0) std::cout << " Type of variational wave function: " << std::string(optarg) << "." << std::endl;
        break;

      case 'e':
        options["integrator"] = optarg;
        //if(rank == 0) std::cout << " ODE integrator for the 𝐭𝐕𝐌𝐂 equations of motion: " << std::string(optarg) << "." << std::endl;
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
  if(options.count("input") == 0){

    if(rank == 0)
      std::cerr << " ##InputError: Option for the input file must be specified with the option --input=INPUT_FILE_NAME" << std::endl << std::endl;
    std::exit(0);

  }
  if(options.count("filename") == 0) options["filename"] = "None";
  if(options.count("model") == 0) options["model"] = "Ising1d";
  if(options.count("ansatz") == 0) options["ansatz"] = "JWF";
  if(options.count("integrator") == 0) options["integrator"] = "Euler";

  //Manages the remaining options from input file
  std::ifstream in(options["input"]);
  if(!in.good()){

    if(rank == 0){

      std::cerr << " ##Error: opening file " << options["input"] << " failed." << std::endl;
      std::cerr << "   File not found." << std::endl;

    }
    std::abort();

  }
  char* string_away = new char[100];
  if(in.is_open()){

    in >> string_away >> options["n_REAL"];
    in >> string_away >> options["SHADOWS_DENSITY"];
    in >> string_away >> options["if_GLOBAL_PHASE"];
    in >> string_away >> options["if_ZERO_IMAGINARY_PART"];
    if(options["model"] == "Ising1d"){

      in >> string_away >> options["J"];
      in >> string_away >> options["h_field"];

    }
    else if(options["model"] == "Heisenberg1d"){

      in >> string_away >> options["h_field"];
      in >> string_away >> options["Jx"];
      in >> string_away >> options["Jy"];
      in >> string_away >> options["Jz"];

    }
    in >> string_away >> options["if_VMC"];
    in >> string_away >> options["if_IMAGINARY-TIME_DYNAMICS"];
    in >> string_away >> options["if_REAL-TIME_DYNAMICS"];
    in >> string_away >> options["QGT_REGULARIZATION_METHOD"];
    in >> string_away >> options["QGT_REGULARIZATION_CONTROL_PARAMETER_VALUE_1"];
    in >> string_away >> options["QGT_REGULARIZATION_CONTROL_PARAMETER_VALUE_2"];
    in >> string_away >> options["INTEGRATION_TIME-STEP"];
    in >> string_away >> options["NUMBER_OF_TIME-STEPS"];
    in >> string_away >> options["n_blks"];
    in >> string_away >> options["n_sweeps"];
    in >> string_away >> options["n_init_eq_steps"];
    in >> string_away >> options["n_equilibration_steps"];
    in >> string_away >> options["n_bunches"];
    in >> string_away >> options["bunch_dimension"];
    in >> string_away >> options["p_equal_site"];
    in >> string_away >> options["p_real_nn"];
    in >> string_away >> options["p_shadow_nn"];
    in >> string_away >> options["p_global_ket_flip"];
    in >> string_away >> options["p_global_bra_flip"];
    in >> string_away >> options["if_restart_from_config"];
    in >> string_away >> options["if_shadows_off"];
    in >> string_away >> options["if_extra_shadow_sum"];
    in >> string_away >> options["n_extra_sum"];
    in >> string_away >> options["n_shadow_blks"];
    in >> string_away >> options["output_file_info"];
    in >> string_away >> options["if_write_MOVE_STATISTICS"];
    in >> string_away >> options["if_write_MCMC_CONFIG"];
    in >> string_away >> options["if_write_FINAL_CONFIG"];
    in >> string_away >> options["if_write_ENERGY_ALL"];
    in >> string_away >> options["if_measure_ENERGY"];
    in >> string_away >> options["if_measure_NON-DIAGONAL_OBS"];
    in >> string_away >> options["if_measure_DIAGONAL_OBS"];
    in >> string_away >> options["if_measure_BLOCK_ENERGY"];
    in >> string_away >> options["if_measure_BLOCK_NON-DIAGONAL_OBS"];
    in >> string_away >> options["if_measure_BLOCK_DIAGONAL_OBS"];
    in >> string_away >> options["if_write_OPT_VQS"];
    in >> string_away >> options["if_write_VQS_EVOLUTION"];
    in >> string_away >> options["if_write_QGT"];
    in >> string_away >> options["if_write_QGT_CONDITION_NUMBER"];
    in >> string_away >> options["if_write_QGT_EIGENVALUES"];

  }
  else{

    std::cerr << " ##PROBLEM: Unable to open the parameters file." << std::endl;
    std::cerr << "   File " << options["input"] << " is not open." << std::endl;
    std::abort();

  }

  //End of the reading
  if(rank == 0) std::cout << " Parameters loaded." << std::endl << std::endl;
  return options;

}


#endif


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
