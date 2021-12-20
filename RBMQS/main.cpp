#include <iostream>
#include "src/library.h"


int main(){


  /* Prepare the simulation */
  auto options = setOptions();  //Load parameters
  //RBM Psi(options["wf_file"]);  //Neural Network Quantum State
  RBM Psi(std::stoi(options["visible"]), std::stoi(options["hidden"]));  //Neural Network Quantum State
  std::string model = options["model"].c_str();  //Problem Hamiltonian

  /* Stochastic Reinforcement Learning */
  if(model == "Ising1d"){

    Ising1d H(Psi.N(), std::stod(options["hfield"]));
    MC_Sampler <RBM, Ising1d> sampler(Psi, H);  //Monte Carlo sampler

    std::cout << "#Set the output file options for the quantity of interest" << std::endl;
    if(options["filestate"] != "none")
      sampler.setFileState(options["filestate"].c_str());
    if(options["fileenergy"] != "none")
      sampler.setFileEnergy(options["fileenergy"].c_str());
    if(options["filesigma"] != "none")
      sampler.setFileSigma(options["filesigma"].c_str());
    if(options["filewave"] != "none")
      sampler.setFileWave(options["filewave"].c_str());
    std::cout << std::endl;

    //Machine Learning of the Quantum State
    std::cout << "#Start the Stochastic Reinforcement Learning of the Quantum State" << std::endl;
    if(std::stoi(options["realTime"])){

      H.quench(std::stod(options["hquench"]));
      sampler.setRealTimeDyn(std::stod(options["deltat"]));
      std::cout << " Performing Time-Dependent Variational Monte Carlo" << std::endl;

    }
    else
      std::cout << " Performing Ground State Variational Monte Carlo" << std::endl;
    for(unsigned int run=0; run<std::stoi(options["Nrun"]); run++){

      std::cout << " Run " << run+1 << " ... ";
      std::flush(std::cout);
      if(run != 0)
        sampler.Reset(options["filestate"].c_str(), options["filewave"].c_str());
      sampler.MC_Run(std::stoi(options["Nsweeps"]), std::stoi(options["Nblks"]), std::stod(options["eq_time"]), std::stod(options["Mmoves"]));
      sampler.CloseFile();
      std::cout << "done" << std::endl;

    }
    std::cout << " Quantum State optimized" << std::endl;

  }
  else{

    std::cerr << "##Error: no model found." << std::endl;
    std::abort();

  }

  std::cout << std::endl;
  return 0;

}
