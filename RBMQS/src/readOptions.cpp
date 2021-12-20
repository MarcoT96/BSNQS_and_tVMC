#ifndef __READOPTIONS__
#define __READOPTIONS__


#include <iostream>
#include <cstdlib>
#include <fstream>


/*######################################################################*/
/*# Various utilities to read the input file and command line options  #*/
/*######################################################################*/


std::map <std::string, std::string> ReadInput(std::string input_file) {

  //Information
  std::cout << " Read simulation parameters from the file " << input_file << std::endl;

  std::ifstream in(input_file.c_str());
  if(!in.good()){

    std::cerr << " ##Error: opening file " << input_file << " failed." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::abort();

  }

  char* string_away = new char[60];
  std::map <std::string, std::string> options {

    {"wf_file", ""},
    {"visible", ""},
    {"hidden", ""},
    {"model", ""},
    {"hfield", ""},
    {"hquench", ""},
    {"realTime", ""},
    {"Gamma", ""},
    {"deltat", ""},
    {"Nrun", ""},
    {"Nblks", ""},
    {"Nsweeps", ""},
    {"eq_time", ""},
    {"Mmoves", ""},
    {"Nflipped", ""},
    {"filestate", ""},
    {"fileenergy", ""},
    {"filesigma", ""},
    {"filewave", ""}

  };

  if(in.is_open()){

    in >> string_away >> options["wf_file"];
    in >> string_away >> options["visible"];
    in >> string_away >> options["hidden"];
    in >> string_away >> options["model"];
    in >> string_away >> options["hfield"];
    in >> string_away >> options["hquench"];
    in >> string_away >> options["realTime"];
    in >> string_away >> options["Gamma"];
    in >> string_away >> options["deltat"];
    in >> string_away >> options["Nrun"];
    in >> string_away >> options["Nblks"];
    in >> string_away >> options["Nsweeps"];
    in >> string_away >> options["eq_time"];
    in >> string_away >> options["Mmoves"];
    in >> string_away >> options["Nflipped"];
    in >> string_away >> options["filestate"];
    in >> string_away >> options["fileenergy"];
    in >> string_away >> options["filesigma"];
    in >> string_away >> options["filewave"];

  }
  else{

    std::cerr << " ##PROBLEM: Unable to open the parameters file." << std::endl;
    std::cerr << "   File " << input_file << " is not open." << std::endl;
    std::abort();

  }
  return options;

}


void PrintHeader(){}


void PrintInfo(){}


std::string FindModel(std::string arg) {

  /*
    Finds the required substring " " in the argument arg.
    If it does not find it, returns the key value
    std::string::npos.
  */

  //Information
  std::cout << " Searching for the model Hamiltonian" << std::endl;

  auto substr = arg.find("Ising");
  if(substr != std::string::npos)
    return "Ising1d";

  return "None";

}


std::string FindCoupling(std::string arg) {

  //Information
  std::cout << " Searching for the couplings in the Hamiltonian" << std::endl;

  auto substr = arg.find("_");
  auto substr1= std::string::npos;
  auto substr2= std::string::npos;
  if (substr != std::string::npos){

    substr1 = arg.find("_", substr+1);
    if(substr1 != std::string::npos)
      substr2 = arg.find("_", substr1+1);

  }

  if(substr1 != std::string::npos && substr2 != std::string::npos)
    return(arg.substr(substr1+1, substr2-substr1-1));
  else{

     std::cerr << " ##Error: searching for the couplings failed." << std::endl;
     std::cerr << "   The filename is not in the format specified for the avaible model." << std::endl;
     std::abort();

  }
  std::exit(0);
  return "   Error";

return "None";

}


std::map <std::string, std::string> setOptions(std::string input_file = "./input/input.dat") {

  //Information
  PrintHeader();
  PrintInfo();
  std::cout << std::endl;
  std::cout << "#Prepare all stuff for the simulation" << std::endl;


  std::map <std::string, std::string> options = ReadInput(input_file);
  if(options["wf_file"] != "none"){

    options["model"] = FindModel(options["wf_file"]);
    options["hfield"] = FindCoupling(options["wf_file"]);

  }
  std::cout << " Parameters loaded" << std::endl << std::endl;
  return options;

}


/*##########################################*/
/*# Various utilities to process the data  #*/
/*##########################################*/


void outIstEnergy(std::vector <std::complex <double>>& E, std::string outfile) {

  //Save on file the istantaneous values of the energy
  //during a single Monte Carlo run

  std::ofstream out(outfile.c_str());
  if(!out.good()){

    std::cerr << "##Error: Cannot open the file " << outfile << " for writing the istantaneous energies." << std::endl;
    std::abort();

  }
  else{

    std::cout << "#Saving the istantaneous energies on file " << outfile << std::endl;
    for(unsigned int j=0; j<E.size(); j++)
      out << E[j].real() << std::endl;

  }
  out.close();

}


void AutoCorrelation(std::vector <std::complex <double>>& X, std::string outfile, unsigned int tmax=1000) {

  double m, m2, mean1, mean2, cov1, cov2, norm;
  double n = 1.0/tmax;
  std::ofstream out(outfile.c_str());

  if(!out.good()){

    std::cerr << "##Error: Cannot open the file " << outfile << " for writing the autocorrelation function." << std::endl;
    std::abort();

  }
  else{

    std::cout << "#Saving the autocorrelation function on file " << outfile << std::endl;
    for(unsigned int t=0; t<=tmax; t++){

      m = 0.0;
      m2 = 0.0;
      mean1 = 0.0;
      mean2 = 0.0;
      cov1 = 0.0;
      cov2 = 0.0;
      norm = 1.0/(tmax-t);

      for(unsigned int tprime=0; tprime<=(tmax-t); tprime++){

        cov1 += X[tprime].real()*X[tprime+t].real();
        mean1 += X[tprime].real();
        mean2 += X[tprime+t].real();

      }
      for(unsigned int tprime=0; tprime<=tmax; tprime++){

        m += X[tprime].real();
        m2 += std::pow(X[tprime].real(), 2);

      }
      out << std::setprecision(10) << (norm*cov1 - norm*mean1*norm*mean2)/(n*m2 - std::pow(n*m, 2)) << std::endl;

    }

  }


}


#endif
