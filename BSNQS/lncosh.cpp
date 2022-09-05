/*#######################################################################*/
//  We write a very simple code to estimate the computation efficiency
//  (if any) related to the calculation of some logarithm that appears
//  many times when a RBM ansatz is used in the sampler.cpp code.
//  In particular let us compare the CPU time spent by the machine to
//  directly calculate the
//
//          ln(cosh(x))
//  through the basic functions
//          std::log() & std::cosh()
//
//  built in the C++ stl, compared to the case in which this calculation
//  is replaced by an asymptotic expansion for large x (x is real), i.e.
//
//          ln[cosh(x)] = ln[1/2 • (eˣ + 1/eˣ)]
//                      ~ ln[1/2 • eˣ]
//                      = ln(eˣ) - ln2
//                      = x - ln2
//
//  trying to establish even the domain in which this expansion results
//  convenient in terms of computational efficiency.
//  We also compare the case in which the quantity ln2 should always be
//  calculated or saved it once in memory and then recall the variable
//  that contains it.
/*#######################################################################*/


#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <ctime>


double f(double arg){

  double ln2 = std::log(2);
  if(arg <= 4)
    return std::log(std::cosh(arg));
  else
    return arg-ln2;
}


int main(){


  //Variables
  const double ln2 = std::log(2);  //Keeps ln(2) in memory
  double lncosh1 = 0;
  double lncosh2 = 0;
  double lncosh3 = 0;
  std::ofstream out("times.dat");

  //Time calculation
  //times.dat file ------> | x | time_for_lncosh | time_for_~ | lncosh | x-ln2 |
  for(double x=1.0; x<=500; x=x+0.5){  //Define the domain

    out << std::setprecision(2) << std::fixed << x << " ";
    std::clock_t t_start = std::clock();
    for(unsigned int Mval=0; Mval<45000; Mval++)
      lncosh1 = std::log(std::cosh(x));
    std::clock_t t_end = std::clock();
    long double time_ms = 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC;  //Calculates elapsed time in milliseconds
    out << std::setprecision(8) << std::fixed << time_ms / 1000.0 << " ";  //Prints on file elapsed time in seconds

    t_start = std::clock();
    for(unsigned int Mval=0; Mval<45000; Mval++)
      lncosh2 = x - ln2;
    t_end = std::clock();
    time_ms = 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC;
    out << std::setprecision(8) << std::fixed << time_ms / 1000.0 << " ";
    out << std::setprecision(8) << lncosh1 << " " << lncosh2 << " ";

    t_start = std::clock();
    for(unsigned int Mval=0; Mval<45000; Mval++)
      lncosh3 = f(x);
    t_end = std::clock();
    time_ms = 1000.0 * (t_end - t_start) / CLOCKS_PER_SEC;
    out << std::setprecision(8) << std::fixed << time_ms / 1000.0 << std::endl;

  }

  out.close();
  return 0;

}
