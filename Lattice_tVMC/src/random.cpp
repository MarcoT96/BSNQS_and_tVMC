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


#ifndef __RANDOM__
#define __RANDOM__


/********************************************************************************************************/
/***********************************  (𝑷𝒔𝒆𝒖𝒅𝒐)-𝒓𝒂𝒏𝒅𝒐𝒎 𝒏𝒖𝒎𝒃𝒆𝒓𝒔 𝒈𝒆𝒏𝒆𝒓𝒂𝒕𝒐𝒓  ***********************************/
/********************************************************************************************************/
/*

  This class considers the family of 𝒍𝒊𝒏𝒆𝒂𝒓 𝒄𝒐𝒏𝒈𝒓𝒖𝒆𝒏𝒕𝒊𝒂𝒍 𝒈𝒆𝒏𝒆𝒓𝒂𝒕𝒐𝒓𝒔

            𝒳n+𝟣 = (𝑎 • 𝒳n + 𝑐) 𝑚𝑜𝑑(𝑚)

  and is able to generate a sequence of (pseudo)-random numbers particularly suitable for the
  parallelization of Monte Carlo codes. In fact it provides reproducibility of runs, very long
  sequences, and assuring an adequate degrees of independence of the parallel streams.
  In particular, the generator acts by creating a so-called pseudo-random trees, in which
  the overlapping between the various branches of the tree is avoided.
  Once generated random numbers uniformly distributed in [𝟢, 𝟣), i.e. 𝒳n / 𝑚, the methods of this class
  allow to generate the simplest probability distributions, which do not require sophisticated
  techniques, but at most the inversion of the cumulative function.
  In this implementation, 𝑚 = 𝟤^𝟦𝟪, which leads to a period of the generator equals to ~ 𝟣𝟢^𝟣𝟦.
  The starting seeds to generate the various independent branches of the tree must meet certain
  properties and are read in the file 𝐢𝐧𝐩𝐮𝐭_𝐫𝐚𝐧𝐝𝐨𝐦_𝐝𝐞𝐯𝐢𝐜𝐞/𝐬𝐞𝐞𝐝*.𝐢𝐧, while the prime numbers necessary to
  the algorithm are read in two possible files: one containing 𝟥𝟪𝟦 suited prime numbers in
  𝐢𝐧𝐩𝐮𝐭_𝐫𝐚𝐧𝐝𝐨𝐦_𝐝𝐞𝐯𝐢𝐜𝐞/𝐏𝐫𝐢𝐦𝐞𝐬_𝟯𝟴𝟰.𝐢𝐧, and one that allows to generate many more branches, which contains
  𝟥𝟤𝟢𝟢𝟣 prime numbers, always following the same criterion of independence of the sequences generated
  in the various MIMD processes, in 𝐢𝐧𝐩𝐮𝐭_𝐫𝐚𝐧𝐝𝐨𝐦_𝐝𝐞𝐯𝐢𝐜𝐞/𝐏𝐫𝐢𝐦𝐞𝐬_𝟯𝟮𝟬𝟬𝟭.𝐢𝐧.
  All these numbers are represented in base 𝟦𝟢𝟫𝟨, that is in base of 𝟤^𝟣𝟤.
  This code was provided by 𝒫𝓇ℴ𝒻. 𝒟𝒶𝓋𝒾𝒹ℯ ℰ. 𝒢𝒶𝑙𝑙𝒾 and 𝒟𝓇. 𝒢𝒾𝒶𝓃𝑙𝓊𝒸𝒶 ℬℯ𝓇𝓉𝒶𝒾𝓃𝒶.

  For more details see

    [Percus & Kalos, 1989, NY University, 0743-7315/89, Journal of Parallel and Distributed Computing].

*/
/********************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etc…
#include <cstdlib>  // <-- std::abort()
#include <cmath>  // <-- std::cos(), std::log(), std::exp(), std::M_PI, std::acos()
#include <filesystem>  // <-- is_directory(), exists(), create_directory()
/*
  Use
    #include <experimental/filesystem>
  if you are in @tolab!
*/
#include <fstream>  // <-- std::ifstream, std::ofstream


using namespace std::__fs::filesystem;  //Use std::experimental::filesystem if you are in @tolab


class Random {

  private:

    int m1, m2, m3, m4;  //Constants of the generator
    int l1, l2, l3, l4;  //The seed of the generator
    int n1, n2, n3, n4;  //The prime number pairs

  public:

    //Constructors and Destructor
    Random() {};
    ~Random() {};

    //Methods
    void SetRandom(int*, int, int);  //Sets the parameters defining the device
    void SaveSeed(int, int only_one_rank=1);  //Save the last ℛ.𝓋. in order to restart from that point in the sequence
    double Rannyu();  //Real ℛ.𝓋. uniformly distributed in [0, 1)
    double Rannyu(double, double);  //Real ℛ.𝓋. uniformly distributed in a certain interval [, )
    int Rannyu_INT(int, int);  //Integer ℛ.𝓋. uniformly distributed in a certain interval [, ]
    double Gauss(double, double);  //Real ℛ.𝓋. generated via Box-Muller algorithm
    double Lorentzian(double, double);  //Real ℛ.𝓋. distributed according to a Lorentzian distribution
    double Exp(double);  //Real ℛ.𝓋. exponentially distributed
    double Exp(double, double, double);  //Real ℛ.𝓋. exponentially distributed in a certain interval [, ]
    double Theta();  //Real ℛ.𝓋. representing a solid angle, i.e. distributed in according to 𝑝(ϑ) = 𝟣/𝟤 𝑠𝑖𝑛(ϑ)

};




/********************************************************************************************************************************/
/******************************************  (𝐏𝐒𝐄𝐔𝐃𝐎)-𝐑𝐀𝐍𝐃𝐎𝐌 𝐍𝐔𝐌𝐁𝐄𝐑𝐒 𝐆𝐄𝐍𝐄𝐑𝐀𝐓𝐎𝐑  *******************************************/
/*******************************************************************************************************************************/
void Random :: SetRandom(int* s, int p1, int p2) {

  /*####################################################################*/
  //  N̲O̲T̲E̲: the acquisition from files described above is done directly
  //        in the classes that contain among their data members an
  //        instance of the 𝐑𝐚𝐧𝐝𝐨𝐦 class. So the vector of the seeds 𝐬
  //        for example is passed from the outside, and so the pair
  //        of prime numbers.
  /*####################################################################*/

  m1 = 502;
  m2 = 1521;
  m3 = 4071;
  m4 = 2107;
  l1 = s[0];
  l2 = s[1];
  l3 = s[2];
  l4 = s[3];
  n1 = 0;
  n2 = 0;
  n3 = p1;
  n4 = p2;

}


void Random :: SaveSeed(int rank, int only_one_rank) {

  //Function variables
  std::ofstream write_Seed;
  if(only_one_rank == 1){  //Default case

    if(rank == 0){

      if(is_directory("./CONFIG") || exists("./CONFIG")){

        write_Seed.open("./CONFIG/seed_node_" + std::to_string(rank) + ".out");
        if(write_Seed.is_open()) write_Seed << "RANDOMSEED\t" << l1 << " " << l2 << " " << l3 << " " << l4 << std::endl;
        else std::cerr << "PROBLEM: Unable to open seed_node_" + std::to_string(rank) + ".out" << std::endl;
        write_Seed.close();

      }

    }

  }
  else{

    if(is_directory("./CONFIG") || exists("./CONFIG")){

      write_Seed.open("./CONFIG/seed_node_" + std::to_string(rank) + ".out");
      if(write_Seed.is_open()) write_Seed << "RANDOMSEED\t" << l1 << " " << l2 << " " << l3 << " " << l4 << std::endl;
      else std::cerr << "PROBLEM: Unable to open seed_node_" + std::to_string(rank) + ".out" << std::endl;
      write_Seed.close();

    }

  }

}


double Random :: Rannyu() {

  const double twom12 = 0.000244140625;
  int i1, i2, i3, i4;
  double r;

  i1 = l1 * m4 + l2 * m3 + l3 * m2 + l4 * m1 + n1;
  i2 = l2 * m4 + l3 * m3 + l4 * m2 + n2;
  i3 = l3 * m4 + l4 * m3 + n3;
  i4 = l4 * m4 + n4;
  l4 = i4 % 4096;
  i3 = i3 + i4 / 4096;
  l3 = i3 % 4096;
  i2 = i2 + i3 / 4096;
  l2 = i2 % 4096;
  l1 = (i1 + i2 / 4096) % 4096;
  r = twom12 * (l1 + twom12 * (l2 + twom12 * (l3 + twom12 * (l4))));

  return r;

}


double Random :: Rannyu(double min, double max) {

  return min + (max - min) * this->Rannyu();

}


int Random :: Rannyu_INT(int min, int max) {

  return min + (int)(((max - min) + 1) * this->Rannyu());

}


double Random :: Gauss(double mean, double sigma) {

  double s = this->Rannyu();
  double t = this->Rannyu();
  double x = std::sqrt(-2.0 * std::log(1.0 - s)) * std::cos(2.0 * M_PI * t);
  return mean + x * sigma;

}


double Random :: Lorentzian(double gamma, double mean) {

  double s = this->Rannyu();
  return gamma * tan(M_PI * (s - 1.0/2.0)) + mean;

}


double Random :: Exp(double lambda) {

  double s = this->Rannyu();
  return - (1.0 / lambda) * std::log(1.0 - s);

}


double Random :: Exp(double lambda, double min, double max) {

  double m = 1.0 - std::exp((-1.0) * lambda * max);
  double M = 1.0 - exp((-1.0) * lambda * min);
  double x = this->Rannyu(m, M);
  return - (1.0 / lambda) * log(1.0 - x);

}


double Random :: Theta() {

  double r = this->Rannyu();
  return std::acos(1.0 - 2.0 * r);

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


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
