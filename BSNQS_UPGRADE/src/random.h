/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Prof. D.E. Galli
_/    _/  _/_/_/  _/_/_/_/ email: Davide.Galli@unimi.it
*****************************************************************
*****************************************************************/


#ifndef __RANDOM__
#define __RANDOM__


class Random {

  private:

    int m1,m2,m3,m4,l1,l2,l3,l4,n1,n2,n3,n4;

  public:

    //Constructors and Destructor
    Random();
    ~Random();

    //Methods
    void SetRandom(int * , int, int);
    void SaveSeed(int);
    double Rannyu(void);  //RV Uniformly Distributed in [0,1)
    double Rannyu(double min, double max);  //RV Uniformly Distributed in [min,max)
    int Rannyu_INT(int min, int max);  //RV Uniformly Distributed in [min, max]
    double Gauss(double mean, double sigma);  //Box-Muller
    double Lorentzian(double gamma, double mean);
    double Exp(double lambda);
    double Exp(double lambda, double min, double max);
    double Theta(void);  //p(Theta)=1/2 sin(Theta)
  		                   //Solid Angle

};


#endif


/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Prof. D.E. Galli
_/    _/  _/_/_/  _/_/_/_/ email: Davide.Galli@unimi.it
*****************************************************************
*****************************************************************/
