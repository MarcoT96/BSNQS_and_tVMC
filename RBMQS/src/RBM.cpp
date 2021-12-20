#ifndef __RBM__
#define __RBM__


#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <complex>
#include "random.h"  //(pseudo)-random numbers generator
                     //NY University [Percus & Kalos, 1989]


/*******************************************************************************************************************************/
/*************************************   Represent Many-Body Quantum State as a  ***********************************************/
/*************************************        Restricted Boltzmann Machine       ***********************************************/
/*******************************************************************************************************************************/

class RBM {

  private:

    //Define the architecture of the Neural Network
    unsigned int _N;  //Number of visible neurons S = (S1,...,SN)
    unsigned int _M;  //Number of hidden neurons {h1,...,hM}

    //Define the parameters of the Neural Network
    //which fully specify the MB Wave Function
    std::vector <std::vector <std::complex <double>>> _W;  //Weight matrix
    std::vector <std::complex <double>> _a;  //Visible bias
    std::vector <std::complex <double>> _b;  //Hidden bias

    //Random device
    Random _rnd;

    /*Useful quantities*/
    //Look-up table for the effective angles
    std::vector <std::complex<double>> _ThetaS;

  public:

    //Constructor and Destructor
    RBM();
    RBM(unsigned int, unsigned int);
    RBM(std::string);
    ~RBM();

    //Access functions
    inline unsigned int N() const {return _N;}
    inline unsigned int M() const {return _M;}
    inline unsigned int N_spin() const {return _N;}
    inline std::vector <std::vector <std::complex <double>>> W() const {return _W;}
    inline std::vector <std::complex <double>> a() const {return _a;}
    inline std::vector <std::complex <double>> b() const {return _b;}
    inline std::vector <std::complex <double>> effAngle() const {return _ThetaS;}
    std::complex <double> Wjk(unsigned int, unsigned int) const;
    std::complex <double> aj(unsigned int) const;
    std::complex <double> bk(unsigned int) const;
    std::complex <double> ThetaS_k(unsigned int) const;
    void printW() const;
    void print_a() const;
    void print_b() const;
    void print_ThetaS() const;

    //Modifier functions
    inline void set_N(unsigned int N) {_N = N;}
    inline void set_M(unsigned int M) {_M = M;}
    void set_Wjk (unsigned int, unsigned int, const std::complex <double>&);
    void set_aj (unsigned int, const std::complex <double>&);
    void set_bk (unsigned int, const std::complex <double>&);
    void set_ThetaS_k (unsigned int, const std::complex <double>&);

    /*
      Useful quantities for the Stochastic Reinforcement Learning
      of the variational parameters and for the Metropolis Algorithm
      and the sampling of the square modulus of the wave function.
    */
    void Init_ThetaS(const std::vector <int>&);
    double lncosh(double) const;  //real argument
    std::complex <double> lncosh(std::complex <double>) const;  //complex argument
    std::complex <double> logWave(const std::vector <int>&) const;
    std::complex <double> Wave(const std::vector <int>&) const;
    void Update_ThetaS(const std::vector <int>&, const std::vector <int>&);
    std::complex <double> logPsiNew_over_PsiOld(const std::vector <int>&, const std::vector <int>&) const;
    std::complex <double> PsiNew_over_PsiOld(const std::vector <int>&, const std::vector <int>&) const;

};

/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


RBM :: RBM() {

  /*
    Trivial constructor
    Initializes all the members to zero
    It will not be the standard choice for the
    construction of the quantum state.
  */

  //Information
  std::cout << "#Create a trivial RBM wave function with zero neurons" << std::endl;
  std::cout << " This should be only used as a check" << std::endl;

  _N = 0;
  _M = 0;

  /* Creates and initializes the Random Number Generator */
  std::cout << " Create and initialize the random number generator" << std::endl;
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input/Primes");
  if(Primes.is_open()){
    Primes >> p1 >> p2;
  }
  else{

    std::cerr << " ##PROBLEM: Unable to open Primes." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }
  Primes.close();
  std::ifstream input("./input/seed1.in");
  std::string property;
  if(input.is_open()){
    while(!input.eof()){
      input >> property;
      if(property == "RANDOMSEED"){
        input >> seed[0] >> seed[1] >> seed[2] >> seed[3];
        _rnd.SetRandom(seed, p1, p2);
      }
    }
    input.close();
    std::cout << " Random device created correctly" << std::endl << std::endl;
  }
  else{

    std::cerr << " ##PROBLEM: Unable to open seed1.in." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }

}


RBM :: RBM(unsigned int n_visible, unsigned int n_hidden)
           : _N(n_visible), _M(n_hidden) {

  /*
    Initially the network weights {a, b, W} are
    set to some small random numbers and only then
    optimized with the procedure outlined in the
    Jupyter-Notebook by the sampler class.
  */

  //Information
  std::cout << "#Create a RBM wave function with randomly initialized variational parameters" << std::endl;
  std::cout << " This should be used as starting point of the Stochastic Reinforcement Learning" << std::endl;

  /* Create and initializes the Random Number Generator */
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input/Primes");
  if(Primes.is_open()){
    Primes >> p1 >> p2;
  }
  else{

    std::cerr << " ##PROBLEM: Unable to open Primes." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }
  Primes.close();
  std::ifstream input("./input/seed1.in");
  std::string property;
  if(input.is_open()){
    while(!input.eof()){
      input >> property;
      if(property == "RANDOMSEED"){
        input >> seed[0] >> seed[1] >> seed[2] >> seed[3];
        _rnd.SetRandom(seed, p1, p2);
      }
    }
    input.close();
    std::cout << " Random device created correctly" << std::endl;
  }
  else{

    std::cerr << " ##PROBLEM: Unable to open seed1.in." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }

  /* Initializes bias and weights [G.Hinton, 2010] */
  _W.resize(_N, std::vector <std::complex <double>> (_M));
  _a.resize(_N);
  _b.resize(_M);
  _ThetaS.resize(_M);

  //Visible bias
  for(unsigned int j=0; j<_N; j++)
    _a[j] = 0.0;
  //Hidden bias
  for(unsigned int k=0; k<_M; k++)
    _b[k] = 0.0;
  //visible-hidden interaction weights
  for(unsigned int j=0; j<_N; j++){
    for(unsigned int k=0; k<_M; k++){

      _W[j][k].real(_rnd.Gauss(0.0, 0.1));
      _W[j][k].imag(_rnd.Gauss(0.0, 0.1));

    }
  }

  std::cout << " RBM correctly initialized with random bias & weights" << std::endl;
  std::cout << " Number of visible neurons = " << _N << std::endl;
  std::cout << " Number of hidden neurons = " << _M << std::endl << std::endl;;

}


RBM :: RBM(std::string filename) {

  /*
    Initializes the wave function parameters from a
    given file; this can be useful in a second moment
    during a check phase after the stochastic optimization or
    to start a time-dependent variational Monte Carlo with
    a previously optimized ground state wave function.
  */

  //Information
  std::cout << "#Create a RBM wave function from an existing file" << std::endl;
  std::cout << " This should be used in a final check simulation at the end of the optimization" << std::endl;
  std::cout << " or in the case of real time dynamics of the quantum state" << std::endl;
  std::cout << " It can be also used for other reasons" << std::endl;

  std::ifstream input_wf(filename.c_str());
  if(!input_wf.good()){

    std::cerr << " ##Error: opening file. " << filename << " failed." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::abort();

  }

  /* Create and initializes the Random Number Generator */
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input/Primes");
  if(Primes.is_open()){
    Primes >> p1 >> p2;
  }
  else{

    std::cerr << " ##PROBLEM: Unable to open Primes." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }
  Primes.close();
  std::ifstream input("./input/seed1.in");
  std::string property;
  if(input.is_open()){
    while(!input.eof()){
      input >> property;
      if(property == "RANDOMSEED"){
        input >> seed[0] >> seed[1] >> seed[2] >> seed[3];
        _rnd.SetRandom(seed, p1, p2);
      }
    }
    input.close();
    std::cout << " Random device created correctly" << std::endl;
  }
  else{

    std::cerr << " ##PROBLEM: Unable to open seed.in." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }

  //Initialize RBM parameters
  input_wf >> _N;
  input_wf >> _M;
  if(!input_wf.good() || _N<0 || _M<0){

    std::cerr << " ##Error: invalid construction of the RBM." << std::endl;
    std::abort();

  }
  _W.resize(_N, std::vector < std::complex <double> > (_M));
  _a.resize(_N);
  _b.resize(_M);
  _ThetaS.resize(_M);
  for(int j=0; j<_N; j++)
    input_wf >> _a[j];
  for(int k=0; k<_M; k++)
    input_wf >> _b[k];
  for(int j=0; j<_N; j++){
    for(int k=0; k<_M; k++){
      input_wf >> _W[j][k];
    }
  }

  if(input_wf.good()){
    std::cout << " RBM correctly loaded from file " << filename << std::endl;
    std::cout << " Number of visible neurons = " << _N << std::endl;
    std::cout << " Number of hidden neurons = " << _M << std::endl << std::endl;
  }
  input_wf.close();

}


RBM :: ~RBM() {

  _rnd.SaveSeed();

}


std::complex <double> RBM :: Wjk(unsigned int j, unsigned int k) const {

  if(j>=_N || k>=_M){

    std::cerr << "#Error: accessing weight matrix failed." << std::endl;
    std::cerr << " Element W_" << j << k << " does not exist" << std::endl;
    return -1.0;

  }
  else
    return _W[j][k];

}


std::complex <double> RBM :: aj(unsigned int j) const {

  if(j>=_N){

    std::cerr << "#Error: accessing visible bias failed." << std::endl;
    std::cerr << " Element a_" << j << " does not exist" << std::endl;
    return -1.0;

  }
  else
    return _a[j];

}


std::complex <double> RBM :: bk(unsigned int k) const {

  if(k>=_M){

    std::cerr << "#Error: accessing hidden bias failed." << std::endl;
    std::cerr << " Element b_" << k << " does not exist" << std::endl;
    return -1.0;

  }
  else
    return _b[k];

}


std::complex <double> RBM :: ThetaS_k(unsigned int k) const {

  if(k>=_M){

    std::cerr << "#Error: accessing effective angles failed." << std::endl;
    std::cerr << " Element ThetaS_" << k << " does not exist" << std::endl;
    return -1.0;

  }
  else
    return _ThetaS[k];

}


void RBM :: printW() const {

  std::cout << "\n====================" << std::endl;
  std::cout << "RBM Weight Matrix" << std::endl;
  std::cout << "====================" << std::endl;
  for(auto visible : _W){  //rows represents the connections
                           //of the single visible Quantum variable
    for(auto w : visible){

      std::cout << w.real();
      if(w.imag() >= 0)
        std::cout << " + i" << w.imag() << "  ";
      else
        std::cout << " - i" << -1.0*w.imag() << "  ";

    }
    std::cout << std::endl;
  }

}


void RBM :: print_a() const{

  std::cout << "\n====================" << std::endl;
  std::cout << "RBM Visible Bias" << std::endl;
  std::cout << "====================" << std::endl;
  for(auto a : _a){

    std::cout << a.real();
    if(a.imag() >= 0)
      std::cout << " + i" << a.imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0*a.imag() << "  " << std::endl;

  }

}


void RBM :: print_b() const{

  std::cout << "\n====================" << std::endl;
  std::cout << "RBM Hidden Bias" << std::endl;
  std::cout << "====================" << std::endl;
  for(auto b : _b){

    std::cout << b.real();
    if(b.imag() >= 0)
      std::cout << " + i" << b.imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0*b.imag() << "  " << std::endl;

  }

}


void RBM :: print_ThetaS() const{

  std::cout << "\n=============================" << std::endl;
  std::cout << "RBM Current Effective Angles" << std::endl;
  std::cout << "=============================" << std::endl;
  for(auto theta : _ThetaS){

    std::cout << theta.real();
    if(theta.imag() >= 0)
      std::cout << " + i" << theta.imag() << std::endl;
    else
      std::cout << " + i" << -1.0*theta.imag() << std::endl;

  }

}


void RBM :: set_Wjk(unsigned int j, unsigned int k, const std::complex <double>& Wjk) {

  if(j>=_N || k>=_M){

    std::cerr << "#Error: accessing weight matrix failed." << std::endl;
    std::cerr << " Element W_" << j << k << " does not exist" << std::endl;
    std::abort();

  }
  else
    _W[j][k] = Wjk;

}


void RBM :: set_aj(unsigned int j, const std::complex <double>& aj) {

  if(j>=_N){

    std::cerr << "#Error: accessing visible bias failed." << std::endl;
    std::cerr << " Element a_" << j << " does not exist" << std::endl;
    std::abort();

  }
  else
    _a[j] = aj;

}


void RBM :: set_bk(unsigned int k, const std::complex <double>& bk) {

  if(k>=_M){

    std::cerr << "#Error: accessing visible bias failed." << std::endl;
    std::cerr << " Element b_" << k << " does not exist" << std::endl;
    std::abort();

  }
  else
    _b[k] = bk;

}


void RBM :: set_ThetaS_k(unsigned int k, const std::complex <double>& thetak) {

  if(k>=_M){

    std::cerr << "#Error: accessing visible bias failed." << std::endl;
    std::cerr << " Element ThetaS_" << k << " does not exist" << std::endl;
    std::abort();

  }
  else
    _ThetaS[k] = thetak;

}


void RBM :: Init_ThetaS(const std::vector <int>& current_state) {

  /*
    The effective angles that appear in the wave functions
    described by this C++ Class depend on (in addition to the
    parameters of the Network (b, W)) the Quantum (spin)
    variables that define the current state of the Quantum system.
    (see the theory in the Jupyter Notebook)
    The effective angles serve both in the estime of the Energy and
    other observables (Metropolis algortihm) and in the stochastic
    optimization of the variational parameters (SR & TD-VMC, see
    Jupyter Notebook).
  */

  if(_ThetaS.size()!=_M)
    _ThetaS.resize(_M);
  else{

    for(int k=0; k<_M; k++){

      _ThetaS[k] = _b[k];
      for(int m=0; m<_N; m++)
        _ThetaS[k] += double(current_state[m])*(_W[m][k]);

    }

  }

}


double RBM :: lncosh(double x) const {

  //Computes the function for real argument
  //Do I need to use an asymptotic expansion
  //for efficiency reasons when x is large?!
  return std::log(std::cosh(x));

}


std::complex <double> RBM :: lncosh(std::complex <double> z) const {

  /*
    Carleo: I don't understand this way
  */
  //Computes the function for complex argument
  /*
  const double Re_z = z.real();
  const double Im_z = z.imag();

  std::complex <double> res = lncosh(theta_r);
  res += std::log(std::complex <double> (std::cos(theta_i), std::tanh(theta_r)*std::sin(theta_i)));

  return res;
  */

  return std::log(std::cosh(z));

}


std::complex <double> RBM :: logWave(const std::vector <int>& current_state) const {

  std::complex <double> rbm(0.0, 0.0);
  std::complex <double> theta(0.0, 0.0);

  //contribution of the visible layer only
  for(int j=0; j<_N; j++)
    rbm += _a[j]*double(current_state[j]);

  //contribution of both the hidden layer only
  //and the interaction visible-hidden
  //i.e. term related to the effective angles
  for(int k=0; k<_M; k++){

    theta = _b[k];
    for(int m=0; m<_N; m++)
      theta += double(current_state[m])*_W[m][k];
    rbm += lncosh(theta);

  }

  return rbm + _M*std::log(2.0);

}


std::complex <double> RBM :: Wave(const std::vector <int>& current_state) const {

  return std::exp(logWave(current_state));

}


void RBM :: Update_ThetaS(const std::vector <int>& current_state, const std::vector <int>& flipped_site) {

  if(flipped_site.size()==0)
    return;
  else{

    for(int k=0; k<_M; k++){

      for(const auto& m_flipped : flipped_site)
        _ThetaS[k] -= 2.0*double(current_state[m_flipped])*_W[m_flipped][k];

    }

  }

}


std::complex <double> RBM :: logPsiNew_over_PsiOld(const std::vector <int>& current_state, const std::vector <int>& flipped_site) const {

  /*
    This function computes the logarithm of the ratio
    between the wave function evaluated in a new proposed state {S_new}
    and the wave function in the current spin state
    (at fixed variational parameters {a, b, W}), i.e. this quantity will
    be used in the determination of the acceptance probability
    in the Metropolis Algorithm).
    The new proposed state is a state with a certain number of
    flipped spins wrt the current spin configuration; in fact the
    second argument of the function represents the list of the
    site to be flipped.
    Note that the ratio between the two evaluated wave function, which is
    the quantity related to the acceptance kernel of the Metropolis algorithm is
    recovered by taking the exponential function of the output of this function.
  */

  if(flipped_site.size()==0)
    return 0.0;  //log(1) = 0
  else{

    std::complex <double> log(0.0, 0.0);
    std::complex <double> theta(0.0, 0.0);
    std::complex <double> theta_prime(0.0, 0.0);

    //Change due to the visible layer
    for(const auto& j_flipped : flipped_site)
      log -= 2.0*_a[j_flipped]*double(current_state[j_flipped]);

    //Change due to the visible-hidden interaction
    //i.e. related to the effective angles
    for(int k=0; k<_M; k++){

      theta = _ThetaS[k];  //speed-up the calculation
      theta_prime = theta; //by exploiting the effective angle
                           //look-up table
      for(const auto& m_flipped : flipped_site)
        theta_prime -= 2.0*double(current_state[m_flipped])*_W[m_flipped][k];
      log += lncosh(theta_prime)-lncosh(theta);

    }
    return log;

  }

}


std::complex <double> RBM :: PsiNew_over_PsiOld(const std::vector <int>& current_state, const std::vector <int>& flipped_site) const {

  return std::exp(logPsiNew_over_PsiOld(current_state, flipped_site));

}


#endif
