#ifndef __SAMPLER__
#define __SAMPLER__


#define ARMA_DONT_USE_WRAPPER


#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>
#include <complex>
#include <armadillo>
#include "random.h"  //(pseudo)-random numbers generator
                     //NY University [Percus & Kalos, 1989]


/*******************************************************************************************************************************/
/**************************************      Monte  Carlo  Sampling    *********************************************************/
/**************************************        Ground State: VMC       *********************************************************/
/**************************************    Unitary Dynamics: TD-VMC    *********************************************************/
/*******************************************************************************************************************************/

/*
  Stochastic Reinforcement Learning of the weight parameters
  via Stochastic Reconfiguration (SR) (GS properties)
  or Time-Dependent VMC (Dynamical properties) and
  Monte Carlo sampling of the Neural Network Quantum State.

  Note: this Class is completely generic, both
        in the representation of the Quantum State
        (not necessarily the RBM ansatz) and in the
        system Hamiltonian.
*/

template <class WaveFunc, class Hamiltonian>
class MC_Sampler {

  private:

    WaveFunc& _wf;  //The wave function ansatz
    Hamiltonian& _H;  //The model Hamiltonian
    const int _Nspin;  //Number of spin in the system
    std::vector <int> _state;  //Current state in the sampling

    //Random device
    //If you want you can use the stl one
    Random _rnd;

    //Sampling members
    double _accept;
    double _totMoves;  //Number of proposed new states
    std::vector <int> _flipped_site;  //Indices of randomly chosen
                                      //spins to be flipped
    std::vector <std::complex <double>> _E;  //measured values of the energy
    std::vector <std::complex <double>> _Sx;  //measured values of the transverse polarization

    //Print options
    bool _writeState;  //Write sampled configurations on file
    std::ofstream _fileState;
    bool _writeEnergy;  //Write estimated energy during the RL
    std::ofstream _fileEnergy;
    bool _writeSigma;  //Write estimated Sx during the RL
    std::ofstream _fileSigma;
    bool _writeWave;  //Write the optimized wave function on file
    std::ofstream _fileWave;

    //SR & TD-VMC variables
    bool _real_time;  //imaginary- or real-time dynamics
    std::vector <std::complex <double>> _mel;  //non-zero matrix elements for the energy
    std::vector <std::complex <double>> _sigmax;  //non-zero matrix elements for sigmax
    std::vector <std::vector <int>> _flipsh;  //list of flipped spin site
                                              //associated to each _mel[j]
    double _Gamma;  //Scaling parameter
    double _deltat;  //Integration step parameter
    std::vector <std::vector <std::complex <double>>> _Oa;  //Visible bias local operators
    std::vector <std::vector <std::complex <double>>> _Ob;  //Hidden bias local operators
    std::vector <std::vector <std::vector <std::complex <double>>>> _Ow;  //Weight matrix local operators
    arma::Mat <std::complex <double>> _S;  //(Hermitian) Covariance matrix
    arma::Col <std::complex <double>> _f;  //Forces on the parameters, i.e. -gradE

  public:

    //Constructor and Destructor
    MC_Sampler(WaveFunc&, Hamiltonian&, double g=0.01);
    ~MC_Sampler();

    //Access functions
    WaveFunc& wf() const {return _wf;}
    Hamiltonian& H() const {return _H;}
    int Nspin() const {return _Nspin;}
    std::vector <int> current_state() const {return _state;}
    void print_state() const;
    double accept() const {return _accept;}
    double Moves() const {return _totMoves;}
    std::vector <std::complex <double>> E() const {return _E;}
    std::complex <double> Ej(unsigned int) const;
    std::vector <std::complex <double>> mel() const {return _mel;}
    std::vector <std::vector <int>> flipsh() const {return _flipsh;}
    double Gamma() const {return _Gamma;}
    double deltat() const {return _deltat;}
    std::vector <std::vector <std::complex <double>>> Oa() const {return _Oa;}
    std::vector <std::vector <std::complex <double>>> Ob() const {return _Ob;}
    std::vector <std::vector <std::vector <std::complex <double>>>> Ow() const {return _Ow;}
    arma::Mat <std::complex <double>> S() const {return _S;}
    std::complex <double> Smn(unsigned int, unsigned int) const;
    arma::Col <std::complex <double>> f() const {return _f;}
    std::complex <double> fk(unsigned int) const;

    //Modifier functions
    void Update_state(const std::vector <int>&);
    void Update_wf();
    void set_Gamma(double);
    void set_deltat(double);

    //Initialization functions
    void Init_State(bool zeroMag=true);  //Initialize a random spin state
    void setRealTimeDyn(double);  //Choose time-dependent VMC
    void setFileState(std::string);  //Create the output file for the sampled spin state
    void setFileEnergy(std::string);  //Create the output file for E({W})
    void setFileSigma(std::string);  //Create the output file for Sx({t})
    void setFileWave(std::string);  //Create the output file for \Psi_{opt} ({W})

    //Measurement functions
    void Energy();  //Istantaneous energy
    void Sx();  //Istantaneous Sx
    void Magnetization();  //Current magnetization
    void LocalOperators();  //Istantaneous local operators
    void Reset(std::string, std::string);  //Reset the variable after each MC Run
    void Blocking(int, std::string filene="energy.dat", std::string filesigma="sigmax.dat");
    void WriteState();  //Save on file the sampled configurations
    void WriteWave();  //Save on file the optimized wave function
    void CloseFile();  //Close output files after each MC Run

    //Sampling functions
    bool RandFlips(std::vector <int>&, int, bool zeroMag=true);
    void Move(int);
    void MC_Run(double, int, double eq_time=0.1, int Mfraction=1, int nflips=-1);

};

/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


template <class WaveFunc, class Hamiltonian>
MC_Sampler <WaveFunc, Hamiltonian> :: MC_Sampler(WaveFunc& wave, Hamiltonian& hamiltonian, double g)
                                    : _wf(wave), _H(hamiltonian), _Nspin(wave.N_spin()) {

  //Information
  std::cout << "#Define the Monte Carlo sampler of the Neural Network Quantum State" << std::endl;
  std::cout << " The sampler is defined for a specific wave function and a particular inferred hamiltonian" << std::endl;

  //Set the output options
  _writeState = false;
  _writeEnergy = false;
  _writeSigma = false;
  _writeWave = false;
  _real_time = false;  //default: ground state VMC

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
  std::ifstream input("./input/seed2.in");
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

    std::cerr << " ##PROBLEM: Unable to open seed2.in." << std::endl;
    std::cerr << "   Initialization of the random device failed." << std::endl;
    std::abort();

  }

  if(g <= 0){

    std::cerr << " ##Error: the scaling parameter must be a positive number." << std::endl;
    std::cerr << " ##Error: construction of the sampler failed." << std::endl;
    std::abort();

  }
  else
    _Gamma = g;
  _deltat = 0.0;
  _sigmax = _H.sigmax();

  /*
    Initialize the Covariance matrix and the forces
    for the Stochastic Reinforcement Learning
    of the variational wave function
  */
  unsigned int dim = _wf.N() + _wf.M() + _wf.N()*_wf.M();  //number of total variational parameters
  _S.zeros(dim, dim);
  _f.zeros(dim);

  _accept = 0;
  _totMoves = 0;

  std::cout << " Monte Carlo sampler correctly initialized" << std::endl << std::endl;

}


template <class WaveFunc, class Hamiltonian>
MC_Sampler <WaveFunc, Hamiltonian> :: ~MC_Sampler() {

  if(_writeState)
    _fileState.close();
  if(_writeEnergy)
    _fileEnergy.close();
  if(_writeSigma)
    _fileSigma.close();
  if(_writeWave)
    _fileWave.close();
  _rnd.SaveSeed();

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: print_state() const {

  std::cout << "\n===============" << std::endl;
  std::cout << "Current state" << std::endl;
  std::cout << "===============" << std::endl;

  for(auto s : _state)
    std::cout << s << "  ";
  std::cout << std::endl;

}


template <class WaveFunc, class Hamiltonian>
std::complex <double> MC_Sampler <WaveFunc, Hamiltonian> :: Ej(unsigned int j) const {

  if(j >= _E.size()){

    std::cerr << " ##Error: accessing Istantaneous energy of the system failed." << std::endl;
    std::cerr << "   Element E_" << j << " does not exist." << std::endl;
    return -1.0;

  }
  else
    return _E[j];

}


template <class WaveFunc, class Hamiltonian>
std::complex <double> MC_Sampler <WaveFunc, Hamiltonian> :: Smn(unsigned int m, unsigned int n) const {

  if(m >= _S.n_rows || n >= _S.n_cols){

    std::cerr << " ##Error: accessing Covariance matrix failed." << std::endl;
    std::cerr << "   Element S_" << m  << n << " does not exist." << std::endl;
    return -1.0;

  }
  else
    return _S(m,n);

}


template <class WaveFunc, class Hamiltonian>
std::complex <double> MC_Sampler <WaveFunc, Hamiltonian> :: fk(unsigned int k) const {

  if(k >= _f.n_rows){

    std::cerr << " ##Error: accessing forces on the parameters failed." << std::endl;
    std::cerr << "   Element f_" << k << " does not exist." << std::endl;
    return -1.0;

  }
  else
    return _f(k);

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Update_state(const std::vector <int>& new_state) {

  if(new_state.size() != _Nspin){

    std::cerr << " ##Error: spin state incompatible with the choosen model Hamiltonian." << std::endl;
    std::abort();

  }
  else
    _state = new_state;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Update_wf() {

  /*
    Solves the linear system

            \delta\alpha_k = \Gamma * ∑_k,k' S^(-1)_k,k' * f_k'

    by inverting the matrix _S and exploiting C++ Armadillo
    library.
    Then updates each variational parameters as

            {W}_k,new = {W}_k,current + \delta\alpha_k

    in the case of Ground State VMC or solve the differential equations

            d/dt {W}_k = -i ∑_k,k' S^(-1)_k,k' * f_k'

    in a Stochastic framework.
  */
  /*############################################################################*/
  /*############################################################################*/


  //Information
  //std::cout << "#Reinforcement Learning of the Neural Network Quantum State" << std::endl;
  //std::cout << " The program uses a constant scaling parameter Gamma = " << _Gamma << std::endl;

  /* Inverting the Covariance matrix */
  //std::cout << " Compute the inverse of the Covariance matrix ... ";
  //std::flush(std::cout);
  arma::Mat <std::complex <double>> Sinv = arma::pinv(_S);  //Moore-Penrose pseudo-inverse
  //std::cout << "done" << std::endl;

  /* Solving the Linear System */
  //std::cout << " Solve the linear system for the variation vector ... ";
  //std::flush(std::cout);
  arma::Col <std::complex <double>> delta = Sinv * _f;
  for(unsigned int j=0; j<delta.n_rows; j++){

    if(!_real_time)
      delta(j) *= _Gamma;
    else{

      const std::complex <double> i(0.0,1.0);
      delta(j) *= i*_deltat;  //Euler Integrator

    }

  }
  //std::cout << "done" << std::endl;

  /* Updating the RBM wave function */
  //std::cout << " Update the variational parameters ... ";
  //std::flush(std::cout);
  unsigned int index;

  //Check
  if(delta.n_rows != _S.n_rows){

    std::cerr << " ##Error: something went wrong in solving the linear equations system." << std::endl;
    std::abort();

  }

  //Updates visible bias
  for(unsigned int k_aj=0; k_aj<_wf.N(); k_aj++)
    _wf.set_aj(k_aj, _wf.aj(k_aj) - delta(k_aj));

  //Updates hidden bias
  for(unsigned int k_bk=0; k_bk<_wf.M(); k_bk++)
    _wf.set_bk(k_bk, _wf.bk(k_bk) - delta(_wf.N()+k_bk));

  //Update the weights matrix
  for(unsigned int kw_row=0; kw_row<_wf.N(); kw_row++){
    for(unsigned int kw_col=0; kw_col<_wf.M(); kw_col++){

      index = _wf.N() + _wf.M() + _wf.M()*kw_row + kw_col;
      _wf.set_Wjk(kw_row, kw_col, _wf.Wjk(kw_row, kw_col) - delta(index));

    }
  }

  //std::cout << "done" << std::endl;


  /*############################################################################*/
  /*############################################################################*/

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: set_Gamma(double g) {

  if(g <= 0){

    std::cerr << " ##Error: the scaling parameter must be a positive number." << std::endl;
    std::abort();

  }
  _Gamma = g;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: set_deltat(double d) {

  if(d <= 0){

    std::cerr << " ##Error: the integration step parameter must be a positive number." << std::endl;
    std::abort();

  }
  _deltat = d;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Init_State(bool zeroMag) {

  //Information
  //if(zeroMag)
    //std::cout << " Initialize the quantum state of the system (randomly) with zero total magnetization ... ";
  //else
    //std::cout << " Initialize the quantum state of the system (randomly) with non-zero total magnetization ... ";
  //std::flush(std::cout);

  //If zeroMag = true the initial state is prepared with
  //zero total magnetization
  _state.resize(_Nspin);
  /*
    conditional ternary operator
    condition ? result1 : result2
  */
  //Randomly choose spin up or spin down
  //This is a particular choice of the basis
  //in the Hilbert space of the system
  for(unsigned int j=0; j<_Nspin; j++)
    _state[j] = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

  if(zeroMag){  //In the case of zero magnetization (default case)

    if(!_Nspin%2){

      std::cerr << " ##Error: Cannot initialize a random spin state with zero magnetization for odd number of spin variables." << std::endl;
      std::abort();

    }
    int tempMag = 1;
    while(tempMag != 0){

      tempMag = 0;
      for(unsigned int j=0; j<_Nspin; j++)
        tempMag += _state[j];
      if(tempMag > 0){

        /* Select a random spin-UP */
        int rs = _rnd.Rannyu_INT(0, _Nspin-1);
        while(_state[rs]<0)
          rs = _rnd.Rannyu_INT(0, _Nspin-1);
        _state[rs] = -1;  //Flip that spin-UP in order to decrease
        tempMag -= 1;     //the positive magnetization

      }
      else if(tempMag < 0){

        /* Select a random spin-DOWN */
        int rs = _rnd.Rannyu_INT(0, _Nspin-1);
        while(_state[rs]>0)
          rs = _rnd.Rannyu_INT(0, _Nspin-1);
        _state[rs] = 1;  //Flip that spin-DOWN in order to increase
        tempMag += 1;    //the negative magnetization

      }

    }

  }
  //std::cout << "done" << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: setRealTimeDyn(double delta) {

  _real_time = true;
  _deltat = delta;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: setFileState(std::string outfile) {

  _writeState = true;
  _fileState.open(outfile.c_str());
  if(!_fileState.good()){

    std::cerr << " ##Error: Cannot open the file " << outfile << " for writing the sampled spin states." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the sampled spin configurations on file " << outfile << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: setFileEnergy(std::string outfile) {

  _writeEnergy = true;
  _fileEnergy.open(outfile.c_str());
  if(!_fileEnergy.good()){

    std::cerr << " ##Error: Cannot open the file " << outfile << " for writing the energy at the end of each Monte Carlo run." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the energy at the end of each Monte Carlo run on file " << outfile << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: setFileSigma(std::string outfile) {

  _writeSigma = true;
  _fileSigma.open(outfile.c_str());
  if(!_fileSigma.good()){

    std::cerr << " ##Error: Cannot open the file " << outfile << " for writing Sx at the end of each Monte Carlo run." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving Sx at the end of each Monte Carlo run on file " << outfile << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: setFileWave(std::string outfile) {

  _writeWave = true;
  _fileWave.open(outfile.c_str());
  if(!_fileWave.good()){

    std::cerr << " ##Error: Cannot open the file " << outfile << " for writing the updated wave function." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the updated wave function on file " << outfile << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Energy() {

  /*
    Measures the value of the local Energy

              E_loc(S) = 1/WaveFunc(S; {W}) * <S | H | Psi({W})>

    on the current state S of the system.
  */

  std::complex <double> E = 0.0;
  //Find the non zero matrix elements of the Hamiltonian
  _H.FindConn(_state, _flipsh, _mel);
  //Stochastic evaluation
  for(int j=0; j<_flipsh.size(); j++)
    E += _mel[j]*_wf.PsiNew_over_PsiOld(_state, _flipsh[j]);

  _E.push_back(E);

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Sx() {

  /*
    Measures the value of the local Sx

              Sx_loc(S) = 1/WaveFunc(S; {W(t)}) * <S | Sx | Psi({W(t)})>

    on the current state S of the system.
  */

  std::complex <double> Sx = 0.0;

  //Stochastic evaluation
  //Here _flipsh has already been filled
  //by the function Energy()
  for(unsigned int j=1; j<_flipsh.size(); j++)
    Sx += _sigmax[j-1]*_wf.PsiNew_over_PsiOld(_state, _flipsh[j]);

  _Sx.push_back(Sx);

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Magnetization() {

  /*
    Check function for the total magnetization of
    the current state of the system.
    For example, in the case of zeroMag=true it has to
    return 0.
  */

  int totMag = 0;
  for(unsigned int j=0; j<_Nspin; j++)
    totMag += _state[j];

  std::cout << " Total magnetization of the current spin state = " << totMag << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: LocalOperators() {

  /*
    Measures the value of all the local operators needed
    in the Stochastic optimization of the variational parameters

              O_k(S) = 1/WaveFunc(S; {W}) * \partial_{{W}_k} WaveFunc(S; {W})

    on the current state S of the system.
    The explicit equations relative to the various local operators associated
    to the different variational parameters {W} = {a, b, W} can be found in the
    Jupyter Notebook.
  */

  std::vector <std::complex <double>> Oa(_wf.N());
  std::vector <std::complex <double>> Ob(_wf.M());
  std::vector <std::vector <std::complex <double>>> O_W(_wf.N(), std::vector < std::complex <double> > (_wf.M()));

  //local operators for the visible bias a_j
  for(unsigned int j=0; j<_wf.N(); j++)
    Oa[j] = double(_state[j]);

  //local operators for the hidden bias b_k
  for(unsigned int k=0; k<_wf.M(); k++)
    Ob[k] = std::tanh(_wf.ThetaS_k(k));

  //local operators for the each
  //weight matrix element Wmn
  for(unsigned int m=0; m<_wf.N(); m++){
    for(unsigned int n=0; n<_wf.M(); n++)
      O_W[m][n] = double(_state[m])*std::tanh(_wf.ThetaS_k(n));
  }

  _Oa.push_back(Oa);
  _Ob.push_back(Ob);
  _Ow.push_back(O_W);

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Reset(std::string fileState, std::string fileWave) {

  /*
    Properly resets all the observables needed
    in the single Monte Carlo RUN.
  */

  //Resets the local operators O_k
  _Oa.clear();
  _Ob.clear();
  _Ow.clear();

  //Resets the istantaneous values of the local energy
  _E.clear();
  //Resets the istantaneous values of the local Sx
  _Sx.clear();

  //Resets the Covariance matrix and the Forces
  unsigned int dim = _wf.N() + _wf.M() + _wf.N()*_wf.M();  //number of total variational parameters
  _S.zeros(dim, dim);
  _f.zeros(dim);

  //Re-open the output files
  if(_writeState)
    _fileState.open(fileState.c_str());
  if(_writeWave)
    _fileWave.open(fileWave.c_str());

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Blocking(int Nblks, std::string filene, std::string filesigma) {

  //Information
  int blks_size = std::floor(double(_E.size()/(Nblks*1.0)));
  //std::cout << " Estimation of observables via Blocking method" << std::endl;
  //std::cout << " The program uses " << Nblks << " blocks of size " <<  blks_size;
  //std::cout << " for a total of " << Nblks*blks_size << " Monte Carlo steps" << std::endl;


  /*
    Estimates the average value of the energy of the system
    and its uncertainty. I also calculate the imaginary part
    to verify that it is statistically zero. This is an important
    check for the correctness of the code.
  */
  /*#####################################################################################################################*/
  /*#####################################################################################################################*/


  //Useful quantities
  double E_blk_re;
  double E_blk_im;
  double E_ave_re = 0.0, E_ave_im = 0.0;
  double E2_ave_re = 0.0, E2_ave_im = 0.0;
  double E_err_re = 0.0, E_err_im= 0.0;
  //std::cout << " Estimate of the energy per spin ... ";
  //std::flush(std::cout);

  //Blocking method
  for(unsigned int j=0; j<Nblks; j++){

    E_blk_re = 0.0;
    E_blk_im = 0.0;
    for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++){
      E_blk_re += _E[l].real();
      E_blk_im += _E[l].imag();
    }
    E_ave_re += E_blk_re/double(blks_size);
    E_ave_im += E_blk_im/double(blks_size);
    E2_ave_re += pow((E_blk_re/double(blks_size)), 2);
    E2_ave_im += pow((E_blk_im/double(blks_size)), 2);

  }

  //Last energy estimate
  E_ave_re = E_ave_re/(Nblks*1.0);
  E_ave_im = E_ave_im/(Nblks*1.0);
  E2_ave_re = E2_ave_re/(Nblks*1.0);
  E2_ave_im = E2_ave_im/(Nblks*1.0);
  E_err_re = sqrt((E2_ave_re-pow(E_ave_re, 2))/(double(Nblks-1)));
  E_err_im = sqrt((E2_ave_im-pow(E_ave_im, 2))/(double(Nblks-1)));
  //std::cout << "done" << std::endl;
  if(_writeEnergy){

    //setFileEnergy(filene.c_str());
    _fileEnergy << std::setprecision(10) << std::fixed << E_ave_re << std::setw(20) << E_err_re << std::setw(20) << E_ave_im << std::setw(20) << E_err_im << std::endl;

  }

  /*
    Estimates the average value of the transverse polarization
    along the x axis and its uncertainty.
  */
  /*#####################################################################################################################*/
  /*#####################################################################################################################*/


  //Useful quantities
  double Sx_blk_re;
  double Sx_blk_im;
  double Sx_ave_re = 0.0, Sx_ave_im = 0.0;
  double Sx2_ave_re = 0.0, Sx2_ave_im = 0.0;
  double Sx_err_re = 0.0, Sx_err_im= 0.0;
  //std::cout << " Estimate of the transverse polarization per spin ... ";
  //std::flush(std::cout);

  //Blocking method
  for(unsigned int j=0; j<Nblks; j++){

    Sx_blk_re = 0.0;
    Sx_blk_im = 0.0;
    for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++){
      Sx_blk_re += _Sx[l].real();
      Sx_blk_im += _Sx[l].imag();
    }
    Sx_ave_re += Sx_blk_re/double(blks_size);
    Sx_ave_im += Sx_blk_im/double(blks_size);
    Sx2_ave_re += pow((Sx_blk_re/double(blks_size)), 2);
    Sx2_ave_im += pow((Sx_blk_im/double(blks_size)), 2);

  }

  //Last energy estimate
  Sx_ave_re = Sx_ave_re/(Nblks*1.0);
  Sx_ave_im = Sx_ave_im/(Nblks*1.0);
  Sx2_ave_re = Sx2_ave_re/(Nblks*1.0);
  Sx2_ave_im = Sx2_ave_im/(Nblks*1.0);
  Sx_err_re = sqrt((Sx2_ave_re-pow(Sx_ave_re, 2))/(double(Nblks-1)));
  Sx_err_im = sqrt((Sx2_ave_im-pow(Sx_ave_im, 2))/(double(Nblks-1)));
  //std::cout << "done" << std::endl;
  if(_writeSigma){

    //setFileSigma(filesigma.c_str());
    _fileSigma << std::setprecision(10) << std::fixed << Sx_ave_re/(_wf.N()*1.0) << std::setw(20) << Sx_err_re/(_wf.N()*1.0) << std::setw(20);
    _fileSigma << Sx_ave_im/(_wf.N()*1.0) << std::setw(20) << Sx_err_im/(_wf.N()*1.0) << std::endl;

  }


  /*#####################################################################################################################*/
  /*#####################################################################################################################*/

  /*
    Estimates the Covariance matrix S_kk' and the
    forces f_k to solve the linear system

            S • \delta\alpha = \Gamma * f
            \delta\alpha_k = \Gamma * ∑_k,k' S^(-1)_k,k' * f_k'

    and update the variational parameters

            \alpha^(new)_k = \alpha^(current)_k + \delta\alpha_k

    in order to perform the Stochastic Reinforcement Learning
    of the RBM; we have two cases:

            – Ground State --> Stochastic Reconfiguration
            – Unitary Dynamics --> Time-Dependent VMC
  */
  /*#####################################################################################################################*/
  /*#####################################################################################################################*/


  //Useful quantities
  unsigned int dim = _wf.N() + _wf.M() + _wf.N()*_wf.M();  //Total number of variational parameters
  unsigned int index, index_prime;  //Useful index in the blocking procedure
  std::complex <double> blk_variable;  //progressive averages
  std::complex <double> blk_variable_conj;
  std::vector <std::complex <double>> mean_O;  //Monte Carlo averages of each local operator
  std::vector <std::complex <double>> mean_O_conj;

  /* Computes <O_k> , k=1,...,dim */
  /*#################################*/
  /*#################################*/
  //std::cout << " Estimate of the mean of the local operators ... ";
  //std::flush(std::cout);
  mean_O.resize(dim, 0.0);
  mean_O_conj.resize(dim, 0.0);

  //terms related to the visible neurons
  for(unsigned int aj=0; aj<_wf.N(); aj++){
    for(int j=0; j<Nblks; j++){

      blk_variable = 0.0;
      blk_variable_conj = 0.0;
      for(int l=j*blks_size; l<(j+1)*blks_size; l++){

        blk_variable += _Oa[l][aj];
        blk_variable_conj += std::conj(_Oa[l][aj]);

      }
      mean_O[aj] += blk_variable/double(blks_size);
      mean_O_conj[aj] += blk_variable_conj/double(blks_size);

    }
  }

  //terms related to the hidden neurons
  for(unsigned int bk=0; bk<_wf.M(); bk++){
    for(int j=0; j<Nblks; j++){

      blk_variable  = 0.0;
      blk_variable_conj = 0.0;
      for(int l=j*blks_size; l<(j+1)*blks_size; l++) {

        blk_variable += _Ob[l][bk];
        blk_variable_conj += std::conj(_Ob[l][bk]);

      }
      mean_O[_wf.N()+bk] += blk_variable/double(blks_size);
      mean_O_conj[_wf.N()+bk] += blk_variable_conj/double(blks_size);

    }
  }

  //terms related to the visible-hidden neurons interaction
  for(int index_row=0; index_row<_wf.N(); index_row++){
    for(int index_col=0; index_col<_wf.M(); index_col++){

      index = _wf.N() + _wf.M() + _wf.M()*index_row + index_col;
      for(int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        blk_variable_conj = 0.0;
        for(int l=j*blks_size; l<(j+1)*blks_size; l++){

          blk_variable += _Ow[l][index_row][index_col];
          blk_variable_conj += std::conj(_Ow[l][index_row][index_col]);

        }
        mean_O[index] += blk_variable/double(blks_size);
        mean_O_conj[index] += blk_variable_conj/double(blks_size);

      }

    }
  }

  //normalization
  for(auto m : mean_O)
    m /= double(Nblks);
  for(auto m : mean_O_conj)
    m /= double(Nblks);
  //std::cout << "done" << std::endl;
  /*#################################*/
  /*#################################*/

  /* Computes the Covariance */
  /*#########################*/
  /*#########################*/
  //std::cout << " Estimate of the Covariance matrix ... ";
  //std::flush(std::cout);

  for(unsigned int k_aj=0; k_aj<_wf.N(); k_aj++){  //first N-rows of S
    for(unsigned int k_aj_prime=0; k_aj_prime<_wf.N(); k_aj_prime++){  //first N-columns of the selected row of S
      for(unsigned int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
          blk_variable += std::conj(_Oa[l][k_aj])*_Oa[l][k_aj_prime];
        _S(k_aj, k_aj_prime) += blk_variable/double(blks_size) - mean_O_conj[k_aj]*mean_O[k_aj_prime];

      }
    }
    for(unsigned int k_bk_prime=0; k_bk_prime<_wf.M(); k_bk_prime++){  //second M-columns of the selected row of S
      for(unsigned int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
          blk_variable += std::conj(_Oa[l][k_aj])*_Ob[l][k_bk_prime];
        _S(k_aj, _wf.N()+k_bk_prime) += blk_variable/double(blks_size) - mean_O_conj[k_aj]*mean_O[_wf.N()+k_bk_prime];

      }
    }
    for(unsigned int kw_row_prime=0; kw_row_prime<_wf.N(); kw_row_prime++){  //third N*M-columns of the selected row of S
      for(unsigned int kw_col_prime=0; kw_col_prime<_wf.M(); kw_col_prime++){

        index = _wf.N() + _wf.M() + _wf.M()*kw_row_prime + kw_col_prime;
        for(unsigned int j=0; j<Nblks; j++){

          blk_variable = 0.0;
          for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
            blk_variable += std::conj(_Oa[l][k_aj])*_Ow[l][kw_row_prime][kw_col_prime];
          _S(k_aj, index) += blk_variable/double(blks_size) - mean_O_conj[k_aj]*mean_O[index];

        }

      }
    }

  }
  for(unsigned int k_bk=0; k_bk<_wf.M(); k_bk++){  //Second M-rows of S
    for(unsigned int k_aj_prime=0; k_aj_prime<_wf.N(); k_aj_prime++){  //first N-columns of the selected row of S
      for(unsigned int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
          blk_variable += std::conj(_Ob[l][k_bk])*_Oa[l][k_aj_prime];
        _S(_wf.N()+k_bk, k_aj_prime) += blk_variable/double(blks_size) - mean_O_conj[_wf.N()+k_bk]*mean_O[k_aj_prime];

      }
    }
    for(unsigned int k_bk_prime=0; k_bk_prime<_wf.M(); k_bk_prime++){  //second M-columns of the selected row of S
      for(unsigned int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
          blk_variable += std::conj(_Ob[l][k_bk])*_Ob[l][k_bk_prime];
        _S(_wf.N()+k_bk, _wf.N()+k_bk_prime) += blk_variable/double(blks_size) - mean_O_conj[_wf.N()+k_bk]*mean_O[_wf.N()+k_bk_prime];

      }
    }
    for(unsigned int kw_row_prime=0; kw_row_prime<_wf.N(); kw_row_prime++){  //third N*M-columns of the selected row of S
      for(unsigned int kw_col_prime=0; kw_col_prime<_wf.M(); kw_col_prime++){

        index = _wf.N() + _wf.M() + _wf.M()*kw_row_prime + kw_col_prime;
        for(unsigned int j=0; j<Nblks; j++){

          blk_variable = 0.0;
          for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
            blk_variable += std::conj(_Ob[l][k_bk])*_Ow[l][kw_row_prime][kw_col_prime];
          _S(_wf.N()+k_bk, index) += blk_variable/double(blks_size) - mean_O_conj[_wf.N()+k_bk]*mean_O[index];

        }

      }
    }
  }
  for(unsigned int kw_row=0; kw_row<_wf.N(); kw_row++){  //Third N*M-rows of S
    for(unsigned int kw_col=0; kw_col<_wf.M(); kw_col++){

      index  =  _wf.N() + _wf.M() + _wf.M()*kw_row + kw_col;
      for(unsigned int k_aj_prime=0; k_aj_prime<_wf.N(); k_aj_prime++){  //first N-columns of the selected row of S
        for(unsigned int j=0; j<Nblks; j++){

          blk_variable = 0.0;
          for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
            blk_variable += std::conj(_Ow[l][kw_row][kw_col])*_Oa[l][k_aj_prime];
          _S(index, k_aj_prime) += blk_variable/double(blks_size) - mean_O_conj[index]*mean_O[k_aj_prime];

        }
      }
      for(unsigned int k_bk_prime=0; k_bk_prime<_wf.M(); k_bk_prime++){  //second M-columns of the selected row of S
        for(unsigned int j=0; j<Nblks; j++){

          blk_variable = 0.0;
          for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
            blk_variable += std::conj(_Ow[l][kw_row][kw_col])*_Ob[l][k_bk_prime];
          _S(index, _wf.N()+k_bk_prime) += blk_variable/double(blks_size) - mean_O_conj[index]*mean_O[_wf.N()+k_bk_prime];

        }
      }
      for(unsigned int kw_row_prime=0; kw_row_prime<_wf.N(); kw_row_prime++){  //third N*M-columns of the selected row of S
        for(unsigned int kw_col_prime=0; kw_col_prime<_wf.M(); kw_col_prime++){

          index_prime = _wf.N() + _wf.M() + _wf.M()*kw_row_prime + kw_col_prime;
          for(unsigned int j=0; j<Nblks; j++){

            blk_variable = 0.0;
            for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
              blk_variable += std::conj(_Ow[l][kw_row][kw_col])*_Ow[l][kw_row_prime][kw_col_prime];
            _S(index, index_prime) += blk_variable/double(blks_size) - mean_O_conj[index]*mean_O[index];

          }

        }
      }

    }
  }

  //normalization
  _S /= double(Nblks);
  //std::cout << "done" << std::endl;
  /*#########################*/
  /*#########################*/

  /* Compute the Forces */
  /*####################*/
  /*####################*/
  //std::cout << " Estimate of the forces ... ";
  //std::flush(std::cout);
  std::complex <double> mean_E = {E_ave_re, E_ave_im};

  for(unsigned int k_aj=0; k_aj<_wf.N(); k_aj++){  //Forces on the visible neurons
    for(unsigned int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
          blk_variable += _E[l]*std::conj(_Oa[l][k_aj]);
        _f(k_aj) += blk_variable/double(blks_size) - mean_E*mean_O_conj[k_aj];

    }
  }
  for(unsigned int k_bk=0; k_bk<_wf.M(); k_bk++){  //Forces on the hidden neurons
    for(unsigned int j=0; j<Nblks; j++){

        blk_variable = 0.0;
        for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
          blk_variable += _E[l]*std::conj(_Ob[l][k_bk]);
        _f(_wf.N()+k_bk) += blk_variable/double(blks_size) - mean_E*mean_O_conj[_wf.N()+k_bk];

    }
  }
  for(unsigned int kw_row=0; kw_row<_wf.N(); kw_row++){  //Forces on the weights parameters
    for(unsigned int kw_col=0; kw_col<_wf.M(); kw_col++){

      index = _wf.N() + _wf.M() + _wf.M()*kw_row + kw_col;
      for(unsigned int j=0; j<Nblks; j++){

          blk_variable = 0.0;
          for(unsigned int l=j*blks_size; l<(j+1)*blks_size; l++)
            blk_variable += _E[l]*std::conj(_Ow[l][kw_row][kw_col]);
          _f(index) += blk_variable/double(blks_size) - mean_E*mean_O_conj[index];

      }

    }
  }

  //normalization
  _f /= double(Nblks);
  //std::cout << "done" << std::endl << std::endl;
  /*####################*/
  /*####################*/


  /*#####################################################################################################################*/
  /*#####################################################################################################################*/

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: WriteState() {

  for(const auto& spin_value : _state)
    _fileState << std::setw(4) << spin_value;
  _fileState << std::endl;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: WriteWave() {

  //Saves the wave function structure
  _fileWave << _wf.N() << "\n" << _wf.M() << std::endl;
  //Saves on file the visible bias
  for(unsigned int j=0; j<_wf.N(); j++)
    _fileWave << _wf.aj(j) << std::endl;
  //Saves on file the hidden bias
  for(unsigned int k=0; k<_wf.M(); k++)
    _fileWave << _wf.bk(k) << std::endl;
  //Saves on file the weights matrix
  for(unsigned int j=0; j<_wf.N(); j++){
    for(unsigned int k=0; k<_wf.M(); k++)
      _fileWave << _wf.Wjk(j, k) << std::endl;
  }

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: CloseFile() {

  if(_writeState)
    _fileState.close();
  if(_writeWave)
    _fileWave.close();

}


template <class WaveFunc, class Hamiltonian>
bool MC_Sampler <WaveFunc, Hamiltonian> :: RandFlips(std::vector <int>& flipped_site, int Nflips, bool zeroMag) {

  /*
    Random spin flips
    Max 2 spin flips in this implementation
    Different strategies can be implemented

    This function lets you decide whether or not
    to do a single spin-flip move: in particular if it
    returns true, the move is done, otherwise not.

    In case the magnetization of the state is different
    from zero and the spins from flipping are two, it is only
    necessary to ensure that the two randomly selected spins are different;
    when the magnetization is zero, you have to make sure that the two flipped
    randomly selected spins as well as not being the same must also have
    opposite sign, to hold the TotMag = 0.
  */

  flipped_site.resize(Nflips);
  flipped_site[0] = _rnd.Rannyu_INT(0, _Nspin-1);  //Choose a random spin to flip
  if(Nflips==2){

    flipped_site[1] = _rnd.Rannyu_INT(0, _Nspin-1);  //Again a random site selected for the flip
    if(!zeroMag)
      return flipped_site[1] != flipped_site[0];
    else
      return _state[flipped_site[1]] != _state[flipped_site[0]];

  }
  return true;

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: Move(int Nflips) {

  if(RandFlips(_flipped_site, Nflips)){

    //Compute the Acceptance kernel of Metropolis
    //algorithm; std::norm is the squared magnitude
    //of a complex number
    double p_metro = std::norm(_wf.PsiNew_over_PsiOld(_state, _flipped_site));
    if(_rnd.Rannyu() < p_metro){  //Metropolis-Hastings test

      //Update effective angles before changing the states
      _wf.Update_ThetaS(_state, _flipped_site);
      //Move the configuration
      for(const auto& m_flipped : _flipped_site)
        _state[m_flipped] *= -1;
      _accept += 1;

    }

  }

}


template <class WaveFunc, class Hamiltonian>
void MC_Sampler <WaveFunc, Hamiltonian> :: MC_Run(double Nsweeps, int Nblks, double eq_time, int Mfraction, int nflips) {

  /*
    Run the Monte Carlo sampling

      # Nsweeps is the number of Monte Carlo sweeps.
        Definition: a Monte Carlo sweep is the single attempt of
                    M spin-flip moves;
      # eq_time is the fraction of MC total time to be discarded
        during the initial thermalization phase;
      # Mfraction is defined above, i.e. Mfraction = M/N;
      # nflips is the number of random spin flips to be done in
        each spin-flip moves of the total M attempts.
        In this implementation it is automatically set to 1 or 2
        depending on the hamiltonian;
  */

  //Information
  //std::cout << "#Starting Monte Carlo sampling of the RBM variational Quantum State" << std::endl;
  //std::cout << " Number of sweeps to be performed = " << Nsweeps << std::endl;
  //std::cout << " Equilibration time = " << eq_time*Nsweeps << std::endl;

  int Nflips = nflips;
  int M = _Nspin*Mfraction;
  if(Nflips==-1)
    Nflips = _H.MinFlips();
  if(Nflips>2 || Nflips<1){

    std::cerr << " ##Error: in this implementation the number of spin flips should be equal to 1 or 2." << std::endl;
    std::abort();

  }

  //Initialization and Equilibration
  if(eq_time >1 || eq_time<0){

    std::cerr << "#Error: the fraction of MC steps to be used for the equilibration phase must be a real number between 0 and 1." << std::endl;
    std::abort();

  }
  //std::cout << " Initialization" << std::endl;
  Init_State();
  _flipped_site.resize(Nflips);
  _wf.Init_ThetaS(_state);
  _accept = 0;
  _totMoves = 0;

  //std::cout << " Thermalization phase ... ";
  //std::flush(std::cout);
  for(double eq_steps=0; eq_steps<Nsweeps*eq_time; eq_steps++){

    for(int moves=0; moves<M; moves++)
      Move(Nflips);

  }
  //std::cout << "done" << std::endl;

  //Monte Carlo Measurement
  _accept = 0;
  _totMoves = 0;
  //std::cout << " Sampling & Measuring istantaneous values ... ";
  //std::flush(std::cout);
  for(unsigned int step=0; step<Nsweeps; step++){

    for(unsigned int moves=0; moves<M; moves++)
      Move(Nflips);
    if(_writeState)
      WriteState();
    Energy();
    Sx();
    LocalOperators();

  }
  //std::cout << "done" << std::endl;

  //Blocking Estimate & Updating the Quantum state
  Blocking(Nblks);
  Update_wf();
  if(_writeWave)
    WriteWave();
  //std::cout << std::endl;

}


#endif
