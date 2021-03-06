#ifndef __SAMPLER__
#define __SAMPLER__


/*********************************************************************************************************/
/********************************  Variational Monte Carlo Sampler  **************************************/
/*********************************************************************************************************/
/*

  We create a Variational Quantum Monte Carlo (πππ) sampler as a C++ class, which is able to
  optimize a generic ππ‘πππ¨π° ππ§π¬ππ­π³ (a variational quantum state π―πͺπ¬) in order to study a
  generic Lattice Quantum System (πππ).
  The main goal of the sampler is to optimize the parameters that uniquely characterize the π―πͺπ¬
  to obtain the ground state of the given Hamiltonian; once found the ground state, it is
  possible to study the real-time dynamics of the system after performing a quantum quench on a
  certain coupling constant.

  The optimization described above takes place within a stochastic setting, in which the
  procedure leads to the resolution of the following equations of motion for the variational
  parameters πΆ (πππππ Equations of Motion):

            Ξ£β πΌΜβ {πΌβ±Ό, πΌβ} = βπ[πΆ]/βπΌβ±Ό      (πππππ)
            Ξ£β πΌΜβ {πΌβ±Ό, πΌβ} = -πβ’βπ[πΆ]/βπΌβ±Ό   (π-πππππ)

  where the ground state properties are recovered with an imaginaty time evolution

            π β π = -ππ.

  This class is also able to apply the above technique to a non-shadow ansatz, where
  different hypotheses are assumed for the form of the variational wave function.

  NΜ²OΜ²TΜ²EΜ²: we use the pseudo-random numbers generator device by [Percus & Kalos, 1989, NY University].

*/
/*********************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etcβ¦
#include <iomanip>  // <-- std::setw(), std::fixed(), std::setprecision()
#include <fstream>  // <-- std::ifstream, std::ofstream, std::flush()
#include <filesystem>  // <-- is_directory(), exists(), create_directory()
/*
  Use
    #include <experimental/filesystem>
  if you are in @tolab!
*/
#include <complex>  // <-- std::complex<>, .real(), .imag()
#include <armadillo>  // <-- arma::Mat, arma::Col, arma::Row, arma::field
#include "random.h"  // <-- Random
#include "ansatz.cpp"  // <-- WaveFunction
#include "model.cpp"  // <-- SpinHamiltonian


using namespace arma;
using namespace std::__fs::filesystem;  //Use std::experimental::filesystem if you are in @tolab


class VMC_Sampler {

  private:

    //Quantum problem defining variables
    WaveFunction& _vqs;  //The wave function ansatz |Ξ¨(π,πΆ)β©
    SpinHamiltonian& _H;  //The Spin Hamiltonian Δ€
    const unsigned int _Nspin;  //Number of spins in the system

    //Constant data-members
    const std::complex <double> _i;  //The imaginary unit π
    const Mat <double> _I;  //The real identity matrix π

    //Random device
    Random _rnd;

    //Quantum configuration variables |π?β© = |π π πΛβ©
    const unsigned int _Nhidden;  //Number of auxiliary quantum variables
    Mat <int> _configuration;  //Current visible configuration of the system |πβ© = |ππ£ ππ€ β¦ ππ­β©
    Mat <int> _hidden_ket;  //The ket configuration of the hidden variables |πβ© = |π½π£ π½π€ β¦ π½π¬β©
    Mat <int> _hidden_bra;  //The bra configuration of the hidden variables β¨πΛ| = β¨π½Λπ¬ β¦ π½Λπ€ π½Λπ£|
    Mat <int> _flipped_site;  //The new sampled visible configuration |πβΏα΅Κ·β©
    Mat <int> _flipped_ket_site;  //The new sampled ket configuration of the hidden variables |πβΏα΅Κ·β©
    Mat <int> _flipped_bra_site;  //The new sampled bra configuration of the hidden variables β¨πΛβΏα΅Κ·|

    //Monte Carlo moves statistics variables
    unsigned int _N_accepted_visible;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· π πΛβ© accepted along the MCMC
    unsigned int _N_proposed_visible;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· π πΛβ© proposed along the MCMC
    unsigned int _N_accepted_ket;  //Number of new configuration |π?βΏα΅Κ·β© = |π πβΏα΅Κ· πΛβ© accepted along the MCMC
    unsigned int _N_proposed_ket;  //Number of new configuration |π?βΏα΅Κ·β© = |π πβΏα΅Κ· πΛβ© proposed along the MCMC
    unsigned int _N_accepted_bra;  //Number of new configuration |π?βΏα΅Κ·β© = |π π πΛβΏα΅Κ·β© accepted along the MCMC
    unsigned int _N_proposed_bra;  //Number of new configuration |π?βΏα΅Κ·β© = |π π πΛβΏα΅Κ·β© proposed along the MCMC
    unsigned int _N_accepted_equal_site;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· πβΏα΅Κ· πΛβΏα΅Κ·β© with equal-site-spin-flip accepted along the MCMC
    unsigned int _N_proposed_equal_site;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· πβΏα΅Κ· πΛβΏα΅Κ·β© with equal-site-spin-flip proposed along the MCMC
    unsigned int _N_accepted_visible_nn_site;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· π πΛβ© with nearest-neighbors-site-spin-flip accepted along the MCMC
    unsigned int _N_proposed_visible_nn_site;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· π πΛβ© with nearest-neighbors-site-spin-flip proposed along the MCMC
    unsigned int _N_accepted_hidden_nn_site;  //Number of new configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· πβΏα΅Κ· πΛβΏα΅Κ·β© with nearest-neighbors-site-spin-flip accepted along the MCMC
    unsigned int _N_proposed_hidden_nn_site;  //Number of new configuration |π?βΏα΅Κ·β© = |π πβΏα΅Κ· πΛβΏα΅Κ·β© with nearest-neighbors-site-spin-flip proposed along the MCMC

    //Monte Carlo storage variables
    field <Row <std::complex <double>>> _Connections;  //Non-zero matrix elements (i.e. the connections) of the observable operators
    field <field <Mat <int>>> _StatePrime;  //List of configuration |π?'β© associated to each observables connections
    Mat <double> _instReweight;  //Measured the πππ°ππ’π π‘π­π’π§π  ratio ingredients along the MCMC
    Mat <std::complex <double>> _instObs_ket;  //Measured values of quantum observables on the configuration |π πβ©  along the MCMC
    Mat <std::complex <double>> _instObs_bra;  //Measured values of quantum observables on the configuration |π πΛβ© along the MCMC
    Mat <std::complex <double>> _instO_ket;  //Measured local operators π(π,π) along the MCMC
    Mat <std::complex <double>> _instO_bra;  //Measured local operators π(π,πΛ) along the MCMC

    //Simulation options variables
    bool _if_shadow;  //Chooses the shadow or the non-shadow algorithm
    bool _if_hidden_off;  //Chooses to shut down the auxiliary variable in a Shadow vqs
    bool _if_vmc;  //Chooses to make a single simple πππ without parameters optimization
    bool _if_imag_time;  //Chooses imaginary-time dinamics, i.e. πππππ with π = -ππ­
    bool _if_real_time;  //Chooses real-time dynamics
    bool _if_QGT_reg;  //Chooses to regularize the Quantum Geometric Tensor by adding a bias
    bool _if_extra_hidden_sum;  //Increases the sampling of |πβ© and β¨πΛ| during the single measure
    bool _if_restart_from_config;  //Chooses to initialize the initial point of the MCMC from a previously optimized visible configuration |πβ©

    //Simulation parameters of the single πππ step
    unsigned int _Nsweeps;  //Number of Monte Carlo sweeps (i.e. #MC-steps of the single πππ step)
    unsigned int _Nblks;  //Number of blocks to estimate uncertainties
    unsigned int _Neq;  //Number of Monte Carlo equilibration steps to do at the beginning of the single πππ step
    unsigned int _M;  //Number of spin-flips moves to perform in the single sweep
    unsigned int _Nflips;  //Number of spin-flips in each spin-flips move
    unsigned int _Nextra;  //Number of extra MC-steps involving only the hidden sampling
    unsigned int _Nblks_extra;  //Number of blocks in the extra hidden sampling
    double _p_equal_site;  //Probability for the equal site Monte Carlo move
    double _p_visible_nn;  //Probability for the visible nearest neighbor Monte Carlo move
    double _p_hidden_nn;  //Probability for the hidden nearest neighbor Monte Carlo move

    //πππππ variables
    double _delta;  //The value of the integration step πΏπ‘
    double _eps;  //The value of the Quantum Geometric Tensor bias Ξ΅
    unsigned int _fixed_hidden_orientation;  //Bias on the value of all the auxiliary degrees of freedom
    Col <double> _cosII;  //The block averages of the non-zero reweighting ratio part β¨cos[β(π£, π½) - β(π£, π½')]β©β±Όα΅Λ‘α΅
    Col <double> _sinII;  //The block averages of the (theoretically)-zero reweighting ratio part β¨sin[β(π£, π½) - β(π£, π½')]β©β±Όα΅Λ‘α΅
    Col <double> _global_cosII;
    Col <double> _global_sinII;
    field <Col <std::complex <double>>> _Observables;  //The block averages of the quantum observables computed along the MCMC β¨πͺβ©β±Όα΅Λ‘α΅
    field <Col <std::complex <double>>> _O;  //The block averages of the local operators computed along the MCMC β¨πββ©β±Όα΅Λ‘α΅, for k = π£,β¦,nα΅Λ‘α΅Κ°α΅
    field <Col <std::complex <double>>> _global_Observables;
    Col <std::complex <double>> _mean_O;  // β¨β¨πββ©α΅Λ‘α΅β©
    Col <std::complex <double>> _mean_O_star;  // β¨β¨πβββ©α΅Λ‘α΅β©
    Col <double> _mean_O_angled;  // β¨βͺπβ«α΅Λ‘α΅β©
    Col <double> _mean_O_square;  // β¨βπβα΅Λ‘α΅β©
    std::complex <double> _E;  // The standard stochastic average of β¨ββ© (without block averaging)
    Mat <std::complex <double>> _Q;  //The Quantum Geometric Tensor β
    Col <std::complex <double>> _F;  //The energy Gradient π½ acting on πΆ

    //Print options and related files
    bool _write_Move_Statistics;  //Writes the acceptance statistics along the single MCMC
    bool _write_MCMC_Config;  //Writes the sampled |π?β© along the single MCMC
    bool _write_final_Config;  //Writes the last sampled |π?β© of each πππ step
    bool _write_opt_Observables;  //Writes optimized Monte Carlo estimates of quantum observables at the end of each πππ step
    bool _write_block_Observables;  //Writes the observables averages in each block of the MCMC, for each πππ step
    bool _write_opt_Params;  //Writes the optimized set π₯α΅α΅α΅ of the variational wave function at the end of the πππππ
    bool _write_all_Params;  //Writes the set of optimized π₯ of the variational wave function after each πππ step
    bool _write_QGT_matrix;  //Writes the Quantum Geometric Tensor matrix of each πππ step
    bool _write_QGT_cond;  //Writes the condition number of the Quantum Geometric Tensor matrix of each πππ step
    bool _write_QGT_eigen;  //Writes the Quantum Geometric Tensor eigenvalues of each πππ step
    std::ofstream _file_Move_Statistics;
    std::ofstream _file_MCMC_Config;
    std::ofstream _file_final_Config;
    std::ofstream _file_opt_Energy;
    std::ofstream _file_opt_SigmaX;
    std::ofstream _file_opt_SigmaY;
    std::ofstream _file_opt_SigmaZ;
    std::ofstream _file_block_Energy;
    std::ofstream _file_block_SigmaX;
    std::ofstream _file_block_SigmaY;
    std::ofstream _file_block_SigmaZ;
    std::ofstream _file_opt_Params;
    std::ofstream _file_all_Params;
    std::ofstream _file_QGT_matrix;
    std::ofstream _file_QGT_cond;
    std::ofstream _file_QGT_eigen;

  public:

    //Constructor and Destructor
    VMC_Sampler(WaveFunction&, SpinHamiltonian&, int);
    ~VMC_Sampler() {};

    //Access functions
    WaveFunction& vqs() const {return _vqs;}  //Returns the reference to the ansatz wave function
    SpinHamiltonian& H() const {return _H;}  //Returns the reference to the spin Hamiltonian
    unsigned int n_spin() const {return _Nspin;}  //Returns the number of quantum degrees of freedom
    unsigned int n_hidden() const {return _Nhidden;}  //Returns the number of auxiliary degrees of freedom
    std::complex <double> i() const {return _i;}  //Returns the imaginary unit π
    Mat <double> I() const {return _I;}  //Returns the identity matrix π
    Mat <int> visible_configuration() const {return _configuration;}  //Returns the sampled visible configuration |πβ©
    Mat <int> hidden_ket() const {return _hidden_ket;}  //Returns the sampled ket configuration of the hidden variables |πβ©
    Mat <int> hidden_bra() const {return _hidden_bra;}  //Returns the sampled bra configuration of the hidden variables β¨πΛ|
    Mat <int> new_visible_config() const {return _flipped_site;}  //Returns the new sampled visible configuration |πβΏα΅Κ·β©
    Mat <int> new_hidden_ket() const {return _flipped_ket_site;}  //Returns the new sampled ket configuration |πβΏα΅Κ·β©
    Mat <int> new_hidden_bra() const {return _flipped_bra_site;}  //Returns the new sampled bra configuration β¨πΛβΏα΅Κ·|
    void print_configuration() const;  //Prints on standard output the current sampled system configuration |π?β© = |π π πΛβ©
    field <Row <std::complex <double>>> get_connections() const {return _Connections;}  //Returns the list of connections
    field <field <Mat <int>>> all_state_prime() const {return _StatePrime;}  //Returns all the configuration |π?'β© connected to the current sampled configuration |π?β©
    Mat <std::complex <double>> InstObs_ket() const {return _instObs_ket;}  //Returns all the measured values of πͺΛ‘α΅αΆ(π,π) after a single VMC run
    Mat <std::complex <double>> InstObs_bra() const {return _instObs_bra;}  //Returns all the measured values of πͺΛ‘α΅αΆ(π,π') after a single VMC run
    Mat <std::complex <double>> InstO_ket() const {return _instO_ket;}  //Returns all the measured local operators π(π,π) after a single VMC run
    Mat <std::complex <double>> InstO_bra() const {return _instO_bra;}  //Returns all the measured local operators π(π,π') after a single VMC run
    Mat <double> InstNorm() const {return _instReweight;}  //Returns all the measured values of πππ [β(π£,π½)-β(π£,π½')] and π ππ[β(π£,π½)-β(π£,π½')] after a single VMC run
    double delta() const {return _delta;}  //Returns the integration step parameter πΏπ‘ used in the dynamics solver
    double QGT_bias() const {return _eps;}  //Returns the regularization bias of the Quantum Geometric Tensor
    unsigned int hidden_bias() const {return _fixed_hidden_orientation;}  //Returns the orientation bias of the hidden variables
    Col <double> cos() const {return _global_cosII;}
    Col <double> sin() const {return _global_sinII;}
    field <Col <std::complex <double>>> Observables() const {return _global_Observables;}
    Mat <std::complex <double>> QGT() const {return _Q;}  //Returns the Monte Carlo estimate of the QGT
    Col <std::complex <double>> F() const {return _F;}  //Returns the Monte Carlo estimate of the energy gradient
    Col <std::complex <double>> O() const {return _mean_O;}
    Col <std::complex <double>> O_star() const {return _mean_O_star;}
    Col <double> _O_angled() const {return _mean_O_angled;}  //Returns the Monte Carlo estimate of the vector of βͺπββ«
    Col <double> _O_square() const {return _mean_O_square;}  //Returns the Monte Carlo estimate of the vector of βπββ
    std::complex <double> E() const {return _E;}  //Returns the Monte Carlo estimate of the energy β¨ββ©

    //Initialization functions
    void Init_Config(const Mat <int>& initial_visible=Mat <int>(),  //Initializes the quantum configuration |π?β© = |π π πΛβ©
                     const Mat <int>& initial_ket=Mat <int>(),
                     const Mat <int>& initial_bra=Mat <int>(),
                     bool zeroMag=true);
    void ShutDownHidden(unsigned int);  //Shuts down the hidden variables
    void setImagTimeDyn(double delta=0.01);  //Chooses the imaginary-time πππππ algorithm
    void setRealTimeDyn(double delta=0.01);  //Chooses the real-time πππππ algorithm
    void setQGTReg(double eps=0.000001);  //Chooses to regularize the QGT
    void setExtraHiddenSum(unsigned int, unsigned int);  //Chooses to make the MC observables less noisy
    void setRestartFromConfig() {_if_restart_from_config = true;}  //Chooses the restart option at the beginning of the MCMC
    void setStepParameters(unsigned int, unsigned int, unsigned int,           //Sets the Monte Carlo parameters for the single VMC step
                           unsigned int, unsigned int, double, double, double,
                           int);

    //Print options functions
    void setFile_Move_Statistics(std::string, int);
    void setFile_MCMC_Config(std::string, int);
    void setFile_final_Config(std::string, int);
    void setFile_opt_Obs(std::string, int);
    void setFile_block_Obs(std::string, int);
    void setFile_opt_Params(std::string, int);
    void setFile_all_Params(std::string, int);
    void setFile_QGT_matrix(std::string, int);
    void setFile_QGT_cond(std::string, int);
    void setFile_QGT_eigen(std::string, int);
    void Write_Move_Statistics(unsigned int, MPI_Comm);
    void Write_MCMC_Config(unsigned int, int);
    void Write_final_Config(unsigned int);
    void Write_opt_Params(int);
    void Write_all_Params(unsigned int, int);
    void Write_QGT_matrix(unsigned int);
    void Write_QGT_cond(unsigned int);
    void Write_QGT_eigen(unsigned int);
    void CloseFile(int);
    void Finalize(int);

    //Measurement functions
    void Reset();
    Col <double> average_in_blocks(const Row <double>&) const;
    Col <std::complex <double>> average_in_blocks(const Row <std::complex <double>>&) const;
    Col <double> Shadow_average_in_blocks(const Row <std::complex <double>>&, const Row <std::complex <double>>&) const;
    Col <double> Shadow_angled_average_in_blocks(const Row <std::complex <double>>&, const Row <std::complex <double>>&) const;
    Col <double> Shadow_square_average_in_blocks(const Row <std::complex <double>>&, const Row <std::complex <double>>&) const;
    void compute_Reweighting_ratio(MPI_Comm);
    Col <double> compute_errorbar(const Col <double>&) const;
    Col <std::complex <double>> compute_errorbar(const Col <std::complex <double>>&) const;
    Col <double> compute_progressive_averages(const Col <double>&) const;
    Col <std::complex <double>> compute_progressive_averages(const Col <std::complex <double>>&) const;
    void compute_Quantum_observables(MPI_Comm);
    void compute_O();
    void compute_QGTandGrad(MPI_Comm);
    void is_asymmetric(const Mat <double>&) const;  //Check the anti-symmetric properties of an Armadillo matrix
    void QGT_Check(int);  //Checks symmetry properties of the Quantum Geometric Tensor
    void Measure();  //Measurement of the istantaneous observables along a single VMC run
    void Estimate(MPI_Comm);  //Monte Carlo estimates of the quantum observable averages
    void Write_Quantum_properties(unsigned int, int);  //Write on appropriate files all the required system quantum properties

    //Monte Carlo moves
    bool RandFlips_visible(Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move for the visibles variable only
    bool RandFlips_hidden(Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move for the hidden variables (ket or bra only)
    bool RandFlips_visible_nn_site(Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move on two visible nearest neighbors lattice site
    bool RandFlips_hidden_nn_site(Mat <int>&, Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move on two hidden nearest neighbors lattice site
    void Move_visible(unsigned int Nflips=1);  //Samples a new system configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· π πΛβ©
    void Move_ket(unsigned int Nflips=1);  //Samples a new system configuration |π?βΏα΅Κ·β© = |π πβΏα΅Κ· πΛβ©
    void Move_bra(unsigned int Nflips=1);  //Samples a new system configuration |π?βΏα΅Κ·β© = |π π πΛβΏα΅Κ·β©
    void Move_equal_site(unsigned int Nflips=1);  //Samples a new system configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· πβΏα΅Κ· πΛβΏα΅Κ·β© with equal-site-spin-flip
    void Move_visible_nn_site(unsigned int Nflips=1);  //Samples a new system configuration |π?βΏα΅Κ·β© = |πβΏα΅Κ· π πΛβ© with nearest-neighbors-site-spin-flip
    void Move_hidden_nn_site(unsigned int Nflips=1);  //Samples a new system configuration |π?βΏα΅Κ·β© = |π πβΏα΅Κ· πΛβΏα΅Κ·β© with nearest-neighbors-site-spin-flip
    void Move();  //Samples a new system configuration

    //Sampling functions
    void Make_Sweep();  //Adds a point in the Monte Carlo Markov Chain
    void Reset_Moves_Statistics();  //Resets the Monte Carlo moves statistics variables
    void VMC_Step(MPI_Comm);  //Performs a single VMC step

    //ODE Integrators
    void Euler(MPI_Comm);  //Updates the variational parameters with the Euler integration method
    void Heun(MPI_Comm);   //Updates the variational parameters with the Heun integration method
    void RK4(MPI_Comm);    //Updates the variational parameters with the fourth order Runge Kutta method

};


/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
VMC_Sampler :: VMC_Sampler(WaveFunction& wave, SpinHamiltonian& hamiltonian, int rank)
             : _vqs(wave), _H(hamiltonian), _Nspin(wave.n_visible()), _i(_H.i()),
               _I(eye(_vqs.n_alpha(), _vqs.n_alpha())), _Nhidden(wave.n_visible() * wave.density()) {

  //Information
  if(rank == 0){

    std::cout << "#Define the πππ sampler of the variational quantum state |Ξ¨(π, πΆ)β©." << std::endl;
    std::cout << " The sampler is defined on a " << _vqs.type_of_ansatz() << " architecture designed for Lattice Quantum Systems." << std::endl;

  }

  /*#######################################################*/
  //  Creates and initializes the Random Number Generator
  //  Each process involved in the parallelization of
  //  the executable code reads a different pair of
  //  numbers from the Primes file, according to its rank.
  /*#######################################################*/
  if(rank == 0)
    std::cout << " Create and initialize the random number generator." << std::endl;
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes");
  if(Primes.is_open()){

    for(unsigned int p = 0; p <= rank; p++)
      Primes >> p1 >> p2;

  }
  else{

    std::cerr << " ##FileError: Unable to open Primes." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }
  Primes.close();
  std::ifstream input("./input_random_device/seed2.in");
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
    if(rank == 0)
      std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed2.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Sets the simulation option variables
  if(_vqs.type_of_ansatz() == "Shadow")
    _if_shadow = true;
  else
    _if_shadow = false;
  _if_hidden_off = false;
  _if_vmc = true;  //Default algorithm β simple πππ
  _if_imag_time = false;
  _if_real_time = false;
  _if_QGT_reg = false;
  _if_extra_hidden_sum = false;
  _if_restart_from_config = false;

  //Sets the output file options
  _write_Move_Statistics = false;
  _write_MCMC_Config = false;
  _write_final_Config = false;
  _write_opt_Observables = false;
  _write_block_Observables = false;
  _write_opt_Params = false;
  _write_all_Params = false;
  _write_QGT_matrix = false;
  _write_QGT_cond = false;
  _write_QGT_eigen = false;

  //Data-members initialization
  _N_accepted_visible = 0;
  _N_proposed_visible = 0;
  _N_accepted_ket = 0;
  _N_proposed_ket = 0;
  _N_accepted_bra = 0;
  _N_proposed_bra = 0;
  _N_accepted_equal_site = 0;
  _N_proposed_equal_site = 0;
  _N_accepted_visible_nn_site = 0;
  _N_proposed_visible_nn_site = 0;
  _N_accepted_hidden_nn_site = 0;
  _N_proposed_hidden_nn_site = 0;
  _eps = 0.0;
  _Nextra = 0;
  _Nblks_extra = 0;

  if(rank == 0)
    std::cout << " πππ sampler correctly initialized." << std::endl;

}


void VMC_Sampler :: print_configuration() const {

  std::cout << "\n=====================================" << std::endl;
  std::cout << "Current configuration |π?β© = |π π πΛβ©" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "|πβ©      = ";
  _configuration.print();
  std::cout << "|πβ©      = ";
  _hidden_ket.print();
  std::cout << "β¨πΛ|     = ";
  _hidden_bra.print();

}


void VMC_Sampler :: Init_Config(const Mat <int>& initial_visible, const Mat <int>& initial_ket, const Mat <int>& initial_bra, bool zeroMag) {

  /*##############################################################################################*/
  //  Initializes the starting point of the MCMC, using the computational basis of ΟΜαΆ» eigenstates
  //
  //        ΟΜαΆ»|+1β© = +|+1β©
  //        ΟΜαΆ»|-1β© = -|-1β©.
  //
  //  We give the possibility to randomly choose spin up or down for each lattice site
  //  by using the conditional ternary operator
  //
  //        condition ? result1 : result2
  //
  //  or to initialize the configuration by providing an acceptable π’π§π’π­π’ππ₯_ππ¨π§ππ’π  for the
  //  variables. Hidden variables are randomly initialized in both cases.
  //  If the boolean data-member π’π_ππππππ_π¨ππ is true, the hidden variables are all initialized
  //  and fixed to a certain constant (ππ’π±ππ_π‘π’ππππ§_π¨π«π’ππ§π­ππ­π’π¨π§), that is they are turned off in
  //  order to make the Shadow ansatz a simple ansatz deprived of the auxiliary variables.
  //  Beware that this is not equivalent to knowing how to analytically integrate the hidden
  //  variables!
  //  If π³ππ«π¨πππ  is true the initial physical configuration |πβ© is prepared with
  //  zero total magnetization.
  //
  //  So, this function initializes the generic configuration to sample along the Markov Chain
  //
  //        |π?β© = |π, π‘, π‘Λβ©.
  //
  //  As concerns the configuration of the hidden variables, we do not make any request with
  //  respect to its magnetization, being non-physical variables.
  /*##############################################################################################*/

  //Initializes the configuration depending on |π²|
  if(_H.dimensionality() == 1){  //π² Ο΅ β€α΅, π½ = π

    if(!_if_restart_from_config)
      _configuration.set_size(1, _Nspin);
    else{

      _configuration = initial_visible;
      _hidden_ket = initial_ket;
      _hidden_bra = initial_bra;

    }
    if(_if_shadow == true && _hidden_ket.is_empty() == true)
      _hidden_ket.set_size(1, _Nhidden);
    if(_if_shadow == true && _hidden_bra.is_empty() == true)
      _hidden_bra.set_size(1, _Nhidden);

  }
  else{  //π² Ο΅ β€α΅, π½ = π

    /*
      .............
      .............
      .............
    */

  }

  //Randomly chooses spin up or spin down in |πβ©
  if(!_if_restart_from_config){

    for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

      for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
        _configuration(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

    }
    //Performs a check on the magnetization
    if(zeroMag){  //Default case

      if(!_Nspin%2){

        std::cerr << " ##SizeError: Cannot initialize a random spin configuration with zero magnetization for an odd number of spins." << std::endl;
        std::cerr << "   Failed to initialize the starting point of the Markov Chain." << std::endl;
        std::abort();

      }
      int tempMag = 1;
      while(tempMag != 0){

        tempMag = 0;
        for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
            tempMag += _configuration(j_row, j_col);

        }
        if(tempMag > 0){

          int rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows-1);  //Select a random spin-UP
          int rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols-1);
          while(_configuration(rs_row, rs_col) < 0){

            rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows-1);
            rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols-1);

          }
          _configuration(rs_row, rs_col) = -1;  //Flip that spin-UP in order to decrease the positive magnetization
          tempMag -= 1;

        }
        else if(tempMag < 0){

          int rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows-1);  //Select a random spin-DOWN
          int rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols-1);
          while(_configuration(rs_row, rs_col) > 0){

            rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows-1);
            rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols-1);

          }
          _configuration(rs_row, rs_col) = 1;  //Flip that spin-DOWN in order to increase the negative magnetization
          tempMag += 1;

        }

      }

    }

  }

  //Initializes |π‘β© and β¨π‘Λ| randomly
  if(_if_shadow){

    if(_if_hidden_off){

      _hidden_ket.fill(_fixed_hidden_orientation);
      _hidden_bra.fill(_fixed_hidden_orientation);

    }
    else{

      if(initial_ket.is_empty()){

        //Randomly chooses spin up or spin down
        for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
            _hidden_ket(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

        }

      }
      if(initial_bra.is_empty()){

        //Randomly chooses spin up or spin down
        for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
            _hidden_bra(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

        }

      }

    }

  }

  //Initializes the variational quantum state
  _vqs.Init_on_Config(_configuration);

}


void VMC_Sampler :: ShutDownHidden(unsigned int orientation_bias) {

  _if_hidden_off = true;
  _fixed_hidden_orientation = orientation_bias;

}


void VMC_Sampler :: setImagTimeDyn(double delta){

  /*#############################################################*/
  //  Allows to update the variational parameters by integration
  //  (with an ODE integrator) of the equation of motion in
  //  imaginary time
  //
  //        π β π = -ππ
  //
  //  and using an integration step parameter πΏπ‘.
  /*#############################################################*/

  _if_vmc = false;
  _if_imag_time = true;
  _if_real_time = false;
  _delta = delta;

}


void VMC_Sampler :: setRealTimeDyn(double delta) {

  /*#############################################################*/
  //  Allows to update the variational parameters by integration
  //  (with an ODE integrator) of the equation of motion in
  //  real time t and using an integration step parameter πΏπ‘.
  /*#############################################################*/

  _if_vmc = false;
  _if_imag_time = false;
  _if_real_time = true;
  _delta = delta;

}


void VMC_Sampler :: setQGTReg(double epsilon) {

  /*##############################################*/
  //  Adds a bias to the Quantum Geometric Tensor
  //
  //        β β β + πβ’π  (π?π½πΆπΉβ΄π)
  //        π β π + πβ’π  (πβ΄π-π?π½πΆπΉβ΄π)
  //
  //  in order to avoid inversion problems in the
  //  integration of the equations of motion.
  //  ππ©π¬π’π₯π¨π§ is the bias strength.
  /*##############################################*/

  _if_QGT_reg = true;
  _eps = epsilon;

}


void VMC_Sampler :: setExtraHiddenSum(unsigned int Nextra, unsigned int Nblks) {

  _if_extra_hidden_sum = true;
  _Nextra = Nextra;
  _Nblks_extra = Nblks;

}


void VMC_Sampler :: setStepParameters(unsigned int Nsweeps, unsigned int Nblks, unsigned int Neq, unsigned int M,
                                      unsigned int Nflips, double p_equal_site, double p_visible_nn, double p_hidden_nn,
                                      int rank) {

  _Nsweeps = Nsweeps;
  _Nblks = Nblks;
  _Neq = Neq;
  _M = M;
  _Nflips = Nflips;
  _p_equal_site = p_equal_site;
  _p_visible_nn = p_visible_nn;
  _p_hidden_nn = p_hidden_nn;

  if(rank == 0){

    std::cout << " Parameters of the simulation in each nodes of the communicator:" << std::endl;
    std::cout << " \tNsweeps in the single π½π΄πͺ step = " << _Nsweeps << std::endl;
    std::cout << " \tNblks in the single π½π΄πͺ step = " << _Nblks << std::endl;
    std::cout << " \tEquilibration steps in the single π½π΄πͺ step = " << _Neq << std::endl;
    std::cout << " \tNumber of spin-flips moves in the single π΄πͺ sweep = " << _M << std::endl;
    std::cout << " \tNumber of spin-flip in the single spin-flips move = " << _Nflips << std::endl;
    std::cout << " \tP(equal site) = " << _p_equal_site * 100.0 << " %" << std::endl;
    std::cout << " \tP(n.n. visible) = " << _p_visible_nn * 100.0 << " %" << std::endl;
    std::cout << " \tP(n.n. hidden) = " << _p_hidden_nn * 100.0 << " %" << std::endl;
    if(_if_extra_hidden_sum){

      std::cout << " \tNumber of extra hidden sampling in each instantaneous measurement = "  << _Nextra << std::endl;
      std::cout << " \tNumber of block for the extra hidden sampling = " << _Nblks_extra << std::endl;

    }
    std::cout << " \tIntegration step parameter = " << _delta << std::endl;
    std::cout << " \tQGT bias = " << _eps << std::endl << std::endl;

  }

}


void VMC_Sampler :: setFile_Move_Statistics(std::string info, int rank) {

  _write_Move_Statistics = true;
  if(rank == 0){

    _file_Move_Statistics.open("Move_Statistics_" + info + ".dat");
    if(!_file_Move_Statistics.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ Move_Statistics_" << info << ".dat βΊβΊ for writing the acceptance statistics at the end of the single πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the acceptance statistics of the moves at the end of the single πππ step on file βΉβΉ Move_Statistics_" << info << ".dat βΊβΊ." << std::endl;
    _file_Move_Statistics << "###########################################################################################################\n";
    _file_Move_Statistics << "# Column Legend\n";
    _file_Move_Statistics << "#\n";
    _file_Move_Statistics << "#Β   1st: the πππ step identifier\n";
    _file_Move_Statistics << "#Β   2nd: the sampling acceptance probability (%) of |πβ©\n";
    _file_Move_Statistics << "#Β   3rd: the sampling acceptance probability (%) of |πβ©\n";
    _file_Move_Statistics << "#   4th: the sampling acceptance probability (%) of β¨πΛ|\n";
    _file_Move_Statistics << "#Β   5th: the sampling acceptance probability (%) of |π π πΛβ© moved on equal sites\n";
    _file_Move_Statistics << "#Β   6th: the sampling acceptance probability (%) of |πβ© moved on nearest-neighbor sites\n";
    _file_Move_Statistics << "#Β   7th: the sampling acceptance probability (%) of |πβ© and β¨πΛ| moved on generally nearest-neighbor sites\n";
    _file_Move_Statistics << "###########################################################################################################\n";

  }

}


void VMC_Sampler :: setFile_MCMC_Config(std::string info, int rank) {

  _write_MCMC_Config = true;
  if(rank == 0){

    //Creates the output directory by checking if CONFIG folder already exists
    if(!is_directory("./CONFIG") || !exists("./CONFIG")) create_directory("./CONFIG");

    _file_MCMC_Config.open("./CONFIG/MCMC_config_" + info + ".dat");
    if(!_file_MCMC_Config.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ MCMC_config_" << info << ".dat βΊβΊ for writing the sampled configurations along a single MCMC." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the sampled configurations along a single MCMC on file βΉβΉ MCMC_config_" << info << ".dat βΊβΊ." << std::endl;
    _file_MCMC_Config << "####################################################\n";
    _file_MCMC_Config << "# Column Legend\n";
    _file_MCMC_Config << "#\n";
    _file_MCMC_Config << "#   1st: the ππ-step identifier\n";
    _file_MCMC_Config << "#Β   2nd: the sampled quantum configuration |π π πΛβ©\n";
    _file_MCMC_Config << "####################################################\n";

  }

}


void VMC_Sampler :: setFile_final_Config(std::string info, int rank) {

  _write_final_Config = true;

  //Creates the output directory by checking if CONFIG folder already exists
  if(rank == 0)
    if(!is_directory("./CONFIG") || !exists("./CONFIG")) create_directory("./CONFIG");

  _file_final_Config.open("./CONFIG/final_config_" + info + "_node_" + std::to_string(rank) + ".dat");
  if(!_file_final_Config.good()){

    std::cerr << " ##FileError: Cannot open the file βΉβΉ final_config_" << info << "node_" << rank << ".dat βΊβΊ for writing the final configurations at the end of each πππ step." << std::endl;
    std::abort();

  }
  else{

    if(rank == 0)
      std::cout << " Saving the final configurations sampled at the end of each πππ step on file βΉβΉ final_config_" << info << "_node_*.dat βΊβΊ." << std::endl;

  }
  _file_final_Config << "####################################################\n";
  _file_final_Config << "# Column Legend\n";
  _file_final_Config << "#\n";
  _file_final_Config << "#   1st: the πππ-step identifier\n";
  _file_final_Config << "#   2nd: the sampled quantum configuration |π π πΛβ©\n";
  _file_final_Config << "####################################################\n";

}


void VMC_Sampler :: setFile_opt_Obs(std::string info, int rank) {

  _write_opt_Observables = true;
  if(rank == 0){

    _file_opt_Energy.open("opt_energy_" + info + ".dat");
    _file_opt_SigmaX.open("opt_sigmaX_" + info + ".dat");
    _file_opt_SigmaY.open("opt_sigmaY_" + info + ".dat");
    _file_opt_SigmaZ.open("opt_sigmaZ_" + info + ".dat");
    if(!_file_opt_Energy.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ opt_energy_" << info << ".dat βΊβΊ for writing E(π,πΆ ) after each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving E(π,πΆ) after each πππ step on file βΉβΉ opt_energy_" << info << ".dat βΊβΊ." << std::endl;
    if(!_file_opt_SigmaX.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ opt_sigmaX_" << info << ".dat βΊβΊ for writing ΟΛ£(π,πΆ) after each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving ΟΛ£(π,πΆ) after each πππ step on file βΉβΉ opt_sigmaX_" << info << ".dat βΊβΊ." << std::endl;
    if(!_file_opt_SigmaY.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ opt_sigmaY_" << info << ".dat βΊβΊ for writing ΟΚΈ(π,πΆ) after each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving ΟΚΈ(π,πΆ) after each πππ step on file βΉβΉ opt_sigmaY_" << info << ".dat βΊβΊ." << std::endl;
    if(!_file_opt_SigmaZ.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ opt_sigmaZ_" << info << ".dat βΊβΊ for writing ΟαΆ»(π,πΆ  ) after each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving ΟαΆ»(π,πΆ) after each πππ step on file βΉβΉ opt_sigmaZ_" << info << ".dat βΊβΊ." << std::endl;

    _file_opt_Energy << "###########################################\n";
    _file_opt_Energy << "# Column Legend\n";
    _file_opt_Energy << "#\n";
    _file_opt_Energy << "#   1st:  the πππ-step identifier\n";
    _file_opt_Energy << "#   2nd:  progressive β¨ππππ°π°β©π\n";
    _file_opt_Energy << "#   3rd:  progressive π[β¨ππππ°π°β©π]\n";
    _file_opt_Energy << "#   4th:  progressive β¨ππππ°π°β©π\n";
    _file_opt_Energy << "#   5rd:  progressive π[β¨ππππ°π°β©π]\n";
    _file_opt_Energy << "#   6th:  progressive β¨π¬α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_opt_Energy << "#Β   7th:  progressive π[β¨π¬α΄Ώ(π,πΆ)β©_πΉ]\n";
    _file_opt_Energy << "#   8th:  progressive β¨π¬α΄΅(π,πΆ)β©_πΉ\n";
    _file_opt_Energy << "#   9th:  progressive π[β¨π¬α΄΅(π,πΆ)β©_πΉ]\n";
    _file_opt_Energy << "#   10th: standard β¨π¬α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_opt_Energy << "#   11th: standard β¨π¬α΄΅(π,πΆ)β©_πΉ\n";
    _file_opt_Energy << "###########################################\n";

    _file_opt_SigmaX << "###########################################\n";
    _file_opt_SigmaX << "# Column Legend\n";
    _file_opt_SigmaX << "#\n";
    _file_opt_SigmaX << "#   1st: the πππ-step identifier\n";
    _file_opt_SigmaX << "#   2nd: progressive β¨ΟΛ£α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_opt_SigmaX << "#   3rd: progressive π[β¨ΟΛ£α΄Ώ(π,πΆ)β©_πΉ]\n";
    _file_opt_SigmaX << "#   4th: progressive β¨ΟΛ£α΄΅(π,πΆ)β©_πΉ\n";
    _file_opt_SigmaX << "#   5th: progressive π[β¨ΟΛ£α΄΅(π,πΆ)β©_πΉ]\n";
    _file_opt_SigmaX << "###########################################\n";

    _file_opt_SigmaY << "###########################################\n";
    _file_opt_SigmaY << "# Column Legend\n";
    _file_opt_SigmaY << "#\n";
    _file_opt_SigmaY << "#   1st: the πππ-step identifier\n";
    _file_opt_SigmaY << "#   2nd: progressive β¨ΟΚΈα΄Ώ(π,πΆ)β©_πΉ\n";
    _file_opt_SigmaY << "#   3rd: progressive π[β¨ΟΚΈα΄Ώ(π,πΆ)β©_πΉ]\n";
    _file_opt_SigmaY << "#   4th: progressive β¨ΟΚΈα΄΅(π,πΆ)β©_πΉ\n";
    _file_opt_SigmaY << "#   5th: progressive π[β¨ΟΚΈα΄΅(π,πΆ)β©_πΉ]\n";
    _file_opt_SigmaY << "###########################################\n";

    _file_opt_SigmaZ << "###########################################\n";
    _file_opt_SigmaZ << "# Column Legend\n";
    _file_opt_SigmaZ << "#\n";
    _file_opt_SigmaZ << "#   1st: the πππ-step identifier\n";
    _file_opt_SigmaZ << "#   2nd: progressive β¨ΟαΆ»α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_opt_SigmaZ << "#   3rd: progressive π[β¨ΟαΆ»α΄Ώ(π,πΆ)β©_πΉ ]\n";
    _file_opt_SigmaZ << "#   4th: progressive β¨ΟαΆ»α΄΅(π,πΆ)β©_πΉ\n";
    _file_opt_SigmaZ << "#   5th: progressive π[β¨ΟαΆ»α΄΅(π,πΆ)β©_πΉ]\n";
    _file_opt_SigmaZ << "###########################################\n";

  }

}


void VMC_Sampler :: setFile_block_Obs(std::string info, int rank) {

  _write_block_Observables = true;
  if(rank == 0){

    _file_block_Energy.open("block_energy_" + info + ".dat");
    _file_block_SigmaX.open("block_sigmaX_" + info + ".dat");
    _file_block_SigmaY.open("block_sigmaY_" + info + ".dat");
    _file_block_SigmaZ.open("block_sigmaZ_" + info + ".dat");
    if(!_file_block_Energy.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ block_energy_" << info << ".dat βΊβΊ for writing all the block averages of E(π,πΆ ) during each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the block averages of E(π,πΆ) during each πππ step on file βΉβΉ block_energy_" << info << ".dat βΊβΊ." << std::endl;
    if(!_file_block_SigmaX.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ block_sigmaX_" << info << ".dat βΊβΊ for writing all the block averages of ΟΛ£(π,πΆ) during each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the block averages of ΟΛ£(π,πΆ) during each πππ step on file βΉβΉ block_sigmaX_" << info << ".dat βΊβΊ." << std::endl;
    if(!_file_block_SigmaY.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ block_sigmaY_" << info << ".dat βΊβΊ for writing all the block averages of ΟΚΈ(π,πΆ) during each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the block averages of ΟΚΈ(π,πΆ) during each πππ step on file βΉβΉ block_sigmaY_" << info << ".dat βΊβΊ." << std::endl;
    if(!_file_block_SigmaZ.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ block_sigmaZ_" << info << ".dat βΊβΊ for writing all the block averages of ΟαΆ»(π,πΆ) during each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the block averages of ΟαΆ»(π,πΆ) during each πππ step on file βΉβΉ block_sigmaZ_" << info << ".dat βΊβΊ." << std::endl;

    _file_block_Energy << "############################################\n";
    _file_block_Energy << "# Column Legend\n";
    _file_block_Energy << "#\n";
    _file_block_Energy << "#   1st:  the πππ-step identifier\n";
    _file_block_Energy << "#   2nd:  the ππ-block identifier\n";
    _file_block_Energy << "#   3rd:  β¨ππππ°π°β©Κ²π in block j\n";
    _file_block_Energy << "#   4th:  β¨ππππ°π°β©Κ²π in block j\n";
    _file_block_Energy << "#   5th:  β¨π¬α΄Ώ(π,πΆ)β©Κ²π in block j\n";
    _file_block_Energy << "#   6th:  progressive β¨ππππ°π°β©π\n";
    _file_block_Energy << "#   7th:  progressive π[β¨ππππ°π°β©π]\n";
    _file_block_Energy << "#Β   8th:  progressive β¨ππππ°π°β©π\n";
    _file_block_Energy << "#   9th:  progressive π[β¨ππππ°π°β©π]\n";
    _file_block_Energy << "#Β   10th:  progressive β¨π¬α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_block_Energy << "#Β   11th:  progressive π[β¨π¬α΄Ώ(π,πΆ)β©_πΉ ]\n";
    _file_block_Energy << "#Β   12th:  progressive β¨π¬α΄΅(π,πΆ)β©_πΉ\n";
    _file_block_Energy << "#Β   13th: progressive π[β¨π¬α΄΅(π,πΆ)β©_πΉ]\n";
    _file_block_Energy << "############################################\n";

    _file_block_SigmaX << "############################################\n";
    _file_block_SigmaX << "# Column Legend\n";
    _file_block_SigmaX << "#\n";
    _file_block_SigmaX << "#Β   1st: the πππ-step identifier\n";
    _file_block_SigmaX << "#Β   2nd: the ππ-block identifier\n";
    _file_block_SigmaX << "#Β   3rd: progressive β¨ΟΛ£α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_block_SigmaX << "#Β   4th: progressive π[β¨ΟΛ£α΄Ώ(π,πΆ)β©_πΉ ]\n";
    _file_block_SigmaX << "#Β   5th: progressive β¨ΟΛ£α΄΅(π,πΆ)β©_πΉ\n";
    _file_block_SigmaX << "#Β   6th: progressive π[β¨ΟΛ£α΄΅(π,πΆ)β©_πΉ]\n";
    _file_block_SigmaX << "############################################\n";

    _file_block_SigmaY << "############################################\n";
    _file_block_SigmaY << "# Column Legend\n";
    _file_block_SigmaY << "#\n";
    _file_block_SigmaY << "#   1st: the πππ-step identifier\n";
    _file_block_SigmaY << "#Β   2nd: the ππ-block identifier\n";
    _file_block_SigmaY << "#Β   3rd: progressive β¨ΟΚΈα΄Ώ(π,πΆ)β©_πΉ\n";
    _file_block_SigmaY << "#Β   4th: progressive π[β¨ΟΚΈα΄Ώ(π,πΆ)β©_πΉ ]\n";
    _file_block_SigmaY << "#Β   5th: progressive β¨ΟΚΈα΄΅(π,πΆ)β©_πΉ\n";
    _file_block_SigmaY << "#Β   6th: progressive π[β¨ΟΚΈα΄΅(π,πΆ)β©_πΉ]\n";
    _file_block_SigmaY << "############################################\n";

    _file_block_SigmaZ << "############################################\n";
    _file_block_SigmaZ << "# Column Legend\n";
    _file_block_SigmaZ << "#\n";
    _file_block_SigmaZ << "#Β   1st: the πππ-step identifier\n";
    _file_block_SigmaZ << "#   2nd: the ππ-block identifier\n";
    _file_block_SigmaZ << "#Β   3rd: progressive β¨ΟαΆ»α΄Ώ(π,πΆ)β©_πΉ\n";
    _file_block_SigmaZ << "#Β   4th: progressive π[β¨ΟαΆ»α΄Ώ(π,πΆ)β©_πΉ ]\n";
    _file_block_SigmaZ << "#Β   5th: progressive β¨ΟαΆ»α΄΅(π,πΆ)β©_πΉ\n";
    _file_block_SigmaZ << "#Β   6th: progressive π[β¨ΟαΆ»α΄΅(π,πΆ)β©_πΉ]\n";
    _file_block_SigmaZ << "############################################\n";

  }

}


void VMC_Sampler :: setFile_opt_Params(std::string info, int rank) {

  _write_opt_Params = true;
  if(rank == 0){

    _file_opt_Params.open("optimized_parameters_" + info + ".wf");
    if(!_file_opt_Params.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ optimized_" << info << ".wf βΊβΊ for writing the optimized set of variational parameters π₯." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the optimized set of variational parameters π₯ on file βΉβΉ optimized_" << info << ".wf βΊβΊ." << std::endl;

    /*
    _file_opt_Params << "#####################################\n";
    _file_opt_Params << "# Column Legend\n";
    _file_opt_Params << "#\n";
    _file_opt_Params << "#Β   1st: π±α΄Ώ\n";
    _file_opt_Params << "#Β   2nd: π±α΄΅\n";
    _file_opt_Params << "#####################################\n";
    */

  }

}


void VMC_Sampler :: setFile_all_Params(std::string info, int rank) {

  _write_all_Params = true;
  if(rank == 0){

    _file_all_Params.open("variational_manifold_" + info + ".wf");
    if(!_file_all_Params.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ variational_manifold_" << info << ".wf βΊβΊ for writing the set of variational parameters π₯ at the end of each πππ step." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the set of variational parameters π₯ at the end of each πππ step on file βΉβΉ variational_manifold_" << info << ".wf βΊβΊ." << std::endl;

    _file_all_Params << "#####################################\n";
    _file_all_Params << "# Column Legend\n";
    _file_all_Params << "#\n";
    _file_all_Params << "#   1st: the πππ-step identifier\n";
    _file_all_Params << "#Β   2nd: π±α΄Ώ\n";
    _file_all_Params << "#Β   3rd: π±α΄΅\n";
    _file_all_Params << "#####################################\n";

  }

}


void VMC_Sampler :: setFile_QGT_matrix(std::string info, int rank) {

  _write_QGT_matrix = true;
  if(rank == 0){

    _file_QGT_matrix.open("qgt_matrix_" + info + ".dat");
    if(!_file_QGT_matrix.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ qgt_matrix_" << info << ".dat βΊβΊ for writing the Quantum Geometric Tensor." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the QGT after each πππ step on file βΉβΉ qgt_matrix_" << info << ".dat βΊβΊ." << std::endl;

    _file_QGT_matrix << "#######################################\n";
    _file_QGT_matrix << "# Column Legend\n";
    _file_QGT_matrix << "#\n";
    _file_QGT_matrix << "#   1st: the πππ-step identifier\n";
    _file_QGT_matrix << "#Β   2nd: the Quantum Geometric Tensor\n";
    _file_QGT_matrix << "#######################################\n";

  }

}


void VMC_Sampler :: setFile_QGT_cond(std::string info, int rank) {

  _write_QGT_cond = true;
  if(rank == 0){

    _file_QGT_cond.open("qgt_cond_" + info + ".dat");
    if(!_file_QGT_cond.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ qgt_cond_" << info << ".dat βΊβΊ for writing the Quantum Geometric Tensor condition number." << std::endl;
      std::abort();

    }
    else
       std::cout << " Saving the QGT condition number after each πππ step on file βΉβΉ qgt_cond_" << info << ".dat βΊβΊ." << std::endl;

    _file_QGT_cond << "###########################################################################\n";
    _file_QGT_cond << "# Column Legend\n";
    _file_QGT_cond << "#\n";
    _file_QGT_cond << "#   1st: the πππ-step identifier\n";
    _file_QGT_cond << "#Β   2nd: the QGT condition number (real part) (no regularization)\n";
    _file_QGT_cond << "#Β   3rd: the QGT condition number (imaginary part) (no regularization)\n";
    _file_QGT_cond << "#Β   4th: the QGT condition number (real part) (with regularization)\n";
    _file_QGT_cond << "#Β   5th: the QGT condition number (imaginary part) (with regularization)\n";
    _file_QGT_cond << "###########################################################################\n";

  }

}


void VMC_Sampler :: setFile_QGT_eigen(std::string info, int rank) {

  _write_QGT_eigen = true;
  if(rank == 0){

    _file_QGT_eigen.open("qgt_eigen_" + info + ".dat");
    if(!_file_QGT_eigen.good()){

      std::cerr << " ##FileError: Cannot open the file βΉβΉ qgt_eigen_" << info << ".dat βΊβΊ for writing the eigenvalues of the Quantum Geometric Tensor." << std::endl;
      std::abort();

    }
    else
      std::cout << " Saving the QGT eigenvalues after each πππ step on file βΉβΉ qgt_eigen_" << info << ".dat βΊβΊ." << std::endl;

    _file_QGT_eigen << "#####################################\n";
    _file_QGT_eigen << "# Column Legend\n";
    _file_QGT_eigen << "#\n";
    _file_QGT_eigen << "#Β   1st: the πππ-step identifier\n";
    _file_QGT_eigen << "#Β   2nd: the QGT eigenvalues\n";
    _file_QGT_eigen << "#####################################\n";

  }

}


void VMC_Sampler :: Write_Move_Statistics(unsigned int opt_step, MPI_Comm common) {

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  //Function variables
  unsigned int global_acc_visible = 0, global_prop_visible = 0;
  unsigned int global_acc_ket = 0, global_prop_ket = 0;
  unsigned int global_acc_bra = 0, global_prop_bra = 0;
  unsigned int global_acc_equal_site = 0, global_prop_equal_site = 0;
  unsigned int global_acc_visible_nn_site = 0, global_prop_visible_nn_site = 0;
  unsigned int global_acc_hidden_nn_site = 0, global_prop_hidden_nn_site = 0;

  //Shares move statistics among all the nodes
  MPI_Reduce(&_N_accepted_visible, &global_acc_visible, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_proposed_visible, &global_prop_visible, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_accepted_ket, &global_acc_ket, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_proposed_ket, &global_prop_ket, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_accepted_bra, &global_acc_bra, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_proposed_bra, &global_prop_bra, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_accepted_equal_site, &global_acc_equal_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_proposed_equal_site, &global_prop_equal_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_accepted_visible_nn_site, &global_acc_visible_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_proposed_visible_nn_site, &global_prop_visible_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_accepted_hidden_nn_site, &global_acc_hidden_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
  MPI_Reduce(&_N_proposed_hidden_nn_site, &global_prop_hidden_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);

  if(rank == 0){

    _file_Move_Statistics << opt_step + 1;
    _file_Move_Statistics << std::scientific;
    //_file_Move_Statistics << std::setprecision(10) << std::fixed;
    _file_Move_Statistics << "\t" << 100.0 * global_acc_visible / global_prop_visible;
    if(_N_proposed_ket == 0) _file_Move_Statistics << "\t" << 0.0;
    else _file_Move_Statistics << "\t" << 100.0 * global_acc_ket / global_prop_ket;
    if(_N_proposed_bra == 0) _file_Move_Statistics << "\t" << 0.0;
    else _file_Move_Statistics << "\t" << 100.0 * global_acc_bra / global_prop_bra;
    if(_N_proposed_equal_site==0) _file_Move_Statistics << "\t" << 0.0;
    else _file_Move_Statistics << "\t" << 100.0 * global_acc_equal_site / global_prop_equal_site;
    if(_N_proposed_visible_nn_site==0) _file_Move_Statistics << "\t" << 0.0;
    else _file_Move_Statistics << "\t" << 100.0 * global_acc_visible_nn_site / global_prop_visible_nn_site;
    if(_N_proposed_hidden_nn_site==0) _file_Move_Statistics << "\t" << 0.0;
    else _file_Move_Statistics << "\t" << 100.0 * global_acc_hidden_nn_site / global_prop_hidden_nn_site << std::endl;
    _file_Move_Statistics << std::endl;

  }

}


void VMC_Sampler :: Write_MCMC_Config(unsigned int mcmc_step, int rank) {

  if(_write_MCMC_Config){

    if(rank == 0){

      _file_MCMC_Config << mcmc_step + 1;

      //Prints the visible configuration |πβ©
      _file_MCMC_Config << "\t|π β©" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
          _file_MCMC_Config << _configuration(j_row, j_col) << std::setw(4);
        _file_MCMC_Config << std::endl << "   " << std::setw(4);

      }

      //Prints the ket configuration |πβ©
      if(_hidden_ket.is_empty()) _file_MCMC_Config << "\t|π β©" << std::endl;
      else{

        _file_MCMC_Config << "\t|π β©" << std::setw(4);
        for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
            _file_MCMC_Config << _hidden_ket(j_row, j_col) << std::setw(4);
          _file_MCMC_Config << std::endl << "   " << std::setw(4);

        }

      }

      //Prints the bra configuration β¨πΛ|
      if(_hidden_bra.is_empty()) _file_MCMC_Config << "\tβ¨πΛ|" << std::endl;
      else{

        _file_MCMC_Config << "\tβ¨πΛ|" << std::setw(4);
        for(unsigned int j_row = 0; j_row < _hidden_bra.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_bra.n_cols; j_col++)
            _file_MCMC_Config << _hidden_bra(j_row, j_col) << std::setw(4);
          _file_MCMC_Config << std::endl;

        }

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_final_Config(unsigned int opt_step) {

  if(_write_final_Config){

    _file_final_Config << opt_step + 1 << "\t|π β©" << std::setw(4);
    //Prints the visible configuration |π β©
    for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

      for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
        _file_final_Config << _configuration(j_row, j_col) << std::setw(4);
      _file_final_Config << std::endl << "   " << std::setw(4);

    }

    //Prints the ket configuration |π β©
    if(_hidden_ket.is_empty()) _file_final_Config << "\t|π β©" << std::endl;
    else{

      _file_final_Config << "\t|π β©" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
          _file_final_Config << _hidden_ket(j_row, j_col) << std::setw(4);
        _file_final_Config << std::endl;;

      }

    }

    //Prints the bra configuration β¨πΛ|
    if(_hidden_bra.is_empty()) _file_final_Config << "\tβ¨πΛ|" << std::endl;
    else{

      _file_final_Config << "\tβ¨πΛ|" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_bra.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_bra.n_cols; j_col++)
          _file_final_Config << _hidden_bra(j_row, j_col) << std::setw(4);
        _file_final_Config << std::endl;

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_opt_Params(int rank) {

  if(_write_opt_Params){

    if(rank == 0){

      if(!_if_shadow) _file_opt_Params << _vqs.n_visible() << "\n" << _vqs.density()*_vqs.n_visible() << std::endl;
      else _file_opt_Params << _vqs.n_visible() << "\n" << _vqs.phi() << std::endl;

      for(unsigned int p = 0; p < _vqs.n_alpha(); p++) _file_opt_Params << _vqs.alpha_at(p).real() << " " << _vqs.alpha_at(p).imag() << std::endl;

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_all_Params(unsigned int opt_step, int rank) {

  if(_write_all_Params){

    if(rank == 0){

      _file_all_Params << opt_step + 1 << " " << _vqs.phi().real() << " " << _vqs.phi().imag() << std::endl;
      for(unsigned int p = 0; p < _vqs.n_alpha(); p++)
        _file_all_Params << opt_step + 1 << " " << _vqs.alpha_at(p).real() << " " << _vqs.alpha_at(p).imag() << std::endl;

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_matrix(unsigned int opt_step) {

  if(_write_QGT_matrix){

    _file_QGT_matrix << opt_step + 1 << "\t";
    _file_QGT_matrix << std::setprecision(20) << std::fixed;
    if(_if_shadow){

      for(unsigned int j = 0; j < _Q.row(0).n_elem; j++) _file_QGT_matrix << _Q.row(0)(j).real() << " ";
      _file_QGT_matrix << std::endl;
      for(unsigned int d = 1; d < _Q.n_rows; d++){

        _file_QGT_matrix << "\t";
        for(unsigned int j = 0; j < _Q.row(d).n_elem; j++) _file_QGT_matrix << _Q.row(d)(j).real() << " ";
        _file_QGT_matrix << std::endl;

      }

    }
    else{

      for(unsigned int j = 0; j < _Q.row(0).n_elem; j++) _file_QGT_matrix << _Q.row(0)(j) << " ";
      _file_QGT_matrix << std::endl;
      for(unsigned int d = 1; d < _Q.n_rows; d++){

        _file_QGT_matrix << "\t";
        for(unsigned int j = 0; j < _Q.row(d).n_elem; j++) _file_QGT_matrix << _Q.row(d)(j) << " ";
        _file_QGT_matrix << std::endl;

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_cond(unsigned int opt_step) {

  if(_write_QGT_cond){

    _file_QGT_cond << opt_step + 1 << "\t";
    _file_QGT_cond << std::setprecision(10) << std::fixed;
    if(_if_shadow){

      double C;
      if(_if_QGT_reg){

        C = cond(real(_Q));
        _file_QGT_cond << C << " ";
        C = cond(real(_Q) + _eps * _I);
        _file_QGT_cond << C << std::endl;

      }
      else{

        C = cond(real(_Q));
        _file_QGT_cond << C << std::endl;

      }

    }
    else{

      std::complex <double> C;
      if(_if_QGT_reg){

        C = cond(_Q);
        _file_QGT_cond << C.real() << " " << C.imag() << " ";
        C = cond(_Q + _eps * _I);
        _file_QGT_cond << C.real() << " " << C.imag() << std::endl;

      }
      else{

        C = cond(_Q);
        _file_QGT_cond << C.real() << " " << C.imag() << std::endl;

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_eigen(unsigned int opt_step) {

  if(_write_QGT_eigen){

    if(_if_shadow){

      vec eigenval;
      if(_if_QGT_reg) eigenval = eig_sym(real(_Q) + _eps * _I);
      else eigenval = eig_sym(real(_Q));
      for(unsigned int e = 0; e < eigenval.n_elem; e++) _file_QGT_eigen << opt_step + 1 << " " << eigenval(e) << "\n";

    }
    else{

      cx_vec eigenval;
      if(_if_QGT_reg) eigenval = eig_gen(_Q + _eps * _I);
      else eigenval = eig_gen(_Q);
      for(unsigned int e = 0; e < eigenval.n_elem; e++) _file_QGT_eigen << opt_step + 1 << " " << eigenval(e) << "\n";

    }

  }
  else
    return;

}


void VMC_Sampler :: CloseFile(int rank) {

  if(_write_final_Config) _file_final_Config.close();
  if(rank == 0){

    if(_write_Move_Statistics) _file_Move_Statistics.close();
    if(_write_MCMC_Config) _file_MCMC_Config.close();
    if(_write_block_Observables){

      _file_block_SigmaX.close();
      _file_block_SigmaY.close();
      _file_block_SigmaZ.close();
      _file_block_Energy.close();

    }
    if(_write_opt_Observables){

      _file_opt_SigmaX.close();
      _file_opt_SigmaY.close();
      _file_opt_SigmaZ.close();
      _file_opt_Energy.close();

    }
    if(_write_all_Params) _file_all_Params.close();
    if(_write_opt_Params) _file_opt_Params.close();
    if(_write_QGT_matrix) _file_QGT_matrix.close();
    if(_write_QGT_cond) _file_QGT_cond.close();
    if(_write_QGT_eigen) _file_QGT_eigen.close();

  }

}


void VMC_Sampler :: Finalize(int rank) {

  _rnd.SaveSeed(rank);

}


void VMC_Sampler :: Reset() {

  /*##########################################################*/
  //  This function must be called every time a new πππ step
  //  is about to begin. In fact, it performs an appropriate
  //  initialization of all the variables necessary for the
  //  stochastic estimation of the quantum observables.
  /*##########################################################*/

  _instReweight.reset();
  _instO_ket.reset();
  _instO_bra.reset();
  _instObs_ket.reset();
  _instObs_bra.reset();

}


void VMC_Sampler :: Measure() {

  /*########################################################################################################*/
  //  Evaluates the instantaneous quantum observables along the MCMC.
  //  In a Quantum Monte Carlo (QMC) algorithm, every time a quantum
  //  configuration |π?β© is sampled via the Metropolis-Hastings test,
  //  an instantaneous evaluation of a certain system properties, represented by
  //  a self-adjoint operator πΈ, can be done by evaluating the Monte Carlo average
  //  of the instantaneous local observables π, defined as:
  //
  //        π β‘ π(π) = Ξ£π' β¨π|πΈ|π'β©β’Ξ¨(π',πΆ)/Ξ¨(π,πΆ)        (πβ΄π-π?π½πΆπΉβ΄π)
  //        π β‘ π(π,π) = Ξ£π' β¨π|πΈ|π'β©β’Ξ¦(π',π,πΆ)/Ξ¦(π,π,πΆ)  (π?π½πΆπΉβ΄π)
  //
  //  where the matrix elements β¨π|πΈ|π'β© are the connections of the
  //  quantum observable operator πΈ related to the visible configuration |πβ© and
  //  the |π'β© configurations are all the system configurations connected to |πβ©.
  //  Whereupon, we can compute the Monte Carlo average value of πππ quantum
  //  observable πΈ on the variational state as
  //
  //        β¨πΈβ© = β¨πβ©             (πβ΄π-π?π½πΆπΉβ΄π)
  //        β¨πΈβ© = βͺπα΄Ώβ« + βπα΄΅β   (π?π½πΆπΉβ΄π)
  //
  //  Therefore, this function has the task of calculating and saving in memory the instantaneous
  //  values of the quantities of interest that allow to estimate, in a second moment, the (Monte Carlo)
  //  average properties, whenever a new configuration is sampled.
  //  To this end, _π’π§π¬π­πππ¬_π€ππ­ and _π’π§π¬π­πππ¬_ππ«π are matrices, whose rows keep in memory the
  //  instantaneous values of the various observables that we want to calculate (π(π,π) and π(π,πΛ)),
  //  and it will have as many columns as the number of points (i.e. the sampled configuration |π?β©)
  //  that form the MCMC on which these instantaneous values are calculated.
  //  The function also calculates the values of the local operators
  //
  //        π(π,π) = βπππ(Ξ¦(π,π,π))/βπ
  //        π(π,πΛ) = βπππ(Ξ¦(π,π,π))/βπ
  //
  //  related to the variational state on the current sampled configuration |π?β©.
  //  The instantaneous values of πππ πΌπΌ and π πππΌπΌ are stored in the rows of the matrix _π’π§π¬π­πππ°ππ’π π‘π­ and
  //  all the above computations will be combined together in the ππ¬π­π’π¦ππ­π() function in order to
  //  obtained the desired Monte Carlo estimation.
  //
  //  NΜ²OΜ²TΜ²EΜ²: in the case of the Shadow wave function it may be necessary to make many more
  //        integrations of the auxiliary variables, compared to those already made in each
  //        simulation together with the visible ones. This is due to the fact that the
  //        correlations induced by the auxiliary variables, which are not physical,
  //        could make the instantaneous measurement of the observables very noisy,
  //        making the algorithm unstable, especially in the inversion of the QGT.
  //        Therefore we add below the possibility to take further samples of the
  //        hidden variables within the single Monte Carlo measurement, to increase the
  //        statistics and make the block observables less noisy along each simulation.
  /*########################################################################################################*/

  //Find the connections of each observables
  _H.FindConn(_configuration, _StatePrime, _Connections);  // β¨π|πΈ|π'β© for all |π'β©

  //Function variables
  unsigned int n_props = _Connections.n_rows;  //Number of quantum observables
  _Observables.set_size(n_props, 1);  //Only sizing, this should be computed in ππ¬π­π’π¦ππ­π()
  _global_Observables.set_size(n_props, 1);
  Col <double> cosin(2, fill::zeros);  //Storage variable for cos[β(π£, π½) - β(π£, π½')] and sin[β(π£, π½) - β(π£, π½')]
  Col <std::complex <double>> A_ket(n_props, fill::zeros);  //Storage value for π(π,π)
  Col <std::complex <double>> A_bra(n_props, fill::zeros);  //Storage value for π(π,πΛ)
  Col <std::complex <double>> O_ket(_vqs.n_alpha(), fill::zeros);
  Col <std::complex <double>> O_bra(_vqs.n_alpha(), fill::zeros);

  //Makes the Shadow measurement less noisy
  if(_if_extra_hidden_sum){

    //Extra sampling of the hidden variables
    if(_Nblks_extra == 0){

      std::cerr << " ##ValueError: not to use βblock averagingβ during the extra hidden sampling set _Nblks_extra = π£." << std::endl;
      std::cerr << "   Failed to measure instantaneous quantum properties of the system." << std::endl;
      std::abort();

    }
    else if(_Nblks_extra == 1){  //No βblock averagingβ

      for(unsigned int extra_step = 0; extra_step < _Nextra; extra_step++){

        for(unsigned int n_bunch = 0; n_bunch < _M; n_bunch++){

          this -> Move_ket(_Nflips);
          this -> Move_bra(_Nflips);

        }
        cosin(0) += _vqs.cosII(_configuration, _hidden_ket, _hidden_bra);
        cosin(1) += _vqs.sinII(_configuration, _hidden_ket, _hidden_bra);
        _vqs.LocalOperators(_configuration, _hidden_ket, _hidden_bra);
        O_ket += _vqs.O().col(0);
        O_bra += _vqs.O().col(1);
        for(unsigned int Nobs = 0; Nobs < n_props; Nobs++){

          for(unsigned int mel = 0; mel < _Connections(Nobs).n_elem; mel++){

            A_ket(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_ket);  // π(π,π)
            A_bra(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_bra);  // π(π,π')

          }

        }

      }
      cosin /= double(_Nextra);  //  β¨β¨πππ β©α΅Λ‘α΅β© & β¨β¨π ππβ©α΅Λ‘α΅β©
      A_ket /= double(_Nextra);  //  β¨β¨π(π,π)β©α΅Λ‘α΅β©
      A_bra /= double(_Nextra);  //  β¨β¨π(π,π')β©α΅Λ‘α΅β©
      O_ket /= double(_Nextra);  //  β¨β¨π(π,π)β©α΅Λ‘α΅β©
      O_bra /= double(_Nextra);  //  β¨β¨π(π,π')β©α΅Λ‘α΅β©

    }
    else{  //βblock averagingβ

      unsigned int blk_size = std::floor(double(_Nextra / _Nblks_extra));
      double cos_blk, sin_blk;
      Col <std::complex <double>> A_ket_blk(n_props);
      Col <std::complex <double>> A_bra_blk(n_props);
      Col <std::complex <double>> O_ket_blk(_vqs.n_alpha());
      Col <std::complex <double>> O_bra_blk(_vqs.n_alpha());
      for(unsigned int block_ID = 0; block_ID < _Nblks_extra; block_ID++){

        cos_blk = 0.0;
        sin_blk = 0.0;
        A_ket_blk.zeros();
        A_bra_blk.zeros();
        O_ket_blk.zeros();
        O_bra_blk.zeros();
        for(unsigned int l =  0; l < blk_size; l++){  //Computes single block estimates of the instantaneous measurement

          for(unsigned int n_bunch = 0; n_bunch < _M; n_bunch++){  //Moves only the hidden configuration

            this -> Move_ket(_Nflips);
            this -> Move_bra(_Nflips);

          }
          cos_blk += _vqs.cosII(_configuration, _hidden_ket, _hidden_bra);
          sin_blk += _vqs.sinII(_configuration, _hidden_ket, _hidden_bra);
          _vqs.LocalOperators(_configuration, _hidden_ket, _hidden_bra);
          O_ket_blk += _vqs.O().col(0);
          O_bra_blk += _vqs.O().col(1);
          for(unsigned int Nobs = 0; Nobs < n_props; Nobs++){

            for(unsigned int mel = 0; mel < _Connections(Nobs).n_elem; mel++){

              A_ket_blk(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_ket);  // π(π,π)
              A_bra_blk(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_bra);  // π(π,π')

            }

          }

        }
        cosin(0) += cos_blk / double(blk_size);  // β¨πππ β©α΅Λ‘α΅
        cosin(1) += sin_blk / double(blk_size);  // β¨π ππβ©α΅Λ‘α΅
        A_ket += A_ket_blk / double(blk_size);  //  β¨π(π,π)β©α΅Λ‘α΅
        A_bra += A_bra_blk / double(blk_size);  //  β¨π(π,π')β©α΅Λ‘α΅
        O_ket += O_ket_blk / double(blk_size);  //  β¨π(π,π)β©α΅Λ‘α΅
        O_bra += O_bra_blk / double(blk_size);  //  β¨π(π,π')β©α΅Λ‘α΅

      }
      cosin /= double(_Nblks_extra);  //  β¨β¨πππ β©α΅Λ‘α΅β© & β¨β¨π ππβ©α΅Λ‘α΅β©
      A_ket /= double(_Nblks_extra);  //  β¨β¨π(π,π)β©α΅Λ‘α΅β©
      A_bra /= double(_Nblks_extra);  //  β¨β¨π(π,π')β©α΅Λ‘α΅β©
      O_ket /= double(_Nblks_extra);  //  β¨β¨π(π,π)β©α΅Λ‘α΅β©
      O_bra /= double(_Nblks_extra);  //  β¨β¨π(π,π')β©α΅Λ‘α΅β©

    }

  }
  else{

    //Computes cos[β(π£, π½) - β(π£, π½')] and sin[β(π£, π½) - β(π£, π½')]
    cosin(0) = _vqs.cosII(_configuration, _hidden_ket, _hidden_bra);
    cosin(1) = _vqs.sinII(_configuration, _hidden_ket, _hidden_bra);

    //Instantaneous evaluation of the quantum observables
    for(unsigned int Nobs = 0; Nobs < n_props; Nobs++){

      for(unsigned int mel = 0; mel < _Connections(Nobs).n_elem; mel++){

        A_ket(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_ket);  // π(π,π)
        A_bra(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_bra);  // π(π,π')

      }

    }

    //Instantaneous evaluation of the local operators
    _vqs.LocalOperators(_configuration, _hidden_ket, _hidden_bra);  //Computes π(π,π) and π(π,π')
    O_ket = _vqs.O().col(0);
    O_bra = _vqs.O().col(1);

  }

  //Adds Monte Carlo statistics
  _instReweight.insert_cols(_instReweight.n_cols, cosin);  // β‘ instantaneous measure of the πππ  and of the π ππ
  _instObs_ket.insert_cols(_instObs_ket.n_cols, A_ket);  // β‘ instantaneous measure of π(π,π)
  _instObs_bra.insert_cols(_instObs_bra.n_cols, A_bra);  // β‘ instantaneous measure of π(π,π')
  _instO_ket.insert_cols(_instO_ket.n_cols, O_ket);  // β‘ instantaneous measure of π(π,π)
  _instO_bra.insert_cols(_instO_bra.n_cols, O_bra);  // β‘ instantaneous measure of π(π,π')

}


void VMC_Sampler :: Estimate(MPI_Comm common) {

  /*#############################################################################################*/
  //  This function is called at the end of the single VMC step and
  //  estimates the averages of the quantum observables
  //  as a Monte Carlo stochastic mean value on the choosen variational quantum state, i.e.:
  //
  //        β¨πΈβ© = β¨πβ©             (πβ΄π-π?π½πΆπΉβ΄π)
  //        β¨πΈβ© = βͺπα΄Ώβ« + βπα΄΅β   (π?π½πΆπΉβ΄π)
  //
  //  with the relative uncertainties via the Blocking Method.
  //  We define the above special expectation value in the following way:
  //
  //        βͺβ¦β« = 1/2β’Ξ£πΞ£πΞ£πΛπ(π,π,πΛ)β’πππ [β(π£,π½)-β(π£,π½')]β’[β¦(π,π) + β¦(π,πΛ)]
  //            = 1/2β’β¨πππ [β(π£,π½)-β(π£,π½')]β’[β¦(π,π) + β¦(π,πΛ)]β© / β¨πππ [β(π£,π½)-β(π£,π½')]β©
  //        ββ¦β = 1/2β’Ξ£πΞ£πΞ£πΛπ(π,π,πΛ)β’π ππ[β(π£,π½)-β(π£,π½')]β’[β¦(π,πΛ) - β¦(π,π)]
  //            = 1/2β’β¨π ππ[β(π£,π½)-β(π£,π½')]β’[β¦(π,πΛ) - β¦(π,π)]β© / β¨πππ [β(π£,π½)-β(π£,π½')]β©
  //
  //  in which the standard expectation value β¨β¦β© are calculated in a standard way with
  //  the Monte Carlo sampling of π(π,π,πΛ), and the normalization given by the cosine
  //  is due to the π«ππ°ππ’π π‘π­π’π§π  technique necessary to correctly estimate the various quantities.
  //  In the non-shadow case we have:
  //
  //        βͺβ¦β« β βΉβΊ, i.e. the standard Monte Carlo expectation value
  //        ββ¦β β 0
  //
  //  The instantaneous values along the single Markov chain necessary to make the Monte Carlo
  //  estimates just defined are computed by the ππππ¬π?π«π() function and are stored in the
  //  following data-members:
  //
  //        _π’π§π¬π­πππ¬_π€ππ­  βΉ--βΊ  quantum observable π(π,π)
  //        _π’π§π¬π­πππ¬_ππ«π  βΉ--βΊ  quantum observable π(π,π')
  //        _π’π§π¬π­πππ°ππ’π π‘π­  βΉ--βΊ  πππ [β(π£,π½)-β(π£,π½')] & π ππ[β(π£,π½)-β(π£,π½')]
  //        _π’π§π¬π­π_π€ππ­  βΉ--βΊ  π(π,π)
  //        _π’π§π¬π­π_ππ«π  βΉ--βΊ  π(π,π')
  //
  //  The Quantum Geometric Tensor and the energy gradient required to optimize the variational
  //  parameters are also (stochastically) computed.
  /*#############################################################################################*/

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  //Computes all necessary MC block estimates without yet adjusting with the reweighting ratio
  this -> compute_Reweighting_ratio(common);
  this -> compute_Quantum_observables(common);

  //Computes all stuff for the update of variational parameters
  if(!_if_vmc){

    this -> compute_O();
    this -> compute_QGTandGrad(common);
    this -> QGT_Check(rank);

  }

}


void VMC_Sampler :: Write_Quantum_properties(unsigned int tdvmc_step, int rank) {

  /*############################################################*/
  //  We save on the output file the real and imaginary part
  //  with the relative uncertainties of the
  //  quantum observables via "block averaging": if everything
  //  has gone well, the imaginary part of the estimates of
  //  quantum operators MUST be statistically zero.
  /*############################################################*/

  //Computes progressive averages of the reweighting ratio with "block averaging" uncertainties
  if(rank == 0){

    Col <double> prog_cos = this -> compute_progressive_averages(_global_cosII);
    Col <double> err_cos = this -> compute_errorbar(_global_cosII);
    Col <double> prog_sin = this -> compute_progressive_averages(_global_sinII);
    Col <double> err_sin = this -> compute_errorbar(_global_sinII);

    if(!_if_shadow){

      //Computes progressive averages of quantum observables with "block averaging" uncertainties
      Col <std::complex <double>> prog_energy = this -> compute_progressive_averages(_global_Observables(0, 0));
      Col <std::complex <double>> err_energy = this -> compute_errorbar(_global_Observables(0, 0));
      Col <std::complex <double>> prog_Sx = this -> compute_progressive_averages(_global_Observables(1, 0));
      Col <std::complex <double>> err_Sx = this -> compute_errorbar(_global_Observables(1, 0));
      Col <std::complex <double>> prog_Sy = this -> compute_progressive_averages(_global_Observables(2, 0));
      Col <std::complex <double>> err_Sy = this -> compute_errorbar(_global_Observables(2, 0));
      Col <std::complex <double>> prog_Sz = this -> compute_progressive_averages(_global_Observables(3, 0));
      Col <std::complex <double>> err_Sz = this -> compute_errorbar(_global_Observables(3, 0));

      //Writes all system properties computations on files
      if(_write_block_Observables){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          //Writes energy
          _file_block_Energy << std::setprecision(10) << std::fixed;
          _file_block_Energy << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_Energy << prog_cos(block_ID) << "\t" << err_cos(block_ID) << "\t";
          _file_block_Energy << prog_sin(block_ID) << "\t" << err_sin(block_ID) << "\t";
          _file_block_Energy << prog_energy(block_ID).real() << "\t" << err_energy(block_ID).real() << "\t";
          _file_block_Energy << prog_energy(block_ID).imag() << "\t" << err_energy(block_ID).imag() << "\t";
          _file_block_Energy << std::endl;

          //Writes ΟΜΛ£
          _file_block_SigmaX << std::setprecision(10) << std::fixed;
          _file_block_SigmaX << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_SigmaX << prog_Sx(block_ID).real() << "\t" << err_Sx(block_ID).real() << "\t";
          _file_block_SigmaX << prog_Sx(block_ID).imag() << "\t" << err_Sx(block_ID).imag() << "\t";
          _file_block_SigmaX << std::endl;

          //Writes block ΟΜΚΈ
          _file_block_SigmaY << std::setprecision(10) << std::fixed;
          _file_block_SigmaY << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_SigmaY << prog_Sy(block_ID).real() << "\t" << err_Sy(block_ID).real() << "\t";
          _file_block_SigmaY << prog_Sy(block_ID).imag() << "\t" << err_Sy(block_ID).imag() << "\t";
          _file_block_SigmaY << std::endl;

          //Writes block ΟΜαΆ»
          _file_block_SigmaZ << std::setprecision(10) << std::fixed;
          _file_block_SigmaZ << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_SigmaZ << prog_Sz(block_ID).real() << "\t" << err_Sz(block_ID).real() << "\t";
          _file_block_SigmaZ << prog_Sz(block_ID).imag() << "\t" << err_Sz(block_ID).imag() << "\t";
          _file_block_SigmaZ << std::endl;

        }

      }

      //Saves optimized quantum observables along the πππππ
      if(_write_opt_Observables){

        // πΈ(π,πΆ) +/- πππΉ[πΈ(π,πΆ)]
        _file_opt_Energy << std::setprecision(20) << std::fixed;
        _file_opt_Energy << tdvmc_step + 1 << "\t";
        _file_opt_Energy << prog_cos(_Nblks - 1) << "\t" << err_cos(_Nblks - 1) << "\t";
        _file_opt_Energy << prog_sin(_Nblks - 1) << "\t" << err_sin(_Nblks - 1) << "\t";
        _file_opt_Energy << prog_energy(_Nblks - 1).real() << "\t" << err_energy(_Nblks - 1).real() << "\t";
        _file_opt_Energy << prog_energy(_Nblks - 1).imag() << "\t" << err_energy(_Nblks - 1).imag() << "\t";
        _file_opt_Energy << _E.real() << "\t" << _E.imag();
        _file_opt_Energy << std::endl;

        // π(π,πΆ) +/- πππΉ[π(π, πΆ)]
        _file_opt_SigmaX << std::setprecision(20) << std::fixed;
        _file_opt_SigmaX << tdvmc_step + 1 << "\t";
        _file_opt_SigmaX << prog_Sx(_Nblks - 1).real() << "\t" << err_Sx(_Nblks - 1).real() << "\t";
        _file_opt_SigmaX << prog_Sx(_Nblks - 1).imag() << "\t" << err_Sx(_Nblks - 1).imag() << "\t";
        _file_opt_SigmaX << std::endl;

        _file_opt_SigmaY << std::setprecision(20) << std::fixed;
        _file_opt_SigmaY << tdvmc_step + 1 << "\t";
        _file_opt_SigmaY << prog_Sy(_Nblks - 1).real() << "\t" << err_Sy(_Nblks - 1).real() << "\t";
        _file_opt_SigmaY << prog_Sy(_Nblks - 1).imag() << "\t" << err_Sy(_Nblks - 1).imag() << "\t";
        _file_opt_SigmaY << std::endl;

        _file_opt_SigmaZ << std::setprecision(20) << std::fixed;
        _file_opt_SigmaZ << tdvmc_step + 1 << "\t";
        _file_opt_SigmaZ << prog_Sz(_Nblks - 1).real() << "\t" << err_Sz(_Nblks - 1).real() << "\t";
        _file_opt_SigmaZ << prog_Sz(_Nblks - 1).imag() << "\t" << err_Sz(_Nblks - 1).imag() << "\t";
        _file_opt_SigmaZ << std::endl;

      }

    }
    else{

      //Computes the true Shadow observable via reweighting ratio in each block
      Col <double> shadow_energy = real(_global_Observables(0, 0)) / _global_cosII;  //Computes β¨ββ©β±Όα΅Λ‘α΅/β¨πππ β©β±Όα΅Λ‘α΅ in each block
      Col <double> shadow_Sx = real(_global_Observables(1, 0)) / _global_cosII;  //Computes β¨ΟΜΛ£β©β±Όα΅Λ‘α΅/β¨πππ β©β±Όα΅Λ‘α΅ in each block
      Col <double> shadow_Sy = real(_global_Observables(2, 0)) / _global_cosII;  //Computes β¨ΟΜΚΈβ©β±Όα΅Λ‘α΅/β¨πππ β©β±Όα΅Λ‘α΅ in each block
      Col <double> shadow_Sz = real(_global_Observables(3, 0)) / _global_cosII;  //Computes β¨ΟΜαΆ»β©β±Όα΅Λ‘α΅/β¨πππ β©β±Όα΅Λ‘α΅ in each block

      //Computes progressive averages of quantum observables with "block averaging" uncertainties
      Col <double> prog_energy = this -> compute_progressive_averages(shadow_energy);
      Col <double> err_energy = this -> compute_errorbar(shadow_energy);
      Col <double> prog_Sx = this -> compute_progressive_averages(shadow_Sx);
      Col <double> err_Sx = this -> compute_errorbar(shadow_Sx);
      Col <double> prog_Sy = this -> compute_progressive_averages(shadow_Sy);
      Col <double> err_Sy = this -> compute_errorbar(shadow_Sy);
      Col <double> prog_Sz = this -> compute_progressive_averages(shadow_Sz);
      Col <double> err_Sz = this -> compute_errorbar(shadow_Sz);

      //Writes all system properties computations on files
      if(_write_block_Observables){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          //Writes energy
          _file_block_Energy << std::setprecision(10) << std::fixed;
          _file_block_Energy << tdvmc_step + 1 << " " << block_ID + 1 << " ";
          _file_block_Energy << _cosII(block_ID) << " " << _sinII(block_ID);
          _file_block_Energy << _Observables(0, 0)(block_ID).real() << " ";
          _file_block_Energy << prog_cos(block_ID) << " " << err_cos(block_ID) << " ";
          _file_block_Energy << prog_sin(block_ID) << " " << err_sin(block_ID) << " ";
          _file_block_Energy << prog_energy(block_ID) << " " << err_energy(block_ID) << " ";
          _file_block_Energy << 0.0 << " " << 0.0 << " ";
          _file_block_Energy << std::endl;

          //Writes ΟΜΛ£
          _file_block_SigmaX << std::setprecision(10) << std::fixed;
          _file_block_SigmaX << tdvmc_step + 1 << " " << block_ID + 1 << " ";
          _file_block_SigmaX << prog_Sx(block_ID) << " " << err_Sx(block_ID) << " ";
          _file_block_SigmaX << 0.0 << " " << 0.0 << " ";
          _file_block_SigmaX << std::endl;

          //Writes block ΟΜΚΈ
          _file_block_SigmaY << std::setprecision(10) << std::fixed;
          _file_block_SigmaY << tdvmc_step + 1 << " " << block_ID + 1 << " ";
          _file_block_SigmaY << prog_Sy(block_ID) << " " << err_Sy(block_ID) << " ";
          _file_block_SigmaY << 0.0 << " " << 0.0 << " ";
          _file_block_SigmaY << std::endl;

          //Writes block ΟΜαΆ»
          _file_block_SigmaZ << std::setprecision(10) << std::fixed;
          _file_block_SigmaZ << tdvmc_step + 1 << " " << block_ID + 1 << " ";
          _file_block_SigmaZ << prog_Sz(block_ID) << " " << err_Sz(block_ID) << " ";
          _file_block_SigmaZ << 0.0 << " " << 0.0 << " ";
          _file_block_SigmaZ << std::endl;

        }

      }

      //Saves optimized quantum observables along the πππππ
      if(_write_opt_Observables){

        // πΈ(π,πΆ) +/- πππΉ[πΈ(π,πΆ)]
        _file_opt_Energy << std::setprecision(20) << std::fixed;
        _file_opt_Energy << tdvmc_step + 1 << " ";
        _file_opt_Energy << prog_cos(_Nblks - 1) << " " << err_cos(_Nblks - 1) << " ";
        _file_opt_Energy << prog_sin(_Nblks - 1) << " " << err_sin(_Nblks - 1) << " ";
        _file_opt_Energy << prog_energy(_Nblks - 1) << " " << err_energy(_Nblks - 1) << " ";
        _file_opt_Energy << 0.0 << " " << 0.0 << " " << _E.real() << " " << _E.imag();
        _file_opt_Energy << std::endl;

        // π(π,πΆ) +/- πππΉ[π(π, πΆ)]
        _file_opt_SigmaX << std::setprecision(20) << std::fixed;
        _file_opt_SigmaX << tdvmc_step + 1 << " ";
        _file_opt_SigmaX << prog_Sx(_Nblks - 1) << " " << err_Sx(_Nblks - 1) << " ";
        _file_opt_SigmaX << 0.0 << " " << 0.0 << " ";
        _file_opt_SigmaX << std::endl;

        _file_opt_SigmaY << std::setprecision(20) << std::fixed;
        _file_opt_SigmaY << tdvmc_step + 1 << " ";
        _file_opt_SigmaY << prog_Sy(_Nblks - 1) << " " << err_Sy(_Nblks - 1) << " ";
        _file_opt_SigmaY << 0.0 << " " << 0.0 << " ";
        _file_opt_SigmaY << std::endl;

        _file_opt_SigmaZ << std::setprecision(20) << std::fixed;
        _file_opt_SigmaZ << tdvmc_step + 1 << " ";
        _file_opt_SigmaZ << prog_Sz(_Nblks - 1) << " " << err_Sz(_Nblks - 1) << " ";
        _file_opt_SigmaZ << 0.0 << " " << 0.0 << " ";
        _file_opt_SigmaZ << std::endl;

      }

    }

    if(!_if_vmc){

      this -> Write_QGT_matrix(tdvmc_step);
      this -> Write_QGT_cond(tdvmc_step);
      this -> Write_QGT_eigen(tdvmc_step);

    }

  }

}


Col <double> VMC_Sampler :: average_in_blocks(const Row <double>& instantaneous_quantity) const {

  /*############################################################*/
  //  This function takes a row from one of the matrix
  //  data-members which contains the instantaneous values
  //  of a certain system properties, calculated along a single
  //  Monte Carlo Markov Chain, and calculates all the averages
  //  in each block of this system properties.
  //  This calculation involves a real-valued quantity.
  /*############################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity.n_elem/_Nblks));  //Sets the block length
  Col <double> blocks_quantity(_Nblks);
  double sum_in_each_block;

  //Computes Monte Carlo averages in each block
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    sum_in_each_block = 0.0;  //Resets the storage variable in each block
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
      sum_in_each_block += instantaneous_quantity(l);
    blocks_quantity(block_ID) = sum_in_each_block / double(blk_size);

  }

  return blocks_quantity;

}


Col <std::complex <double>> VMC_Sampler :: average_in_blocks(const Row <std::complex <double>>& instantaneous_quantity) const {

  /*############################################################*/
  //  This function takes a row from one of the matrix
  //  data-members which contains the instantaneous values
  //  of a certain system properties, calculated at each
  //  Monte Carlo Markov Chain, and calculates all the averages
  //  in each block of this system properties.
  //  This calculation involves a complex-valued quantity.
  /*############################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity.n_elem/_Nblks));  //Sets the block length
  Col <std::complex <double>> blocks_quantity(_Nblks);
  std::complex <double> sum_in_each_block;

  //Computes Monte Carlo averages in each block
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    sum_in_each_block = 0.0;  //Resets the storage variable in each block
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

      sum_in_each_block += instantaneous_quantity(l);

    }
    blocks_quantity(block_ID) = sum_in_each_block / double(blk_size);

  }

  return blocks_quantity;

}


Col <double> VMC_Sampler :: Shadow_average_in_blocks(const Row <std::complex <double>>& instantaneous_quantity_ket,
                                                     const Row <std::complex <double>>& instantaneous_quantity_bra) const {

  /*################################################################*/
  //  Computes
  //
  //        β¨πΈβ©α΅Λ‘α΅ = βͺπα΄Ώβ«α΅Λ‘α΅ + βπα΄΅βα΅Λ‘α΅
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_Nblks));  //Sets the block length
  Col <double> blocks_quantity(_Nblks);
  double sum_in_each_block;

  //Computes Monte Carlo Shadow averages in each block ( ! without the reweighting ration ! )
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    sum_in_each_block = 0.0;
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

      sum_in_each_block += _instReweight.row(0)(l) * (instantaneous_quantity_ket(l).real() + instantaneous_quantity_bra(l).real());
      sum_in_each_block += _instReweight.row(1)(l) * (instantaneous_quantity_bra(l).imag() - instantaneous_quantity_ket(l).imag());

    }
    sum_in_each_block *= 0.5;
    blocks_quantity(block_ID) = sum_in_each_block / double(blk_size);

  }

  return blocks_quantity;

}


Col <double> VMC_Sampler :: Shadow_angled_average_in_blocks(const Row <std::complex <double>>& instantaneous_quantity_ket,
                                                            const Row <std::complex <double>>& instantaneous_quantity_bra) const {

  /*################################################################*/
  //  Computes
  //
  //        βͺπα΄Ώβ«α΅Λ‘α΅
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_Nblks));  //Sets the block length
  Col <double> blocks_angled_quantity(_Nblks);
  double angled_sum_in_each_block;

  //Computes Monte Carlo Shadow βangledβ averages in each block ( ! without the reweighting ration ! )
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    angled_sum_in_each_block = 0.0;
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
      angled_sum_in_each_block += _instReweight.row(0)(l) * (instantaneous_quantity_ket(l).real() + instantaneous_quantity_bra(l).real());
    angled_sum_in_each_block *= 0.5;
    blocks_angled_quantity(block_ID) = angled_sum_in_each_block / double(blk_size);

  }

  return blocks_angled_quantity;

}


Col <double> VMC_Sampler :: Shadow_square_average_in_blocks(const Row <std::complex <double>>& instantaneous_quantity_ket,
                                                            const Row <std::complex <double>>& instantaneous_quantity_bra) const {

  /*################################################################*/
  //  Computes
  //
  //        βπα΄΅βα΅Λ‘α΅
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_Nblks));  //Sets the block length
  Col <double> blocks_square_quantity(_Nblks);
  double square_sum_in_each_block;

  //Computes Monte Carlo Shadow βsquareβ averages in each block ( ! without the reweighting ration ! )
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    square_sum_in_each_block = 0.0;
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
      square_sum_in_each_block += _instReweight.row(1)(l) * (instantaneous_quantity_bra(l).imag() - instantaneous_quantity_ket(l).imag());
    square_sum_in_each_block *= 0.5;
    blocks_square_quantity(block_ID) = square_sum_in_each_block / double(blk_size);

  }

  return blocks_square_quantity;

}


void VMC_Sampler :: compute_Reweighting_ratio(MPI_Comm common) {

  //MPI variables for parallelization
  int rank,size;
  MPI_Comm_size(common, &size);
  MPI_Comm_rank(common, &rank);

  _global_cosII.zeros(_Nblks);
  _global_sinII.zeros(_Nblks);
  _cosII = this -> average_in_blocks(_instReweight.row(0));  //Computes β¨πππ β©β±Όα΅Λ‘α΅ in each block, for j = π£,β¦,π­α΅Λ‘α΅
  _sinII = this -> average_in_blocks(_instReweight.row(1));  //Computes β¨π ππβ©β±Όα΅Λ‘α΅ in each block, for j = π£,β¦,π­α΅Λ‘α΅

  MPI_Barrier(common);

  //Shares block averages among all the nodes
  MPI_Reduce(_cosII.begin(), _global_cosII.begin(), _Nblks, MPI_DOUBLE, MPI_SUM, 0, common);
  MPI_Reduce(_sinII.begin(), _global_sinII.begin(), _Nblks, MPI_DOUBLE, MPI_SUM, 0, common);
  if(rank == 0){

    _global_cosII /= double(size);
    _global_sinII /= double(size);

  }

}


void VMC_Sampler :: compute_Quantum_observables(MPI_Comm common) {

  /*#################################################################################*/
  //  ππ¨π¦π©π?π­ππ¬ πππ ππ§ππ«π π².
  //  We compute the stochastic average via the Blocking technique of
  //
  //        πΈ(π,πΆ) = β¨ββ© β β¨β°β©            (πβ΄π-π?π½πΆπΉβ΄π)
  //        πΈ(π,πΆ) = β¨ββ© β βͺβ°α΄Ώβ« + ββ°α΄΅β   (π?π½πΆπΉβ΄π)
  //
  //  We remember that the matrix rows _π’π§π¬π­πππ¬_π€ππ­(0) and _π’π§π¬π­πππ¬_ππ«π(0) contains
  //  the instantaneous values of the Hamiltonian operator along the MCMC, i.e.
  //  β°(π,π) and β°(π,πΛ).
  /*#################################################################################*/
  /*#################################################################################*/
  //  ππ¨π¦π©π?π­ππ¬ πππ ππ’π§π π₯π ππ©π’π§ πππ¬ππ«π―πππ₯ππ¬.
  //  We compute the stochastic average via the Blocking technique of
  //
  //        πΛ£(π,πΆ) = β¨πΌΛ£β© β β¨πΛ£β©             (πβ΄π-π?π½πΆπΉβ΄π)
  //        πΚΈ(π,πΆ) = β¨πΌΚΈβ© β β¨πΚΈβ©
  //        παΆ»(π,πΆ) = β¨πΌαΆ»β© β β¨παΆ»β©
  //        πΛ£(π,πΆ) = β¨πΌΛ£β© β βͺπΛ£α΄Ώβ« + βπΛ£α΄΅β   (π?π½πΆπΉβ΄π)
  //        πΚΈ(π,πΆ) = β¨πΌΚΈβ© β βͺπΚΈα΄Ώβ« + βπΚΈα΄΅β
  //        παΆ»(π,πΆ) = β¨πΌαΆ»β© β βͺπαΆ»α΄Ώβ« + βπαΆ»α΄΅β
  //
  //  We remember that the matrix rows _π’π§π¬π­πππ¬_π€ππ­(f) and _π’π§π¬π­πππ¬_ππ«π(f) contains
  //  the instantaneous values of the spin projection operator along the MCMC, i.e.
  //  π(π,π) and π(π,πΛ), with f = 1, 2, 3 and where {ΟΜαΆ } the are Pauli matrices
  //  in the computational basis.
  /*#################################################################################*/

  //MPI variables for parallelization
  int rank,size;
  MPI_Comm_size(common, &size);
  MPI_Comm_rank(common, &rank);

  //Computes β¨πͺβ©β±Όα΅Λ‘α΅ in each block
  for(unsigned int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++)
    _global_Observables(n_obs, 0).zeros(_Nblks);

  if(!_if_shadow){

    for(unsigned int n_obs = 0; n_obs < _Observables.n_rows; n_obs++)
      _Observables(n_obs, 0) = this -> average_in_blocks(_instObs_ket.row(n_obs));

  }
  else{

    for(unsigned int n_obs = 0; n_obs < _Observables.n_rows; n_obs++){

      _Observables(n_obs, 0).set_size(_Nblks);
      _Observables(n_obs, 0).set_real(this -> Shadow_average_in_blocks(_instObs_ket.row(n_obs), _instObs_bra.row(n_obs)));
      _Observables(n_obs, 0).set_imag(zeros(_Nblks));

    }

  }

  MPI_Barrier(common);

  //Shares block averages among all the nodes
  MPI_Reduce(_Observables(0, 0).begin(), _global_Observables(0, 0).begin(), _Nblks, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
  MPI_Reduce(_Observables(1, 0).begin(), _global_Observables(1, 0).begin(), _Nblks, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
  MPI_Reduce(_Observables(2, 0).begin(), _global_Observables(2, 0).begin(), _Nblks, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
  MPI_Reduce(_Observables(3, 0).begin(), _global_Observables(3, 0).begin(), _Nblks, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);

  if(rank == 0){

    _global_Observables(0, 0) /= double(size);
    _global_Observables(1, 0) /= double(size);
    _global_Observables(2, 0) /= double(size);
    _global_Observables(3, 0) /= double(size);

  }

}


Col <double> VMC_Sampler :: compute_errorbar(const Col <double>& block_averages) const {

  /*################################################################*/
  //  Computes the statistical uncertainties of a certain quantity
  //  by using the βblock averagingβ, where the argument represents
  //  the set of the single-block Monte Carlo averages β¨β¦β©β±Όα΅Λ‘α΅ of
  //  that quantity β¦, with j = π£,β¦,π­α΅Λ‘α΅.
  //  This calculation involves a real-valued quantity.
  /*################################################################*/

  //Function variables
  Col <double> errors(block_averages.n_elem);
  Col <double> squared_block_averages;  // β¨β¦β©β±Όα΅Λ‘α΅ β’Β β¨β¦β©β±Όα΅Λ‘α΅
  double sum_ave, sum_ave_squared;  //Storage variables

  //Block averaging method
  squared_block_averages = block_averages % block_averages;  //Armadillo Schur product
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    sum_ave  = 0.0;
    sum_ave_squared = 0.0;
    for(unsigned int j = 0; j < (block_ID + 1); j++){

      sum_ave += block_averages(j);
      sum_ave_squared += squared_block_averages(j);

    }
    sum_ave /= double(block_ID + 1);
    sum_ave_squared /= double(block_ID + 1);
    if(block_ID == 0)
      errors(block_ID) = 0.0;
    else
      errors(block_ID) = std::sqrt(std::abs(sum_ave_squared - sum_ave * sum_ave) / (double(block_ID)));

  }

  return errors;

}


Col <std::complex <double>> VMC_Sampler :: compute_errorbar(const Col <std::complex <double>>& block_averages) const {

  /*################################################################*/
  //  Computes the statistical uncertainties of a certain quantity
  //  by using the βblock averagingβ, where the argument represents
  //  the set of the single-block Monte Carlo averages β¨β¦β©β±Όα΅Λ‘α΅ of
  //  that quantity β¦, with j = π£,β¦,π­α΅Λ‘α΅.
  //  This calculation involves a complex-valued quantity.
  /*################################################################*/

  //Function variables
  Col <std::complex <double>> errors(block_averages.n_elem);
  Col <double> block_averages_re = real(block_averages);
  Col <double> block_averages_im = imag(block_averages);

  //Block averaging method, keeping real and imaginary part separated
  errors.set_real(compute_errorbar(block_averages_re));
  errors.set_imag(compute_errorbar(block_averages_im));

  return errors;

}


Col <double> VMC_Sampler :: compute_progressive_averages(const Col <double>& block_averages) const {

  /*################################################################*/
  //  Computes the progressive averages of a certain quantity
  //  by using the βblock averagingβ, where the argument represents
  //  the set of the single-block Monte Carlo averages β¨β¦β©β±Όα΅Λ‘α΅ of
  //  that quantity β¦, with j = π£,β¦,π­α΅Λ‘α΅.
  //  This calculation involves a real-valued quantity.
  /*################################################################*/

  //Function variables
  Col <double> prog_ave(_Nblks);
  double sum_ave;

  //Block averaging
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    sum_ave = 0.0;
    for(unsigned int j = 0; j < (block_ID + 1); j++)
      sum_ave += block_averages(j);
    sum_ave /= double(block_ID + 1);
    prog_ave(block_ID) = sum_ave;

  }

  return prog_ave;

}


Col <std::complex <double>> VMC_Sampler :: compute_progressive_averages(const Col <std::complex <double>>& block_averages) const {

  /*################################################################*/
  //  Computes the progressive averages of a certain quantity
  //  by using the βblock averagingβ, where the argument represents
  //  the set of the single-block Monte Carlo averages β¨β¦β©β±Όα΅Λ‘α΅ of
  //  that quantity β¦, with j = π£,β¦,π­α΅Λ‘α΅.
  //  This calculation involves a complex-valued quantity.
  /*################################################################*/

  //Function variables
  Col <std::complex <double>> prog_ave(block_averages.n_elem);
  Col <double> block_averages_re = real(block_averages);
  Col <double> block_averages_im = imag(block_averages);

  //Block averaging method, keeping real and imaginary part separated
  prog_ave.set_real(compute_progressive_averages(block_averages_re));
  prog_ave.set_imag(compute_progressive_averages(block_averages_im));

  return prog_ave;

}


void VMC_Sampler :: compute_O() {

  //Gives size
  _O.set_size(_vqs.n_alpha(), 2);

  if(!_if_shadow){

    for(unsigned int lo_ID = 0; lo_ID < _O.n_rows; lo_ID++){

      _O(lo_ID, 0) = this -> average_in_blocks(_instO_ket.row(lo_ID));  // β¨πββ©β±Όα΅Λ‘α΅
      _O(lo_ID, 1) = this -> average_in_blocks(conj(_instO_ket.row(lo_ID)));  // β¨πβββ©β±Όα΅Λ‘α΅

    }

  }
  else{

    for(unsigned int lo_ID = 0; lo_ID < _O.n_rows; lo_ID++){

      //Computes βͺπββ«β±Όα΅Λ‘α΅
      _O(lo_ID, 0).set_size(_Nblks);
      _O(lo_ID, 0).set_real(this -> Shadow_angled_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O(lo_ID, 0).set_imag(zeros(_Nblks));

      //Computes βπβββ±Όα΅Λ‘α΅
      _O(lo_ID, 1).set_size(_Nblks);
      _O(lo_ID, 1).set_real(this -> Shadow_square_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O(lo_ID, 1).set_imag(zeros(_Nblks));

    }

  }

}


void VMC_Sampler :: compute_QGTandGrad(MPI_Comm common) {

  /*#################################################################################*/
  //  ππ¨π¦π©π?π­ππ¬ πππ ππ?ππ§π­π?π¦ πππ¨π¦ππ­π«π’π πππ§π¬π¨π«.
  //  We compute stochastically the πππ defined as
  //
  //        β = πββ                                  (πβ΄π-π?π½πΆπΉβ΄π)
  //        πββ β β¨πββπββ© - β¨πβββ©β’β¨πββ©.
  //
  //        β = π + πΌβ’π½β’πΌ                            (π?π½πΆπΉβ΄π)
  //        πββ β βͺπβπββ« - βͺπββ«β’βͺπββ« - βπβββπββ
  //        πΌββ β -βπβπββ + βπβββͺπββ« - βͺπββ«βπββ
  //        where π½ is the inverse matrix of π.
  /*#################################################################################*/
  /*#################################################################################*/
  //  ππ¨π¦π©π?π­ππ¬ πππ ππ§ππ«π π² ππ«πππ’ππ§π­.
  //  We compute stochastically the Gradient which drive the optimization defined as
  //
  //        π½β β β¨β°πβββ© - β¨β°β©β’β¨πβββ©                  (πβ΄π-π?π½πΆπΉβ΄π)
  //
  //        π½α΄Ώ β π - πΌβ’π½β’π¨                           (π?π½πΆπΉβ΄π)
  //        π½α΄΅ β π¨ + πΌβ’π½β’π
  //
  //  with
  //
  //        πβ β -β¨ββ©β’βπββ + βͺπββ’β°α΄΅β« + βπββ’β°α΄Ώβ
  //        π¨β β β¨ββ©β’βͺπββ« + βπββ’β°α΄΅β - βͺπββ’β°α΄Ώβ«
  //
  //  where πΌ and π½ are introduced before in the calculation of β.
  /*#################################################################################*/

  //MPI variables for parallelization
  int rank,size;
  MPI_Comm_size(common, &size);
  MPI_Comm_rank(common, &rank);

  //Function variables
  unsigned int n_alpha = _vqs.n_alpha();
  unsigned int blk_size = std::floor(double(_Nsweeps/_Nblks));  //Sets the block length
  _mean_O.zeros(n_alpha);
  _mean_O_star.zeros(n_alpha);
  _mean_O_angled.zeros(n_alpha);
  _mean_O_square.zeros(n_alpha);
  Mat <std::complex <double>> Q(n_alpha, n_alpha, fill::zeros);
  Col <std::complex <double>> F(n_alpha, fill::zeros);
  _Q.zeros(n_alpha, n_alpha);
  _F.zeros(n_alpha);

  if(!_if_shadow){

    Col <std::complex <double>> mean_O(n_alpha);  // β¨β¨πββ©α΅Λ‘α΅β©
    Col <std::complex <double>> mean_O_star(n_alpha);  // β¨β¨πβββ©α΅Λ‘α΅β©
    std::complex <double> block_qgt, block_gradE;

    //Computes πΈ(π,πΆ) = β¨ββ© stochastically without progressive errorbars
    std::complex <double> E = mean(_Observables(0, 0));

    for(unsigned int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O(lo_ID) = mean(_O(lo_ID, 0));
      mean_O_star(lo_ID) = mean(_O(lo_ID, 1));

    }

    //Computes β = πββ stochastically without progressive errorbars
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = 0; n < n_alpha; n++){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          block_qgt = 0.0;
          for(unsigned int l = block_ID * blk_size; l < (block_ID +  1) * blk_size; l++)
            block_qgt += std::conj(_instO_ket(m, l)) * _instO_ket(n, l);  //Accumulate πββπβ in each block
          Q(m, n) += block_qgt / double(blk_size);  // β¨πβββ©α΅Λ‘α΅

        }

      }

    }
    Q /= double(_Nblks);  // β¨ββ© β β¨β¨πβββ©α΅Λ‘α΅β©

    //Computes π½ = π½β stochastically without progressive errorbars
    for(unsigned int k = 0; k < n_alpha; k++){

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_gradE = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
          block_gradE += _instObs_ket(0, l) * std::conj(_instO_ket(k, l));  //Accumulate β°πββ in each block
        F(k) += block_gradE / double(blk_size);  // β¨π½ββ©α΅Λ‘α΅

      }

    }
    F /= double(_Nblks);  // β¨π½β© β β¨β¨π½ββ©α΅Λ‘α΅β©

    MPI_Barrier(common);

    //Shares block averages among all the nodes
    MPI_Reduce(mean_O.begin(), _mean_O.begin(), n_alpha, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(mean_O_star.begin(), _mean_O_star.begin(), n_alpha, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(&E, &_E, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(Q.begin(), _Q.begin(), _Q.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(F.begin(), _F.begin(), _F.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);

    if(rank == 0){

      _mean_O /= double(size);
      _mean_O_star /= double(size);
      _E /= double(size);
      _Q /= double(size);
      _F /= double(size);

      _Q = _Q - kron(_mean_O_star, _mean_O.st());
      _F = _F - _E * mean_O_star;

    }

  }
  else{

    Col <double> mean_O_angled(n_alpha);  // β¨βͺπββ«α΅Λ‘α΅β© with reweighting correction
    Col <double> mean_O_square(n_alpha);  // β¨βπββα΅Λ‘α΅β© with reweighting correction
    Mat <double> S(n_alpha, n_alpha, fill::zeros);  // πββ β βͺπβπββ« - βͺπββ«β’βͺπββ« - βπβββπββ
    Mat <double> A(n_alpha, n_alpha, fill::zeros);  // πΌββ β -βπβπββ + βπβββͺπββ« - βͺπββ«βπββ
    Col <double> Gamma(n_alpha, fill::zeros);  // πβ β -β¨ββ©β’βπββ + βͺπββ’β°α΄΅β« + βπββ’β°α΄Ώβ
    Col <double> Omega(n_alpha, fill::zeros);  // π¨β β β¨ββ©β’βͺπββ« + βπββ’β°α΄΅β - βͺπββ’β°α΄Ώβ«
    double block_corr_angled, block_corr_square;
    double mean_cos = mean(_cosII);

    for(unsigned int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O_angled(lo_ID) = mean(real(_O(lo_ID, 0))) / mean_cos;
      mean_O_square(lo_ID) = mean(real(_O(lo_ID, 1))) / mean_cos;

    }

    //Computes πΈ(π,πΆ) = β¨ββ© stochastically without progressive errorbars
    std::complex <double> E;
    E.real(mean(real(_Observables(0, 0))) / mean_cos);  // β¨β¨ββ©α΅Λ‘α΅β© with reweighting correction
    E.imag(0.0);

    //Computes β = π + πΌβ’π½β’πΌ stochastically without progressive errorbars
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = m; n < n_alpha; n++){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          block_corr_angled = 0.0;
          block_corr_square = 0.0;
          for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

            //Accumulate πβπβ in each block (angled part)
            block_corr_angled += _instReweight(0, l) * (_instO_ket(m, l).real() * _instO_bra(n, l).real() + _instO_bra(m, l).real() * _instO_ket(n, l).real());
            //Accumulate πβπβ in each block (square part)
            if(m != n)
              block_corr_square += _instReweight(1, l) * (_instO_bra(m, l).real() * _instO_ket(n, l).real() - _instO_ket(m, l).real() * _instO_bra(n, l).real());


          }
          if(m == n)
            S(m, m) += 0.5 * block_corr_angled / double(blk_size);  //Computes the diagonal elements of S
          else{

            S(m, n) += 0.5 * block_corr_angled / double(blk_size);  //This is a symmetric matrix, so we calculate only the upper triangular matrix
            S(n, m) = S(m, n);
            A(m, n) -= 0.5 * block_corr_square / double(blk_size);  //This is an anti-symmetric matrix, so we calculate only the upper triangular matrix
            A(n, m) = (-1.0) * A(m, n);

          }

        }

      }

    }
    S /= double(_Nblks);  // β¨β¨βͺπβπββ«α΅Λ‘α΅β©β© without reweighting correction
    A /= double(_Nblks);  // β¨β¨βπβπββα΅Λ‘α΅β©β© without reweighting correction
    S /= mean_cos;
    A /= mean_cos;
    S = S - kron(mean_O_angled, mean_O_angled.t()) + kron(mean_O_square, mean_O_square.t());
    A = A + kron(mean_O_square, mean_O_angled.t()) - kron(mean_O_angled, mean_O_square.t());
    if(_if_QGT_reg)
      S = S + _eps * _I;
    Mat <double> AB = A * pinv(S);
    Q.set_real(symmatu(S + AB * A));  // β¨ββ© β β¨β¨π + πΌβ’π½β’πΌβ©α΅Λ‘α΅β©

    //Computes π½ = {π½α΄Ώ, π½α΄΅} stochastically without progressive errorbars
    for(unsigned int k = 0; k < n_alpha; k++){  //Computes β¨πββ©α΅Λ‘α΅

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate πββ’β°α΄΅ in each block (angled part)
          block_corr_angled += _instReweight(0, l) * (_instO_ket(k, l).real() * _instObs_bra(0, l).imag() + _instO_bra(k, l).real() * _instObs_ket(0, l).imag());
          //Accumulate πββ’β°α΄Ώ in each block (square part)
          block_corr_square += _instReweight(1, l) * (_instO_bra(k, l).real() * _instObs_ket(0, l).real() - _instO_ket(k, l).real() * _instObs_bra(0, l).real());

        }
        Gamma(k) += 0.5 * (block_corr_angled + block_corr_square) / double(blk_size);

      }

    }
    for(unsigned int k = 0; k < n_alpha; k++){  //Computes β¨π¨ββ©α΅Λ‘α΅

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate πββ’β°α΄Ώ in each block (angled part)
          block_corr_angled += _instReweight(0, l) * (_instO_ket(k, l).real() * _instObs_bra(0, l).real() + _instO_bra(k, l).real() * _instObs_ket(0, l).real());
          //Accumulate πββ’β°α΄΅ in each block (square part)
          block_corr_square += _instReweight(1, l) * (_instO_bra(k, l).real() * _instObs_ket(0, l).imag() - _instO_ket(k, l).real() * _instObs_bra(0, l).imag());

        }
        Omega(k) += 0.5 * (block_corr_square - block_corr_angled) / double(blk_size);

      }

    }
    Gamma /= double(_Nblks);  // β¨β¨πββ©α΅Λ‘α΅β© without reweighting correction
    Omega /= double(_Nblks);  // β¨β¨π¨ββ©α΅Λ‘α΅β© without reweighting correction
    Gamma /= mean_cos;
    Omega /=  mean_cos;
    Gamma -= E.real() * mean_O_square;  // β¨πββ© with reweighting correction
    Omega += E.real() * mean_O_angled;  // β¨π¨ββ© with reweighting correction
    F.set_real(Gamma - AB * Omega);  // β¨π½α΄Ώβ© β β¨β¨π - πΌβ’π½β’π¨β©α΅Λ‘α΅β©
    F.set_imag(Omega + AB * Gamma);  // β¨π½α΄΅β© β β¨β¨π¨ + πΌβ’π½β’πβ©α΅Λ‘α΅β©

    MPI_Barrier(common);

    //Shares block averages among all the nodes
    MPI_Reduce(mean_O_angled.begin(), _mean_O_angled.begin(), n_alpha, MPI_DOUBLE, MPI_SUM, 0, common);
    MPI_Reduce(mean_O_square.begin(), _mean_O_square.begin(), n_alpha, MPI_DOUBLE, MPI_SUM, 0, common);
    MPI_Reduce(&E, &_E, 1, MPI_DOUBLE, MPI_SUM, 0, common);
    MPI_Reduce(Q.begin(), _Q.begin(), _Q.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(F.begin(), _F.begin(), _F.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);

    if(rank == 0){

      _mean_O_angled /= double(size);
      _mean_O_square /= double(size);
      _E.imag(0.0);
      _E /= double(size);
      _Q /= double(size);
      _F /= double(size);

    }

  }

}


void VMC_Sampler :: QGT_Check(int rank) {

  if(rank == 0){

    if(_if_shadow){

      if(real(_Q).is_symmetric() == false)
        std::cerr << "  ##EstimationError: the Quantum Geometric Tensor must be symmetric!" << std::endl;
      else
        return;

    }
    else{

      if(_Q.is_hermitian() == false)
        std::cerr << "  ##EstimationError: the Quantum Geometric Tensor must be hermitian!" << std::endl;
      else
        return;

    }

  }

}


void VMC_Sampler :: is_asymmetric(const Mat <double>& A) const {

  unsigned int failed = 0;

  for(unsigned int m = 0; m < A.n_rows; m++){

    for(unsigned int n = m; n < A.n_cols; n++){

      if(A(m, n) != (-1.0) * A(n, m))
        failed++;

    }

  }

  if(failed != 0)
    std::cout << "The matrix is not anti-Symmetric." << std::endl;
  else
    return;

}


void VMC_Sampler :: Reset_Moves_Statistics() {

  _N_accepted_visible = 0;
  _N_proposed_visible = 0;
  _N_accepted_ket = 0;
  _N_proposed_ket = 0;
  _N_accepted_bra = 0;
  _N_proposed_bra = 0;
  _N_accepted_equal_site = 0;
  _N_proposed_equal_site = 0;
  _N_accepted_visible_nn_site = 0;
  _N_proposed_visible_nn_site = 0;
  _N_accepted_hidden_nn_site = 0;
  _N_proposed_hidden_nn_site = 0;

}


bool VMC_Sampler :: RandFlips_visible(Mat <int>& flipped_site, unsigned int Nflips) {

  /*#############################################################################*/
  //  Random spin flips for the visible quantum degrees of freedom.
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the physical quantum degrees of freedom only.
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting πππ₯π’π©π¬ lattice sites
  //  and create a new quantum configuration
  //
  //        |π?βΏα΅Κ·β© = |πβΏα΅Κ· π‘ π‘Λβ©
  //
  //  representing it as the list of indeces of the visible flipped
  //  lattice sites (see π¦π¨πππ₯.ππ©π©).
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Initializes the new configuration according to |π²|
  if(_H.dimensionality() == 1){  //π² Ο΅ β€α΅, π½ = π

    flipped_site.set_size(Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++)
      flipped_site(j, 0) = _rnd.Rannyu_INT(0, _Nspin-1);  //Choose a random spin to flip

  }
  else{  //π² Ο΅ β€α΅, π½ = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_site);
  if(test.n_elem == flipped_site.n_rows)
    return true;
  else
    return false;

}


void VMC_Sampler :: Move_visible(unsigned int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen πππ
  //  in which only the visible variables have been tried
  //  to move, i.e.
  //
  //        |π?βΏα΅Κ·β© = |πβΏα΅Κ· π‘ π‘Λβ©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of spins.
  //  In particular, it first randomly selects πππ₯π’π©π¬ lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  πππ§πππ₯π’π©π¬_π―π’π¬π’ππ₯π is true, then it decides whether or not
  //  to accept |π?βΏα΅Κ·β© through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_visible(_flipped_site, Nflips)){

    _N_proposed_visible++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _hidden_ket, _flipped_ket_site,
                                             _hidden_bra, _flipped_bra_site,
                                             "visible");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_visible++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(unsigned int fs_row = 0; fs_row < _flipped_site.n_rows; fs_row++){  //Move the quantum spin configuration

        if(_H.dimensionality() == 1)  //π² Ο΅ β€α΅, π½ = π
          _configuration(0, _flipped_site(fs_row, 0)) *= -1;
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

    }

  }
  else
    return;

}


bool VMC_Sampler :: RandFlips_hidden(Mat <int>& flipped_hidden_site, unsigned int Nflips) {

  /*##############################################################################*/
  //  Random spin flips for the hidden quantum degrees of freedom (ket or bra).
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the auxiliary quantum degrees of freedom in the ket configuration only.
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting πππ₯π’π©π¬ lattice sites
  //  and create a new quantum configuration
  //
  //        |π?βΏα΅Κ·β© = |π π‘βΏα΅Κ· π‘Λβ©
  //                or
  //        |π?βΏα΅Κ·β© = |π π‘ π‘ΛβΏα΅Κ·β©
  //
  //  representing it as the list of indeces of the hidden flipped
  //  lattice sites (see π¦π¨πππ₯.ππ©π©).
  //  The function prevents from flipping the same site more than once.
  /*##############################################################################*/

  //Initializes the new configuration according to |π²|
  if(_H.dimensionality() == 1){  //π² Ο΅ β€α΅, π½ = π

    flipped_hidden_site.set_size(Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++)
      flipped_hidden_site(j, 0) = _rnd.Rannyu_INT(0, _Nhidden-1);  //Choose a random spin to flip

  }
  else{  //π² Ο΅ β€α΅, π½ = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_hidden_site);
  if(test.n_elem == flipped_hidden_site.n_rows)
    return true;
  else
    return false;

}


void VMC_Sampler :: Move_ket(unsigned int Nflips) {

  /*##################################################################*/
  //  This function proposes a new configuration for the chosen πππ
  //  in which only the hidden variables (ket) have been tried
  //  to move, i.e.
  //
  //        |π?βΏα΅Κ·β© = |π π‘βΏα΅Κ· π‘Λβ©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of auxiliary spins.
  //  In particular, it first randomly selects πππ₯π’π©π¬ hidden lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  πππ§πππ₯π’π©π¬_π‘π’ππππ§ is true, then it decides whether or not
  //  to accept |π?βΏα΅Κ·β© through the Metropolis-Hastings test.
  /*##################################################################*/

  if(this -> RandFlips_hidden(_flipped_ket_site, Nflips)){

    _N_proposed_ket++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _hidden_ket, _flipped_ket_site,
                                             _hidden_bra, _flipped_bra_site,
                                             "ket");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_ket++;
      //_vqs.Update_on_Config(_configuration, _flipped_site);
      for(unsigned int fs_row = 0; fs_row < _flipped_ket_site.n_rows; fs_row++){  //Move the quantum ket configuration

        if(_H.dimensionality() == 1)  //π² Ο΅ β€α΅, π½ = π
          _hidden_ket(0, _flipped_ket_site(fs_row, 0)) *= -1;
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Move_bra(unsigned int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen πππ
  //  in which only the hidden variables (bra) have been tried
  //  to move, i.e.
  //
  //        |π?βΏα΅Κ·β© = |π π‘ π‘ΛβΏα΅Κ·β©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of auxiliary spins.
  //  In particular, it first randomly selects πππ₯π’π©π¬ hidden lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  πππ§πππ₯π’π©π¬_ππ«π is true, then it decides whether or not
  //  to accept |π?βΏα΅Κ·β© through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_hidden(_flipped_bra_site, Nflips)){

    _N_proposed_bra++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _hidden_ket, _flipped_ket_site,
                                             _hidden_bra, _flipped_bra_site,
                                             "bra");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_bra++;
      //_vqs.Update_on_Config(_configuration, _flipped_site);
      for(unsigned int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum bra configuration

        if(_H.dimensionality() == 1)  //π² Ο΅ β€α΅, π½ = π
          _hidden_bra(0, _flipped_bra_site(fs_row, 0)) *= -1;
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

    }

  }
  else
    return;

}


void  VMC_Sampler :: Move_equal_site(unsigned int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen πππ
  //  in which the visible and the hidden variables have been
  //  tried to move, i.e.
  //
  //        |π?βΏα΅Κ·β© = |πβΏα΅Κ· π‘βΏα΅Κ· π‘ΛβΏα΅Κ·β©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of spins on
  //  π¨π§ π­π‘π π¬ππ¦π π₯ππ­π­π’ππ π¬iπ­ππ¬.
  //  In particular, it first randomly selects πππ₯π’π©π¬ lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  πππ§πππ₯π’π©π¬_π―π’π¬π’ππ₯π is true, then it decides whether or not
  //  to accept |π?βΏα΅Κ·β© through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_visible(_flipped_site, Nflips)){

    _N_proposed_equal_site++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _hidden_ket, _flipped_ket_site,
                                             _hidden_bra, _flipped_bra_site,
                                             "equal site");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_equal_site++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(unsigned int fs_row = 0; fs_row < _flipped_site.n_rows; fs_row++){  //Move the quantum configuration

        if(_H.dimensionality() == 1){  //π² Ο΅ β€α΅, π½ = π

          _configuration(0, _flipped_site(fs_row, 0)) *= -1;
          _hidden_ket(0, _flipped_site(fs_row, 0)) *= -1;
          _hidden_bra(0, _flipped_site(fs_row, 0)) *= -1;

        }
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

    }

  }
  else
    return;

}


bool VMC_Sampler :: RandFlips_visible_nn_site(Mat <int>& flipped_visible_nn_site, unsigned int Nflips) {

  /*#############################################################################*/
  //  Random spin flips for the visible quantum degrees of freedom.
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the physical quantum degrees of freedom only.
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting πππ₯π’π©π¬ lattice sites
  //  and create a new quantum configuration
  //
  //        |π?βΏα΅Κ·β© = |πβΏα΅Κ· π‘ π‘Λβ©
  //
  //  representing it as the list of indeces of the visible flipped
  //  lattice sites (see π¦π¨πππ₯.ππ©π©).
  //  If a certain lattice site is selected, π’π­π¬ ππ’π«π¬π­ π«π’π π‘π­ π§πππ«ππ¬π­ π§ππ’π π‘ππ¨π«
  //  site it is automatically added to the list of flipped sites.
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Function variables
  unsigned int index_site;

  //Initializes the new configuration according to |π²|
  if(_H.dimensionality() == 1){  //π² Ο΅ β€α΅, π½ = π

    flipped_visible_nn_site.set_size(2*Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++){

      if(_H.if_pbc())
        index_site = _rnd.Rannyu_INT(0, _Nspin-1);
      else
        index_site  = _rnd.Rannyu_INT(0, _Nspin-2);
      flipped_visible_nn_site(j, 0) = index_site;  //Choose a random spin to flip

      //Adds the right nearest neighbor lattice site
      if(_H.if_pbc()){

        if(index_site == _Nspin-1)
          flipped_visible_nn_site(j+1, 0) = 0;  //Pbc
        else
          flipped_visible_nn_site(j+1, 0) = index_site + 1;

      }
      else
        flipped_visible_nn_site(j+1) = index_site + 1;

    }

  }
  else{  //π² Ο΅ β€α΅, π½ = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_visible_nn_site);
  if(test.n_elem == flipped_visible_nn_site.n_rows)
    return true;
  else
    return false;

}


void VMC_Sampler :: Move_visible_nn_site(unsigned int Nflips) {

   /*###############################################################*/
  //  This function proposes a new configuration for the chosen πππ
  //  in which only the visible variables have been tried
  //  to move, i.e.
  //
  //        |π?βΏα΅Κ·β© = |πβΏα΅Κ· π‘ π‘Λβ©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of spins
  //  with their respective π«π’π π‘π­ π§πππ«ππ¬π­ π§ππ’π π‘ππ¨π« lattice site.
  //  In particular, it first randomly selects πππ₯π’π©π¬ lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  πππ§πππ₯π’π©π¬_π―π’π¬π’ππ₯π_π§π§_π¬π’π­π is true, then it decides whether or not
  //  to accept |π?βΏα΅Κ·β© through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_visible_nn_site(_flipped_site, Nflips)){

    _N_proposed_visible_nn_site++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _hidden_ket, _flipped_ket_site,
                                             _hidden_bra, _flipped_bra_site,
                                             "visible");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_visible_nn_site++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(unsigned int fs_row = 0; fs_row < _flipped_site.n_rows; fs_row++){  //Move the quantum spin configuration

        if(_H.dimensionality() == 1)  //π² Ο΅ β€α΅, π½ = π
          _configuration(0, _flipped_site(fs_row, 0)) *= -1;
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

    }

  }
  else
    return;

}


bool VMC_Sampler :: RandFlips_hidden_nn_site(Mat <int>& flipped_ket_site, Mat <int>& flipped_bra_site, unsigned int Nflips) {

  /*#############################################################################*/
  //  Random spin flips for the hidden quantum degrees of freedom.
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the hidden quantum degrees of freedom only (both ket and bra).
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting πππ₯π’π©π¬ lattice sites
  //  and create a new quantum configuration
  //
  //        |π?βΏα΅Κ·β© = |π π‘βΏα΅Κ· π‘ΛβΏα΅Κ·β©
  //
  //  representing it as the list of indeces of the hidden flipped
  //  lattice sites (see π¦π¨πππ₯.ππ©π©).
  //  If a certain lattice site is selected, π’π­π¬ ππ’π«π¬π­ π«π’π π‘π­ π§πππ«ππ¬π­ π§ππ’π π‘ππ¨π«
  //  site it is automatically added to the list of flipped sites.
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Function variables
  unsigned int index_site_ket;
  unsigned int index_site_bra;

  //Initializes the new configuration according to |π²|
  if(_H.dimensionality() == 1){  //π² Ο΅ β€α΅, π½ = π

    flipped_ket_site.set_size(2*Nflips, 1);
    flipped_bra_site.set_size(2*Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++){

      if(_H.if_pbc()){

          index_site_ket = _rnd.Rannyu_INT(0, _Nspin-1);
          index_site_bra = _rnd.Rannyu_INT(0, _Nspin-1);

      }
      else{

        index_site_ket  = _rnd.Rannyu_INT(0, _Nspin-2);
        index_site_bra = _rnd.Rannyu_INT(0, _Nspin-2);

      }
      flipped_ket_site(j, 0) = index_site_ket;  //Choose a random spin to flip
      flipped_bra_site(j, 0) = index_site_bra;  //Choose a random spin to flip

      //Adds the right nearest neighbor lattice site
      if(_H.if_pbc()){

        if(index_site_ket == _Nspin-1)
          flipped_ket_site(j+1, 0) = 0;  //Pbc
        if(index_site_bra == _Nspin-1)
          flipped_bra_site(j+1, 0) = 0;  //Pbc
        else{

          flipped_ket_site(j+1, 0) = index_site_ket + 1;
          flipped_bra_site(j+1, 0) = index_site_bra + 1;

        }

      }
      else{

        flipped_ket_site(j+1) = index_site_ket + 1;
        flipped_bra_site(j+1) = index_site_bra + 1;

      }

    }

  }
  else{  //π² Ο΅ β€α΅, π½ = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test_ket = find_unique(flipped_ket_site);
  uvec test_bra = find_unique(flipped_bra_site);
  if(test_ket.n_elem == flipped_ket_site.n_rows && test_bra.n_elem == flipped_bra_site.n_rows)
    return true;
  else
    return false;

}


void VMC_Sampler :: Move_hidden_nn_site(unsigned int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen πππ
  //  in which only the hidden variables (both ket and bra)
  //  have been tried to move, i.e.
  //
  //        |π?βΏα΅Κ·β© = |π π‘βΏα΅Κ· π‘ΛβΏα΅Κ·β©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of auxiliary spins
  //  with their respective π«π’π π‘π­ π§πππ«ππ¬π­ π§ππ’π π‘ππ¨π« lattice site.
  //  In particular, it first randomly selects πππ₯π’π©π¬ hidden lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  πππ§πππ₯π’π©π¬_π‘π’ππππ§_π§π§_π¬π’π­π is true, then it decides whether or not
  //  to accept |π?βΏα΅Κ·β©through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_hidden_nn_site(_flipped_ket_site, _flipped_bra_site, Nflips)){

    _N_proposed_hidden_nn_site++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _hidden_ket, _flipped_ket_site,
                                             _hidden_bra, _flipped_bra_site,
                                             "braket");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_hidden_nn_site++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(unsigned int fs_row = 0; fs_row < _flipped_ket_site.n_rows; fs_row++){  //Move the quantum ket configuration

        if(_H.dimensionality() == 1)  //π² Ο΅ β€α΅, π½ = π
          _hidden_ket(0, _flipped_ket_site(fs_row, 0)) *= -1;
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

      for(unsigned int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum bra configuration

        if(_H.dimensionality() == 1)  //π² Ο΅ β€α΅, π½ = π
          _hidden_bra(0, _flipped_bra_site(fs_row, 0)) *= -1;
        else{  //π² Ο΅ β€α΅, π½ = π

          /*
            .........
            .........
            .........
          */

        }

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Move() {

  /*#################################################################*/
  //  This function proposes a new configuration for the chosen πππ,
  //
  //        |SβΏα΅Κ·β© = |πβΏα΅Κ· π‘βΏα΅Κ· π‘ΛβΏα΅Κ·β©
  //
  //  by flipping a certain (given) number πππ₯π’π©π¬ of spins.
  //  In particular, it first randomly selects πππ₯π’π©π¬ lattice
  //  sites to flip. The selected sites will be in general different
  //  for the three different types of variables (π, π‘, and π‘Λ).
  //  This function is a combination of the previously defined Monte
  //  Carlo moves.
  /*#################################################################*/

  //Check on the probability
  if(_p_equal_site < 0 || _p_equal_site > 1 || _p_visible_nn < 0 || _p_visible_nn > 1 || _p_hidden_nn < 0 || _p_hidden_nn > 1){

    std::cerr << " ##ValueError: the function options MUST be a probability!" << std::endl;
    std::cerr << "   Failed to move the system configuration." << std::endl;
    std::abort();

  }

  //Moves with a certain probability
  this -> Move_visible(_Nflips);
  if(_if_shadow == true && _if_hidden_off == false){

    this -> Move_ket(_Nflips);
    this -> Move_bra(_Nflips);
    if(_rnd.Rannyu() < _p_equal_site)
      this -> Move_equal_site(_Nflips);
    if(_rnd.Rannyu() < _p_hidden_nn)
      this -> Move_hidden_nn_site(_Nflips);

  }
  if(_rnd.Rannyu() < _p_visible_nn)
    this -> Move_visible_nn_site(_Nflips);

}


void VMC_Sampler :: Make_Sweep() {

  for(unsigned int n_bunch = 0; n_bunch < _M; n_bunch++)
    this -> Move();

}


void VMC_Sampler :: VMC_Step(MPI_Comm common) {

  /*###############################################################################################*/
  //  Runs the single optimization step.
  //  We perform the single Variational Monte Carlo optimization run using
  //  the following parameters:
  //
  //        β’ NΜ²Λ’Μ²Κ·Μ²α΅Μ²α΅Μ²α΅Μ²: is the number of Monte Carlo sweeps.
  //                  In each single MC sweep a bunch of spins is considered,
  //                  randomly chosen and whose dimension is expressed by the variable NΜ²αΆ Μ²Λ‘Μ²β±Μ²α΅Μ²Λ’Μ²,
  //                  and it is tried to flip this bunch of spins with the probability defined
  //                  by the Metropolis-Hastings algorithm; this operation is repeated a certain
  //                  number of times in the single sweep, where this certain number is defined
  //                  by the variables MΜ²; once the new proposed configuration is accepted or not,
  //                  instantaneous quantum properties are measured on that state, and the single
  //                  sweep ends; different Monte Carlo moves are applied in different situations,
  //                  involving all or only some of the visible and/or hidden variables;
  //
  //        β’Β eΜ²qΜ²α΅Μ²β±Μ²α΅Μ²α΅Μ²: is the number of Monte Carlo steps, i.e. the number
  //                  of sweeps to be employed in the thermalization phase
  //                  of the system (i.e., the phase in which new quantum
  //                  configurations are sampled but nothing is measured;
  //
  //        β’ NΜ²α΅Μ²Λ‘Μ²α΅Μ²Λ’Μ²: is the number of blocks to be used in the estimation of the
  //                 Monte Carlo quantum averages and uncertainties of the observables
  //                 via the Blocking method;
  //
  //  The single VMC run allows us to move a single step in the variational
  //  parameter optimization procedure.
  /*###############################################################################################*/

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  //Initialization and Equilibration
  if(_if_restart_from_config)
    this -> Init_Config(_configuration, _hidden_ket, _hidden_bra);
  else
    this -> Init_Config();
  for(unsigned int eq_step = 0; eq_step < _Neq; eq_step++)
    this -> Make_Sweep();

  //Monte Carlo measurement
  for(unsigned int mcmc_step = 0; mcmc_step < _Nsweeps; mcmc_step++){  //The Monte Carlo Markov Chain

    this -> Make_Sweep();  //Samples a new system configuration |π?βΏα΅Κ·β© (i.e. a new point of the mcmc)
    this -> Measure();  //Measure quantum properties on the new sampled system configuration |π?βΏα΅Κ·β©
    this -> Write_MCMC_Config(mcmc_step, rank);  //Records the sampled |π?βΏα΅Κ·β©

  }

  //Computes the quantum averages
  this -> Estimate(common);

}


void VMC_Sampler :: Euler(MPI_Comm common) {

  /*#########################################################################*/
  //  Updates the variational parameters (π,πΆ) according to the choosen
  //  πππππ equations of motion through the Euler integration method.
  //  The equations for the parameters optimization are:
  //
  //        ==================
  //          πβ΄π-π?π½πΆπΉβ΄π
  //        ==================
  //          β’ ππ¦ππ π’π§ππ«π²-π­π’π¦π ππ²π§ππ¦π’ππ¬ (ππππππ)
  //              π(Ο)β’πΆΜ(Ο) = - π½(Ο)
  //          β’ ππππ₯-π­π’π¦π ππ²π§ππ¦π’ππ¬ (πππππ)
  //              π(π‘)β’πΆΜ(π‘) =  - π β’ π½(π‘)
  //
  //        ============
  //          π?π½πΆπΉβ΄π
  //        ============
  //          β’ ππ¦ππ π’π§ππ«π²-π­π’π¦π ππ²π§ππ¦π’ππ¬ (ππππππ)
  //              β(Ο) β’ πΆΜα΄Ώ(Ο) = π½α΄΅(Ο)
  //              β(Ο) β’ πΆΜα΄΅(Ο) = - π½α΄Ώ(Ο)
  //              πΜα΄Ώ(Ο) = - πΆΜα΄Ώ(Ο) β’ βͺπβ« - πΆΜα΄΅(Ο) β’ βπβ - β¨ββ©
  //              πΜα΄΅(Ο) = + πΆΜα΄Ώ(Ο) β’ βπβ - πΆΜα΄΅(Ο) β’ βͺπβ«
  //          β’ ππππ₯-π­π’π¦π ππ²π§ππ¦π’ππ¬ (πππππ)
  //              β(π‘) β’ πΆΜα΄Ώ(π‘) = π½α΄Ώ(π‘)
  //              β(π‘) β’ πΆΜα΄΅(π‘) = π½α΄΅(π‘)
  //              πΜα΄Ώ(π‘) = - πΆΜα΄Ώ(π‘) β’ βͺπβ« - πΆΜα΄΅(π‘) β’ βπβ
  //              πΜα΄΅(π‘) = + πΆΜα΄Ώ(π‘) β’ βπβ - πΆΜα΄΅(π‘) β’ βͺπβ« - β¨ββ©
  //
  //  where in the πβ΄π-π?π½πΆπΉβ΄π case we assume π = 0.
  //  In the Euler method we obtain the new parameters in the following way:
  //  πΎπ»
  //
  //        πΌΜ(π‘) = π»{πΌ(π‘)}
  //
  //  ππ½β―π
  //
  //        πΌ(π‘+ππ‘) = πΌ(π‘) + ππ‘ β’ π»{πΌ(π‘)}
  //
  //  where π»{πΌ(π‘)} is numerically integrated by using the π¬π¨π₯π―π() method
  //  of the C++ Armadillo library.
  /*#########################################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  πβ΄π-π?π½πΆπΉβ΄π  */
    /*################*/
    if(!_if_shadow){

      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      std::complex <double> new_phi;
      if(rank == 0){

        //Function variables
        Col <std::complex <double>> alpha_dot;
        std::complex <double> phi_dot;

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            alpha_dot = solve(_Q + _eps * _I, - _i * _F);
          else
            alpha_dot = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero())
            phi_dot = as_scalar(- alpha_dot.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            alpha_dot = solve(_Q + _eps * _I, - _F);
          else
            alpha_dot = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero())
            phi_dot = _i * as_scalar(alpha_dot.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = _vqs.alpha() + _delta * alpha_dot;
        if(_vqs.if_phi_neq_zero()) new_phi = _vqs.phi() + _delta * phi_dot;

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

      /*############*/
     /*  π?π½πΆπΉβ΄π  */
    /*############*/
    else{

      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      std::complex <double> new_phi;
      if(rank == 0){

        //Function variables
        Col <double> alpha_dot_re;
        Col <double> alpha_dot_im;
        double phi_dot_re;
        double phi_dot_im;

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            alpha_dot_re = solve(real(_Q) + _eps * _I, real(_F));
            alpha_dot_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            alpha_dot_re = solve(real(_Q), real(_F));
            alpha_dot_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_dot_re = as_scalar(- alpha_dot_re.t() * _mean_O_angled - alpha_dot_im.t() * _mean_O_square);
            phi_dot_im = as_scalar(alpha_dot_re.t() * _mean_O_square - alpha_dot_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            alpha_dot_re = solve(real(_Q) + _eps * _I, imag(_F));
            alpha_dot_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            alpha_dot_re = solve(real(_Q), imag(_F));
            alpha_dot_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_dot_re = as_scalar(- alpha_dot_re.t() * _mean_O_angled - alpha_dot_im.t() * _mean_O_square) - _E.real();
            phi_dot_im = as_scalar(alpha_dot_re.t() * _mean_O_square - alpha_dot_im.t() * _mean_O_angled);

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(real(_vqs.alpha()) + _delta * alpha_dot_re);
        new_alpha.set_imag(imag(_vqs.alpha()) + _delta * alpha_dot_im);
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(_vqs.phi().real() + _delta * phi_dot_re);
          new_phi.imag(_vqs.phi().imag() + _delta * phi_dot_im);

        }

      }

      //Updates parameters of all nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi_real(new_phi.real());
        _vqs.set_phi_imag(new_phi.real());

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Heun(MPI_Comm common) {

  /*###############################################################*/
  //  The Heun method is a so-called predictor-corrector method,
  //  which achieves a second order accuracy.
  //  In the Heun method we first obtain the auxiliary updates
  //  of the variational parameters
  //
  //        πΆΜ(π‘ + πΏπ‘) = πΆ(π‘) + πΏπ‘β’π»{πΌ(π‘)}
  //
  //  as in the Euler method. We remember that
  //
  //        πΌΜ(π‘) = π»{πΌ(π‘)}.
  //
  //  These updates are used to performed a second optimization
  //  step via the πππ_ππ­ππ©() function, and then obtained a second
  //  order updates as
  //
  //        πΆ(π‘ + πΏπ‘) = πΆ(π‘) + 1/2β’πΏπ‘β’[π»{πΌ(π‘)} + f{πΆΜ(π‘ + πΏπ‘)}].
  //
  //  The first πππ step in this integration is performed in the
  //  main program.
  /*###############################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  πβ΄π-π?π½πΆπΉβ΄π  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_t = _vqs.alpha();  // πΆ(π‘)
      Col <std::complex <double>> alpha_dot_t;  // πΌΜ(π‘) = π»{πΌ(π‘)}
      Col <std::complex <double>> alpha_dot_tilde_t;  // f{πΆΜ(π‘ + πΏπ‘)}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      std::complex <double> phi_t = _vqs.phi();
      std::complex <double> phi_dot_t;
      std::complex <double> phi_dot_tilde_t;
      std::complex <double> new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            alpha_dot_t = solve(_Q + _eps * _I, - _i * _F);
          else
            alpha_dot_t = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero())
            phi_dot_t = as_scalar(- alpha_dot_t.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            alpha_dot_t = solve(_Q + _eps * _I, - _F);
          else
            alpha_dot_t = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero())
            phi_dot_t = _i * as_scalar(alpha_dot_t.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + _delta * alpha_dot_t;
        if(_vqs.if_phi_neq_zero()) new_phi = phi_t + _delta * phi_dot_t;

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second πππ step at time π‘ + πΏπ‘
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            alpha_dot_tilde_t = solve(_Q + _eps * _I, - _i * _F);
          else
            alpha_dot_tilde_t = solve(_Q, - _i * _F);
          if(_vqs.if_phi_neq_zero())
            phi_dot_tilde_t = as_scalar(- alpha_dot_tilde_t.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            alpha_dot_tilde_t = solve(_Q + _eps * _I, - _F);
          else
            alpha_dot_tilde_t = solve(_Q, - _F);
          if(_vqs.if_phi_neq_zero())
            phi_dot_tilde_t = _i * as_scalar(alpha_dot_tilde_t.st() * _mean_O) - _E.real();

        }

        //Final update of the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * (alpha_dot_t + alpha_dot_tilde_t);  // πΆ(π‘ + πΏπ‘)
        if(_vqs.if_phi_neq_zero()) new_phi = phi_t + 0.5 * _delta * (phi_dot_t + phi_dot_tilde_t);

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

      /*############*/
     /*  π?π½πΆπΉβ΄π  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // πα΄Ώ(π‘)
      double phi_t_im = _vqs.phi().imag();  // πα΄΅(π‘)
      Col <double> alpha_t_re = real(_vqs.alpha());  // πΆα΄Ώ(π‘)
      Col <double> alpha_t_im = imag(_vqs.alpha());  // πΆα΄΅(π‘)
      Col <double> alpha_dot_t_re;  // πΌΜα΄Ώ(π‘) = π»{πΌα΄Ώ(π‘)}
      Col <double> alpha_dot_t_im;  // πΌΜα΄΅(π‘) = π»{πΌα΄΅(π‘)}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      double phi_dot_t_re;  // πΜα΄Ώ(π‘)
      double phi_dot_t_im;  // πΜα΄΅(π‘)
      Col <double> alpha_dot_tilde_t_re;
      Col <double> alpha_dot_tilde_t_im;
      double phi_dot_tilde_re;
      double phi_dot_tilde_im;
      std::complex <double> new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            alpha_dot_t_re = solve(real(_Q) + _eps * _I, real(_F));
            alpha_dot_t_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            alpha_dot_t_re = solve(real(_Q), real(_F));
            alpha_dot_t_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_dot_t_re = as_scalar(- alpha_dot_t_re.t() * _mean_O_angled - alpha_dot_t_im.t() * _mean_O_square);
            phi_dot_t_im = as_scalar(alpha_dot_t_re.t() * _mean_O_square - alpha_dot_t_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            alpha_dot_t_re = solve(real(_Q) + _eps * _I, imag(_F));
            alpha_dot_t_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            alpha_dot_t_re = solve(real(_Q), imag(_F));
            alpha_dot_t_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_dot_t_re = as_scalar(- alpha_dot_t_re.t() * _mean_O_angled - alpha_dot_t_im.t() * _mean_O_square) - _E.real();
            phi_dot_t_im = as_scalar(alpha_dot_t_re.t() * _mean_O_square - alpha_dot_t_im.t() * _mean_O_angled);

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + _delta * alpha_dot_t_re);
        new_alpha.set_imag(alpha_t_im + _delta * alpha_dot_t_im);
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(phi_t_re + _delta * phi_dot_t_re);
          new_phi.imag(phi_t_im + _delta * phi_dot_t_im);

        }

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second πππ step at time π‘ + πΏπ‘
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            alpha_dot_tilde_t_re = solve(real(_Q) + _eps * _I, real(_F));
            alpha_dot_tilde_t_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            alpha_dot_tilde_t_re = solve(real(_Q), real(_F));
            alpha_dot_tilde_t_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_dot_tilde_re = as_scalar(- alpha_dot_tilde_t_re.t() * _mean_O_angled - alpha_dot_tilde_t_im.t() * _mean_O_square);
            phi_dot_tilde_im = as_scalar(alpha_dot_tilde_t_re.t() * _mean_O_square - alpha_dot_tilde_t_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            alpha_dot_tilde_t_re = solve(real(_Q) + _eps * _I, imag(_F));
            alpha_dot_tilde_t_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            alpha_dot_tilde_t_re = solve(real(_Q), imag(_F));
            alpha_dot_tilde_t_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_dot_tilde_re = as_scalar(- alpha_dot_tilde_t_re.t() * _mean_O_angled - alpha_dot_tilde_t_im.t() * _mean_O_square) - _E.real();
            phi_dot_tilde_im = as_scalar(alpha_dot_tilde_t_re.t() * _mean_O_square - alpha_dot_tilde_t_im.t() * _mean_O_angled);

          }

        }

        //Final update of the variational parameters
        new_alpha.set_real(alpha_t_re + 0.5 * _delta * (alpha_dot_t_re + alpha_dot_tilde_t_re));
        new_alpha.set_imag(alpha_t_im + 0.5 * _delta * (alpha_dot_t_im + alpha_dot_tilde_t_im));
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(phi_t_re + 0.5 * _delta * (phi_dot_t_re + phi_dot_tilde_re));
          new_phi.imag(phi_t_im + 0.5 * _delta * (phi_dot_t_im + phi_dot_tilde_im));

        }

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: RK4(MPI_Comm common) {

  /*############################################################################*/
  //  The fourth order Runge Kutta method (πππ) is a one-step explicit
  //  method that achieves a fourth-order accuracy by evaluating the
  //  function π»{πΌ(π‘)} four times at each time-step.
  //  It is defined as follows:
  //
  //        πΌβ(π‘ + πΏβ) = πΌβ(π‘) + π£/π¨β’πΏββ’[ΞΊπ£ + ΞΊπ€ + ΞΊπ₯ + ΞΊπ¦]
  //
  //  where we have defined
  //
  //        ΞΊπ£ = π»{πΌ(π‘)}
  //        ΞΊπ€ = π»{πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ£}
  //        ΞΊπ₯ = π»{πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ€}
  //        ΞΊπ¦ = π»{πΌ(π‘) + πΏββ’ΞΊπ₯}.
  //
  //  We remember that
  //
  //        πΌΜ(π‘) = π»{πΌ(π‘)}.
  //
  //  The first πππ step in this integration is performed in the main program.
  /*############################################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  πβ΄π-π?π½πΆπΉβ΄π  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_t = _vqs.alpha();  // πΆ(π‘)
      std::complex <double> phi_t = _vqs.phi();  // π(π‘)
      Col <std::complex <double>> k1;  // ΞΊπ£ = π»{πΌ(π‘)}
      Col <std::complex <double>> k2;  // ΞΊπ€ = π»{πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ£}
      Col <std::complex <double>> k3;  // ΞΊπ₯ = π»{πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ€}
      Col <std::complex <double>> k4;  // ΞΊπ¦ = π»{πΌ(π‘) + πΏββ’ΞΊπ₯}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      std::complex <double> phi_k1, phi_k2, phi_k3, phi_k4;
      std::complex <double> new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            k1 = solve(_Q + _eps * _I, - _i * _F);
          else
            k1 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero())
            phi_k1 = as_scalar(- k1.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            k1 = solve(_Q + _eps * _I, - _F);
          else
            k1 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero())
            phi_k1 = _i * as_scalar(k1.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * k1;  // πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ£
        if(_vqs.if_phi_neq_zero()) new_phi = phi_t + 0.5 * _delta * phi_k1;

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second πππ step with parameters πΌ(π‘) β πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ£
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            k2 = solve(_Q + _eps * _I, - _i * _F);
          else
            k2 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero())
            phi_k2 = as_scalar(- k2.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            k2 = solve(_Q + _eps * _I, - _F);
          else
            k2 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero())
            phi_k2 = _i * as_scalar(k2.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * k2;  // πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ€
        if(_vqs.if_phi_neq_zero()) new_phi = phi_t + 0.5 * _delta * phi_k2;

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /**************/
      /* THIRD STEP */
      /**************/
      //Makes a second πππ step with parameters πΌ(π‘) β πΌ(π‘) + π£/π€β’πΏββ’ΞΊπ€
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            k3 = solve(_Q + _eps * _I, - _i * _F);
          else
            k3 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero())
            phi_k3 = as_scalar(- k3.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            k3 = solve(_Q + _eps * _I, - _F);
          else
            k3 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero())
            phi_k3 = _i * as_scalar(k3.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + _delta * k3;  // πΌ(π‘) + πΏββ’ΞΊπ₯
        if(_vqs.if_phi_neq_zero()) new_phi = phi_t + _delta * phi_k3;

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* FOURTH STEP */
      /***************/
      //Makes a second πππ step with parameters πΌ(π‘) β πΌ(π‘) + πΏββ’ΞΊπ₯
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg)
            k4 = solve(_Q + _eps * _I, - _i * _F);
          else
            k4 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero())
            phi_k4 = as_scalar(- k4.st() * _mean_O) - _i * _E.real();

        }
        else{  // ππππππ

          if(_if_QGT_reg)
            k4 = solve(_Q + _eps * _I, - _F);
          else
            k4 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero())
            phi_k4 = _i * as_scalar(k4.st() * _mean_O) - _E.real();

        }

        //Final update of the variational parameters
        new_alpha = alpha_t + (1.0/6.0) * _delta * (k1 + k2 + k3 + k4);  // πΌβ(π‘ + πΏβ)
        if(_vqs.if_phi_neq_zero()) new_phi = phi_t + (1.0/6.0) * _delta * (phi_k1 + phi_k2 + phi_k3 + phi_k4);

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

      /*############*/
     /*  π?π½πΆπΉβ΄π  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // πα΄Ώ(π‘)
      double phi_t_im = _vqs.phi().imag();  // πα΄΅(π‘)
      Col <double> alpha_t_re = real(_vqs.alpha());  // πΆα΄Ώ(π‘)
      Col <double> alpha_t_im = imag(_vqs.alpha());  // πΆα΄΅(π‘)
      Col <double> k1_re;  // ΞΊπ£α΄Ώ = π»{πΌα΄Ώ(π‘)}
      Col <double> k1_im;  // ΞΊπ£α΄΅ = π»{πΌα΄΅(π‘)}
      Col <double> k2_re;  // ΞΊπ€α΄Ώ = π»{πΌα΄Ώ(π‘) + π£/π€β’πΏββ’ΞΊπ£α΄Ώ}
      Col <double> k2_im;  // ΞΊπ€α΄΅ = π»{πΌα΄΅(π‘) + π£/π€β’πΏββ’ΞΊπ£α΄΅}
      Col <double> k3_re;  // ΞΊπ₯α΄Ώ = π»{πΌα΄Ώ(π‘) + π£/π€β’πΏββ’ΞΊπ€α΄Ώ}
      Col <double> k3_im;  // ΞΊπ₯α΄΅ = π»{πΌα΄΅(π‘) + π£/π€β’πΏββ’ΞΊπ€α΄΅}
      Col <double> k4_re;  // ΞΊπ¦α΄Ώ = π»{πΌα΄Ώ(π‘) + πΏββ’ΞΊπ₯α΄Ώ}
      Col <double> k4_im;  // ΞΊπ¦α΄΅ = π»{πΌα΄΅(π‘) + πΏββ’ΞΊπ₯α΄΅}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      double phi_k1_re, phi_k2_re, phi_k3_re, phi_k4_re;
      double phi_k1_im, phi_k2_im, phi_k3_im, phi_k4_im;
      std::complex <double> new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            k1_re = solve(real(_Q) + _eps * _I, real(_F));
            k1_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            k1_re = solve(real(_Q), real(_F));
            k1_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k1_re = as_scalar(- k1_re.t() * _mean_O_angled - k1_im.t() * _mean_O_square);
            phi_k1_im = as_scalar(k1_re.t() * _mean_O_square - k1_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            k1_re = solve(real(_Q) + _eps * _I, imag(_F));
            k1_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            k1_re = solve(real(_Q), imag(_F));
            k1_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k1_re = as_scalar(- k1_re.t() * _mean_O_angled - k1_im.t() * _mean_O_square) - _E.real();
            phi_k1_im = as_scalar(k1_re.t() * _mean_O_square - k1_im.t() * _mean_O_angled);

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + 0.5 * _delta * k1_re);
        new_alpha.set_imag(alpha_t_im + 0.5 * _delta * k1_im);
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(phi_t_re + 0.5 * _delta * phi_k1_re);
          new_phi.imag(phi_t_im + 0.5 * _delta * phi_k1_im);

        }

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second πππ step at time π‘ + πΏπ‘
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            k2_re = solve(real(_Q) + _eps * _I, real(_F));
            k2_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            k2_re = solve(real(_Q), real(_F));
            k2_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k2_re = as_scalar(- k2_re.t() * _mean_O_angled - k2_im.t() * _mean_O_square);
            phi_k2_im = as_scalar(k2_re.t() * _mean_O_square - k2_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            k2_re = solve(real(_Q) + _eps * _I, imag(_F));
            k2_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            k2_re = solve(real(_Q), imag(_F));
            k2_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k2_re = as_scalar(- k2_re.t() * _mean_O_angled - k2_im.t() * _mean_O_square) - _E.real();
            phi_k2_im = as_scalar(k2_re.t() * _mean_O_square - k2_im.t() * _mean_O_angled);

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + 0.5 * _delta * k2_re);
        new_alpha.set_imag(alpha_t_im + 0.5 * _delta * k2_im);
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(phi_t_re + 0.5 * _delta * phi_k2_re);
          new_phi.imag(phi_t_im + 0.5 * _delta * phi_k2_re);

        }

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /**************/
      /* THIRD STEP */
      /**************/
      //Makes a second πππ step at time π‘ + πΏπ‘
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            k3_re = solve(real(_Q) + _eps * _I, real(_F));
            k3_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            k3_re = solve(real(_Q), real(_F));
            k3_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k3_re = as_scalar(- k3_re.t() * _mean_O_angled - k3_im.t() * _mean_O_square);
            phi_k3_im = as_scalar(k3_re.t() * _mean_O_square - k3_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            k3_re = solve(real(_Q) + _eps * _I, imag(_F));
            k3_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            k3_re = solve(real(_Q), imag(_F));
            k3_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k3_re = as_scalar(- k3_re.t() * _mean_O_angled - k3_im.t() * _mean_O_square) - _E.real();
            phi_k3_im = as_scalar(k3_re.t() * _mean_O_square - k3_im.t() * _mean_O_angled);

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + _delta * k3_re);
        new_alpha.set_imag(alpha_t_im + _delta * k3_im);
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(phi_t_re + _delta * phi_k3_re);
          new_phi.imag(phi_t_im + _delta * phi_k3_re);

        }

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* FOURTH STEP */
      /***************/
      //Makes a second πππ step at time π‘ + πΏπ‘
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // πππππ

          if(_if_QGT_reg){

            k4_re = solve(real(_Q) + _eps * _I, real(_F));
            k4_im = solve(real(_Q) + _eps * _I, imag(_F));

          }
          else{

            k4_re = solve(real(_Q), real(_F));
            k4_im = solve(real(_Q), imag(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k4_re = as_scalar(- k4_re.t() * _mean_O_angled - k4_im.t() * _mean_O_square);
            phi_k4_im = as_scalar(k4_re.t() * _mean_O_square - k4_im.t() * _mean_O_angled) - _E.real();

          }

        }
        else{  // ππππππ

          if(_if_QGT_reg){

            k4_re = solve(real(_Q) + _eps * _I, imag(_F));
            k4_im = solve(real(_Q) + _eps * _I, (-1.0) * real(_F));

          }
          else{

            k4_re = solve(real(_Q), imag(_F));
            k4_im = solve(real(_Q), (-1.0) * real(_F));

          }
          if(_vqs.if_phi_neq_zero()){

            phi_k4_re = as_scalar(- k4_re.t() * _mean_O_angled - k4_im.t() * _mean_O_square) - _E.real();
            phi_k4_im = as_scalar(k4_re.t() * _mean_O_square - k4_im.t() * _mean_O_angled);

          }

        }

        //Final update of the variational parameters
        new_alpha.set_real(alpha_t_re + (1.0/6.0) * _delta * (k1_re + k2_re + k3_re + k4_re));
        new_alpha.set_imag(alpha_t_im + (1.0/6.0) * _delta * (k1_im + k2_im + k3_im + k4_im));
        if(_vqs.if_phi_neq_zero()){

          new_phi.real(phi_t_re + (1.0/6.0) * _delta * (phi_k1_re + phi_k2_re + phi_k3_re + phi_k4_re));
          new_phi.imag(phi_t_im + (1.0/6.0) * _delta * (phi_k1_im + phi_k2_im + phi_k3_im + phi_k4_im));

        }

      }

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_vqs.if_phi_neq_zero()){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

  }
  else
    return;

}


#endif
