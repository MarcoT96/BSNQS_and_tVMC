#ifndef __SAMPLER__
#define __SAMPLER__


/*********************************************************************************************************/
/********************************  Variational Monte Carlo Sampler  **************************************/
/*********************************************************************************************************/
/*

  We create a Variational Quantum Monte Carlo (𝐕𝐌𝐂) sampler as a C++ class, which is able to
  optimize a generic 𝐒𝐡𝐚𝐝𝐨𝐰 𝐚𝐧𝐬𝐚𝐭𝐳 (a variational quantum state 𝐯𝐪𝐬) in order to study a
  generic Lattice Quantum System (𝐋𝐐𝐒).
  The main goal of the sampler is to optimize the parameters that uniquely characterize the 𝐯𝐪𝐬
  to obtain the ground state of the given Hamiltonian; once found the ground state, it is
  possible to study the real-time dynamics of the system after performing a quantum quench on a
  certain coupling constant.

  The optimization described above takes place within a stochastic setting, in which the
  procedure leads to the resolution of the following equations of motion for the variational
  parameters 𝜶 (𝐓𝐃𝐕𝐌𝐂 Equations of Motion):

            Σₖ 𝛼̇ₖ {𝛼ⱼ, 𝛼ₖ} = ∂𝙀[𝜶]/∂𝛼ⱼ      (𝐓𝐃𝐕𝐌𝐂)
            Σₖ 𝛼̇ₖ {𝛼ⱼ, 𝛼ₖ} = -𝑖•∂𝙀[𝜶]/∂𝛼ⱼ   (𝑖-𝐓𝐃𝐕𝐌𝐂)

  where the ground state properties are recovered with an imaginaty time evolution

            𝒕 → 𝝉 = -𝑖𝒕.

  This class is also able to apply the above technique to a non-shadow ansatz, where
  different hypotheses are assumed for the form of the variational wave function.

  N̲O̲T̲E̲: we use the pseudo-random numbers generator device by [Percus & Kalos, 1989, NY University].

*/
/*********************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etc…
#include <iomanip>  // <-- std::setw(), std::fixed(), std::setprecision()
#include <fstream>  // <-- std::ifstream, std::ofstream, std::flush()
#include <complex>  // <-- std::complex<>, .real(), .imag()
#include <armadillo>  // <-- arma::Mat, arma::Col, arma::Row, arma::field
#include "random.h"  // <-- Random
#include "ansatz.cpp"  // <-- WaveFunction
#include "model.cpp"  // <-- SpinHamiltonian


using namespace arma;


class VMC_Sampler {

  private:

    //Quantum problem defining variables
    WaveFunction& _vqs;  //The wave function ansatz |Ψ(𝜙,𝜶)⟩
    SpinHamiltonian& _H;  //The Spin Hamiltonian Ĥ
    const unsigned int _Nspin;  //Number of spins in the system

    //Constant data-members
    const std::complex <double> _i;  //The imaginary unit 𝑖
    const Mat <double> _I;  //The real identity matrix 𝟙

    //Random device
    Random _rnd;

    //Quantum configuration variables |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
    const unsigned int _Nhidden;  //Number of auxiliary quantum variables
    Mat <int> _configuration;  //Current visible configuration of the system |𝒗⟩ = |𝓋𝟣 𝓋𝟤 … 𝓋𝖭⟩
    Mat <int> _hidden_ket;  //The ket configuration of the hidden variables |𝒉⟩ = |𝒽𝟣 𝒽𝟤 … 𝒽𝖬⟩
    Mat <int> _hidden_bra;  //The bra configuration of the hidden variables ⟨𝒉ˈ| = ⟨𝒽ˈ𝖬 … 𝒽ˈ𝟤 𝒽ˈ𝟣|
    Mat <int> _flipped_site;  //The new sampled visible configuration |𝒗ⁿᵉʷ⟩
    Mat <int> _flipped_ket_site;  //The new sampled ket configuration of the hidden variables |𝒉ⁿᵉʷ⟩
    Mat <int> _flipped_bra_site;  //The new sampled bra configuration of the hidden variables ⟨𝒉ˈⁿᵉʷ|

    //Monte Carlo moves statistics variables
    unsigned int _N_accepted_visible;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ accepted along the MCMC
    unsigned int _N_proposed_visible;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ proposed along the MCMC
    unsigned int _N_accepted_ket;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ accepted along the MCMC
    unsigned int _N_proposed_ket;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ proposed along the MCMC
    unsigned int _N_accepted_bra;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ accepted along the MCMC
    unsigned int _N_proposed_bra;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ proposed along the MCMC
    unsigned int _N_accepted_equal_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with equal-site-spin-flip accepted along the MCMC
    unsigned int _N_proposed_equal_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with equal-site-spin-flip proposed along the MCMC
    unsigned int _N_accepted_visible_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ with nearest-neighbors-site-spin-flip accepted along the MCMC
    unsigned int _N_proposed_visible_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ with nearest-neighbors-site-spin-flip proposed along the MCMC
    unsigned int _N_accepted_hidden_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with nearest-neighbors-site-spin-flip accepted along the MCMC
    unsigned int _N_proposed_hidden_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with nearest-neighbors-site-spin-flip proposed along the MCMC

    //Monte Carlo storage variables
    field <Row <std::complex <double>>> _Connections;  //Non-zero matrix elements (i.e. the connections) of the observable operators
    field <field <Mat <int>>> _StatePrime;  //List of configuration |𝒮'⟩ associated to each observables connections
    Mat <double> _instReweight;  //Measured the 𝐑𝐞𝐰𝐞𝐢𝐠𝐡𝐭𝐢𝐧𝐠 ratio ingredients along the MCMC
    Mat <std::complex <double>> _instObs_ket;  //Measured values of quantum observables on the configuration |𝒗 𝒉⟩  along the MCMC
    Mat <std::complex <double>> _instObs_bra;  //Measured values of quantum observables on the configuration |𝒗 𝒉ˈ⟩ along the MCMC
    Mat <std::complex <double>> _instO_ket;  //Measured local operators 𝕆(𝒗,𝒉) along the MCMC
    Mat <std::complex <double>> _instO_bra;  //Measured local operators 𝕆(𝒗,𝒉ˈ) along the MCMC

    //Simulation options variables
    bool _if_shadow;  //Chooses the shadow or the non-shadow algorithm
    bool _if_hidden_off;  //Chooses to shut down the auxiliary variable in a Shadow vqs
    bool _if_vmc;  //Chooses to make a single simple 𝐕𝐌𝐂 without parameters optimization
    bool _if_imag_time;  //Chooses imaginary-time dinamics, i.e. 𝐓𝐃𝐕𝐌𝐂 with 𝛕 = -𝑖𝐭
    bool _if_real_time;  //Chooses real-time dynamics
    bool _if_QGT_reg;  //Chooses to regularize the Quantum Geometric Tensor by adding a bias
    bool _if_extra_hidden_sum;  //Increases the sampling of |𝒉⟩ and ⟨𝒉ˈ| during the single measure
    bool _if_restart_from_config;  //Chooses to initialize the initial point of the MCMC from a previously optimized visible configuration |𝒗⟩

    //Simulation parameters of the single 𝐕𝐌𝐂 step
    unsigned int _Nsweeps;  //Number of Monte Carlo sweeps (i.e. #MC-steps of the single 𝐕𝐌𝐂 step)
    unsigned int _Nblks;  //Number of blocks to estimate uncertainties
    unsigned int _Neq;  //Number of Monte Carlo equilibration steps to do at the beginning of the single 𝐕𝐌𝐂 step
    unsigned int _M;  //Number of spin-flips moves to perform in the single sweep
    unsigned int _Nflips;  //Number of spin-flips in each spin-flips move
    unsigned int _Nextra;  //Number of extra MC-steps involving only the hidden sampling
    unsigned int _Nblks_extra;  //Number of blocks in the extra hidden sampling
    double _p_equal_site;  //Probability for the equal site Monte Carlo move
    double _p_visible_nn;  //Probability for the visible nearest neighbor Monte Carlo move
    double _p_hidden_nn;  //Probability for the hidden nearest neighbor Monte Carlo move

    //𝐓𝐃𝐕𝐌𝐂 variables
    double _delta;  //The value of the integration step 𝛿𝑡
    double _eps;  //The value of the Quantum Geometric Tensor bias ε
    unsigned int _fixed_hidden_orientation;  //Bias on the value of all the auxiliary degrees of freedom
    Col <double> _cosII;  //The block averages of the non-zero reweighting ratio part ⟨cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]⟩ⱼᵇˡᵏ
    Col <double> _sinII;  //The block averages of the (theoretically)-zero reweighting ratio part ⟨sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]⟩ⱼᵇˡᵏ
    field <Col <std::complex <double>>> _Observables;  //The block averages of the quantum observables computed along the MCMC ⟨𝒪⟩ⱼᵇˡᵏ
    field <Col <std::complex <double>>> _O;  //The block averages of the local operators computed along the MCMC ⟨𝕆ₖ⟩ⱼᵇˡᵏ, for k = 𝟣,…,nᵃˡᵖʰᵃ
    Col <double> _mean_O_angled;  // ⟨≪𝕆≫ᵇˡᵏ⟩
    Col <double> _mean_O_square;  // ⟨⌈𝕆⌋ᵇˡᵏ⟩
    std::complex <double> _E;  // The standard stochastic average of ⟨ℋ⟩ (without block averaging)
    Mat <std::complex <double>> _Q;  //The Quantum Geometric Tensor ℚ
    Col <std::complex <double>> _F;  //The energy Gradient 𝔽 acting on 𝜶

    //Print options and related files
    bool _write_Move_Statistics;  //Writes the acceptance statistics along the single MCMC
    bool _write_MCMC_Config;  //Writes the sampled |𝒮⟩ along the single MCMC
    bool _write_final_Config;  //Writes the last sampled |𝒮⟩ of each 𝐕𝐌𝐂 step
    bool _write_opt_Observables;  //Writes optimized Monte Carlo estimates of quantum observables at the end of each 𝐕𝐌𝐂 step
    bool _write_block_Observables;  //Writes the observables averages in each block of the MCMC, for each 𝐕𝐌𝐂 step
    bool _write_opt_Params;  //Writes the optimized set 𝓥ᵒᵖᵗ of the variational wave function at the end of the 𝐓𝐃𝐕𝐌𝐂
    bool _write_all_Params;  //Writes the set of optimized 𝓥 of the variational wave function after each 𝐕𝐌𝐂 step
    bool _write_QGT_matrix;  //Writes the Quantum Geometric Tensor matrix of each 𝐕𝐌𝐂 step
    bool _write_QGT_cond;  //Writes the condition number of the Quantum Geometric Tensor matrix of each 𝐕𝐌𝐂 step
    bool _write_QGT_eigen;  //Writes the Quantum Geometric Tensor eigenvalues of each 𝐕𝐌𝐂 step
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
    VMC_Sampler(WaveFunction&, SpinHamiltonian&);
    ~VMC_Sampler();

    //Access functions
    WaveFunction& vqs() const {return _vqs;}  //Returns the reference to the ansatz wave function
    SpinHamiltonian& H() const {return _H;}  //Returns the reference to the spin Hamiltonian
    unsigned int n_spin() const {return _Nspin;}  //Returns the number of quantum degrees of freedom
    unsigned int n_hidden() const {return _Nhidden;}  //Returns the number of auxiliary degrees of freedom
    std::complex <double> i() const {return _i;}  //Returns the imaginary unit 𝑖
    Mat <double> I() const {return _I;}  //Returns the identity matrix 𝟙
    Mat <int> visible_configuration() const {return _configuration;}  //Returns the sampled visible configuration |𝒗⟩
    Mat <int> hidden_ket() const {return _hidden_ket;}  //Returns the sampled ket configuration of the hidden variables |𝒉⟩
    Mat <int> hidden_bra() const {return _hidden_bra;}  //Returns the sampled bra configuration of the hidden variables ⟨𝒉ˈ|
    Mat <int> new_visible_config() const {return _flipped_site;}  //Returns the new sampled visible configuration |𝒗ⁿᵉʷ⟩
    Mat <int> new_hidden_ket() const {return _flipped_ket_site;}  //Returns the new sampled ket configuration |𝒉ⁿᵉʷ⟩
    Mat <int> new_hidden_bra() const {return _flipped_bra_site;}  //Returns the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|
    void print_configuration() const;  //Prints on standard output the current sampled system configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
    field <Row <std::complex <double>>> get_connections() const {return _Connections;}  //Returns the list of connections
    field <field <Mat <int>>> all_state_prime() const {return _StatePrime;}  //Returns all the configuration |𝒮'⟩ connected to the current sampled configuration |𝒮⟩
    Mat <std::complex <double>> InstObs_ket() const {return _instObs_ket;}  //Returns all the measured values of 𝒪ˡᵒᶜ(𝒗,𝒉) after a single VMC run
    Mat <std::complex <double>> InstObs_bra() const {return _instObs_bra;}  //Returns all the measured values of 𝒪ˡᵒᶜ(𝒗,𝒉') after a single VMC run
    Mat <std::complex <double>> InstO_ket() const {return _instO_ket;}  //Returns all the measured local operators 𝕆(𝒗,𝒉) after a single VMC run
    Mat <std::complex <double>> InstO_bra() const {return _instO_bra;}  //Returns all the measured local operators 𝕆(𝒗,𝒉') after a single VMC run
    Mat <double> InstNorm() const {return _instReweight;}  //Returns all the measured values of 𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] and 𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] after a single VMC run
    double delta() const {return _delta;}  //Returns the integration step parameter 𝛿𝑡 used in the dynamics solver
    double QGT_bias() const {return _eps;}  //Returns the regularization bias of the Quantum Geometric Tensor
    unsigned int hidden_bias() const {return _fixed_hidden_orientation;}  //Returns the orientation bias of the hidden variables
    Col <double> cos() const {return _cosII;}
    Col <double> sin() const {return _sinII;}
    field <Col <std::complex <double>>> Observables() const {return _Observables;}
    field <Col <std::complex <double>>> O() const {return _O;}
    Mat <std::complex <double>> QGT() const {return _Q;}  //Returns the Monte Carlo estimate of the QGT
    Col <std::complex <double>> F() const {return _F;}  //Returns the Monte Carlo estimate of the energy gradient
    Col <double> mean_O_angled() const {return _mean_O_angled;}  //Returns the Monte Carlo estimate of the vector of ≪𝕆ₖ≫
    Col <double> mean_O_square() const {return _mean_O_square;}  //Returns the Monte Carlo estimate of the vector of ⌈𝕆ₖ⌋
    std::complex <double> E() const {return _E;}  //Returns the Monte Carlo estimate of the energy ⟨ℋ⟩

    //Initialization functions
    void Init_Config(const Mat <int>& initial_visible=Mat <int>(),  //Initializes the quantum configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
                     const Mat <int>& initial_ket=Mat <int>(),
                     const Mat <int>& initial_bra=Mat <int>(),
                     bool zeroMag=true);
    void ShutDownHidden(unsigned int);  //Shuts down the hidden variables
    void setImagTimeDyn(double delta=0.01);  //Chooses the imaginary-time 𝐓𝐃𝐕𝐌𝐂 algorithm
    void setRealTimeDyn(double delta=0.01);  //Chooses the real-time 𝐓𝐃𝐕𝐌𝐂 algorithm
    void setQGTReg(double eps=0.000001);  //Chooses to regularize the QGT
    void setExtraHiddenSum(unsigned int, unsigned int);  //Chooses to make the MC observables less noisy
    void setRestartFromConfig() {_if_restart_from_config = true;}  //Chooses the restart option at the beginning of the MCMC
    void setStepParameters(unsigned int, unsigned int, unsigned int,           //Sets the Monte Carlo parameters for the single VMC step
                           unsigned int, unsigned int, double, double, double);

    //Print options functions
    void setFile_Move_Statistics(std::string);
    void setFile_MCMC_Config(std::string);
    void setFile_final_Config(std::string);
    void setFile_opt_Obs(std::string);
    void setFile_block_Obs(std::string);
    void setFile_opt_Params(std::string);
    void setFile_all_Params(std::string);
    void setFile_QGT_matrix(std::string);
    void setFile_QGT_cond(std::string);
    void setFile_QGT_eigen(std::string);
    void Write_Move_Statistics(unsigned int);
    void Write_MCMC_Config(unsigned int);
    void Write_final_Config(unsigned int);
    void Write_opt_Params();
    void Write_all_Params(unsigned int);
    void Write_QGT_matrix(unsigned int);
    void Write_QGT_cond(unsigned int);
    void Write_QGT_eigen(unsigned int);
    void CloseFile();

    //Measurement functions
    void Reset();
    Col <double> average_in_blocks(const Row <double>&) const;
    Col <std::complex <double>> average_in_blocks(const Row <std::complex <double>>&) const;
    Col <double> Shadow_average_in_blocks(const Row <std::complex <double>>&, const Row <std::complex <double>>&) const;
    Col <double> Shadow_angled_average_in_blocks(const Row <std::complex <double>>&, const Row <std::complex <double>>&) const;
    Col <double> Shadow_square_average_in_blocks(const Row <std::complex <double>>&, const Row <std::complex <double>>&) const;
    void compute_Reweighting_ratio();
    Col <double> compute_errorbar(const Col <double>&) const;
    Col <std::complex <double>> compute_errorbar(const Col <std::complex <double>>&) const;
    Col <double> compute_progressive_averages(const Col <double>&) const;
    Col <std::complex <double>> compute_progressive_averages(const Col <std::complex <double>>&) const;
    void compute_Quantum_observables();
    void compute_O();
    void compute_QGTandGrad();
    void is_asymmetric(const Mat <double>&) const;  //Check the anti-symmetric properties of an Armadillo matrix
    void QGT_Check();  //Checks symmetry properties of the Quantum Geometric Tensor
    void Measure();  //Measurement of the istantaneous observables along a single VMC run
    void Estimate();  //Monte Carlo estimates of the quantum observable averages
    void Write_Quantum_properties(unsigned int);  //Write on appropriate files all the required system quantum properties

    //Monte Carlo moves
    bool RandFlips_visible(Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move for the visibles variable only
    bool RandFlips_hidden(Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move for the hidden variables (ket or bra only)
    bool RandFlips_visible_nn_site(Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move on two visible nearest neighbors lattice site
    bool RandFlips_hidden_nn_site(Mat <int>&, Mat <int>&, unsigned int);  //Decides how to make a single bunch_of_spin-flip move on two hidden nearest neighbors lattice site
    void Move_visible(unsigned int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
    void Move_ket(unsigned int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
    void Move_bra(unsigned int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
    void Move_equal_site(unsigned int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with equal-site-spin-flip
    void Move_visible_nn_site(unsigned int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ with nearest-neighbors-site-spin-flip
    void Move_hidden_nn_site(unsigned int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with nearest-neighbors-site-spin-flip
    void Move();  //Samples a new system configuration

    //Sampling functions
    void Make_Sweep();  //Adds a point in the Monte Carlo Markov Chain
    void Reset_Moves_Statistics();  //Resets the Monte Carlo moves statistics variables
    void VMC_Step();  //Performs a single VMC step

    //ODE Integrators
    void Euler();  //Updates the variational parameters with the Euler integration method
    void Heun();   //Updates the variational parameters with the Heun integration method
    void RK4();    //Updates the variational parameters with the fourth order Runge Kutta method

};


/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
VMC_Sampler :: VMC_Sampler(WaveFunction& wave, SpinHamiltonian& hamiltonian)
             : _vqs(wave), _H(hamiltonian), _Nspin(wave.n_visible()), _i(_H.i()),
               _I(eye(_vqs.n_alpha(), _vqs.n_alpha())), _Nhidden(wave.n_visible() * wave.density()) {

  //Information
  std::cout << "#Define the 𝐕𝐌𝐂 sampler of the variational quantum state |Ψ(𝜙, 𝜶)⟩." << std::endl;
  std::cout << " The sampler is defined on a " << _vqs.type_of_ansatz() << " architecture designed for Lattice Quantum Systems." << std::endl;

  //Sets the simulation option variables
  if(_vqs.type_of_ansatz() == "Shadow")
    _if_shadow = true;
  else
    _if_shadow = false;
  _if_hidden_off = false;
  _if_vmc = true;  //Default algorithm → simple 𝐕𝐌𝐂
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

  //Creates and initializes the Random Number Generator
  std::cout << " Create and initialize the random number generator." << std::endl;
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes");
  if(Primes.is_open()){
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
    std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed2.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

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

  std::cout << " 𝐕𝐌𝐂 sampler correctly initialized." << std::endl;

}


VMC_Sampler :: ~VMC_Sampler() {

  _rnd.SaveSeed();

}


void VMC_Sampler :: print_configuration() const {

  std::cout << "\n=====================================" << std::endl;
  std::cout << "Current configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "|𝒗⟩      = ";
  _configuration.print();
  std::cout << "|𝒉⟩      = ";
  _hidden_ket.print();
  std::cout << "⟨𝒉ˈ|     = ";
  _hidden_bra.print();

}


void VMC_Sampler :: Init_Config(const Mat <int>& initial_visible, const Mat <int>& initial_ket, const Mat <int>& initial_bra, bool zeroMag) {

  /*##############################################################################################*/
  //  Initializes the starting point of the MCMC, using the computational basis of σ̂ᶻ eigenstates
  //
  //        σ̂ᶻ|+1⟩ = +|+1⟩
  //        σ̂ᶻ|-1⟩ = -|-1⟩.
  //
  //  We give the possibility to randomly choose spin up or down for each lattice site
  //  by using the conditional ternary operator
  //
  //        condition ? result1 : result2
  //
  //  or to initialize the configuration by providing an acceptable 𝐢𝐧𝐢𝐭𝐢𝐚𝐥_𝐜𝐨𝐧𝐟𝐢𝐠 for the
  //  variables. Hidden variables are randomly initialized in both cases.
  //  If the boolean data-member 𝐢𝐟_𝒉𝒊𝒅𝒅𝒆𝒏_𝐨𝐟𝐟 is true, the hidden variables are all initialized
  //  and fixed to a certain constant (𝐟𝐢𝐱𝐞𝐝_𝐡𝐢𝐝𝐝𝐞𝐧_𝐨𝐫𝐢𝐞𝐧𝐭𝐚𝐭𝐢𝐨𝐧), that is they are turned off in
  //  order to make the Shadow ansatz a simple ansatz deprived of the auxiliary variables.
  //  Beware that this is not equivalent to knowing how to analytically integrate the hidden
  //  variables!
  //  If 𝐳𝐞𝐫𝐨𝐌𝐚𝐠 is true the initial physical configuration |𝒗⟩ is prepared with
  //  zero total magnetization.
  //
  //  So, this function initializes the generic configuration to sample along the Markov Chain
  //
  //        |𝒮⟩ = |𝒗, 𝐡, 𝐡ˈ⟩.
  //
  //  As concerns the configuration of the hidden variables, we do not make any request with
  //  respect to its magnetization, being non-physical variables.
  /*##############################################################################################*/

  //Initializes the configuration depending on |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

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
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

    /*
      .............
      .............
      .............
    */

  }

  //Randomly chooses spin up or spin down in |𝒗⟩
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

  //Initializes |𝐡⟩ and ⟨𝐡ˈ| randomly
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
  //        𝒕 → 𝝉 = -𝑖𝒕
  //
  //  and using an integration step parameter 𝛿𝑡.
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
  //  real time t and using an integration step parameter 𝛿𝑡.
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
  //        ℚ → ℚ + 𝜀•𝟙  (𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝕊 → 𝕊 + 𝜀•𝟙  (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  in order to avoid inversion problems in the
  //  integration of the equations of motion.
  //  𝐞𝐩𝐬𝐢𝐥𝐨𝐧 is the bias strength.
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
                                      unsigned int Nflips, double p_equal_site, double p_visible_nn, double p_hidden_nn) {

  _Nsweeps = Nsweeps;
  _Nblks = Nblks;
  _Neq = Neq;
  _M = M;
  _Nflips = Nflips;
  _p_equal_site = p_equal_site;
  _p_visible_nn = p_visible_nn;
  _p_hidden_nn = p_hidden_nn;

  std::cout << " Parameters of the simulation:" << std::endl;
  std::cout << " \tNsweeps in the single 𝑽𝑴𝑪 step = " << _Nsweeps << std::endl;
  std::cout << " \tNblks in the single 𝑽𝑴𝑪 step = " << _Nblks << std::endl;
  std::cout << " \tEquilibration steps in the single 𝑽𝑴𝑪 step = " << _Neq << std::endl;
  std::cout << " \tNumber of spin-flips moves in the single MC Sweep = " << _M << std::endl;
  std::cout << " \tNumber of spin-flip in the single spin-flips move = " << _Nflips << std::endl;
  std::cout << " \tP(equal site) = " << _p_equal_site * 100.0 << " %" << std::endl;
  std::cout << " \tP(n.n. visible) = " << _p_visible_nn * 100.0 << " %" << std::endl;
  std::cout << " \tP(n.n. hidden) = " << _p_hidden_nn * 100.0 << " %" << std::endl;
  std::cout << " \tNumber of extra hidden sampling in each instantaneous measurement = "  << _Nextra << std::endl;
  std::cout << " \tNumber of block for the extra hidden sampling = " << _Nblks_extra << std::endl;
  std::cout << " \tIntegration step parameter = " << _delta << std::endl;
  std::cout << " \tQGT bias = " << _eps << std::endl << std::endl;

}


void VMC_Sampler :: setFile_Move_Statistics(std::string info) {

  _write_Move_Statistics = true;
  _file_Move_Statistics.open("Move_Statistics_" + info + ".dat");
  if(!_file_Move_Statistics.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ Move_Statistics_" << info << ".dat ›› for writing the acceptance statistics at the end of the single 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the acceptance statistics of the moves at the end of the single 𝐕𝐌𝐂 step on file ‹‹ Move_Statistics_" << info << ".dat ››." << std::endl;
  _file_Move_Statistics << "###########################################################################################################\n";
  _file_Move_Statistics << "# Column Legend\n";
  _file_Move_Statistics << "#\n";
  _file_Move_Statistics << "#   1st: the 𝐕𝐌𝐂 step identifier\n";
  _file_Move_Statistics << "#   2nd: the sampling acceptance probability (%) of |𝒗⟩\n";
  _file_Move_Statistics << "#   3rd: the sampling acceptance probability (%) of |𝒉⟩\n";
  _file_Move_Statistics << "#   4th: the sampling acceptance probability (%) of ⟨𝒉ˈ|\n";
  _file_Move_Statistics << "#   5th: the sampling acceptance probability (%) of |𝒗 𝒉 𝒉ˈ⟩ moved on equal sites\n";
  _file_Move_Statistics << "#   6th: the sampling acceptance probability (%) of |𝒗⟩ moved on nearest-neighbor sites\n";
  _file_Move_Statistics << "#   7th: the sampling acceptance probability (%) of |𝒉⟩ and ⟨𝒉ˈ| moved on generally nearest-neighbor sites\n";
  _file_Move_Statistics << "###########################################################################################################\n";

}


void VMC_Sampler :: setFile_MCMC_Config(std::string info) {

  _write_MCMC_Config = true;
  _file_MCMC_Config.open("MCMC_config_" + info + ".dat");
  if(!_file_MCMC_Config.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ MCMC_config_" << info << ".dat ›› for writing the sampled configurations along a single MCMC." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the sampled configurations along a single MCMC on file ‹‹ MCMC_config_" << info << ".dat ››." << std::endl;
  _file_MCMC_Config << "####################################################\n";
  _file_MCMC_Config << "# Column Legend\n";
  _file_MCMC_Config << "#\n";
  _file_MCMC_Config << "#   1st: the 𝐌𝐂-step identifier\n";
  _file_MCMC_Config << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
  _file_MCMC_Config << "####################################################\n";

}


void VMC_Sampler :: setFile_final_Config(std::string info) {

  _write_final_Config = true;
  _file_final_Config.open("final_config_" + info + ".dat");
  if(!_file_final_Config.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ final_config_" << info << ".dat ›› for writing the final configurations at the end of each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the final configurations sampled at the end of each 𝐕𝐌𝐂 step on file ‹‹ final_config_" << info << ".dat ››." << std::endl;
  _file_final_Config << "####################################################\n";
  _file_final_Config << "# Column Legend\n";
  _file_final_Config << "#\n";
  _file_final_Config << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_final_Config << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
  _file_final_Config << "####################################################\n";

}


void VMC_Sampler :: setFile_opt_Obs(std::string info) {

  _write_opt_Observables = true;
  _file_opt_Energy.open("opt_energy_" + info + ".dat");
  _file_opt_SigmaX.open("opt_sigmaX_" + info + ".dat");
  _file_opt_SigmaY.open("opt_sigmaY_" + info + ".dat");
  _file_opt_SigmaZ.open("opt_sigmaZ_" + info + ".dat");
  if(!_file_opt_Energy.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ opt_energy_" << info << ".dat ›› for writing E(𝜙,𝜶 ) after each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving E(𝜙,𝜶) after each 𝐕𝐌𝐂 step on file ‹‹ opt_energy_" << info << ".dat ››." << std::endl;
  if(!_file_opt_SigmaX.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ opt_sigmaX_" << info << ".dat ›› for writing σˣ(𝜙,𝜶) after each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving σˣ(𝜙,𝜶) after each 𝐕𝐌𝐂 step on file ‹‹ opt_sigmaX_" << info << ".dat ››." << std::endl;
  if(!_file_opt_SigmaY.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ opt_sigmaY_" << info << ".dat ›› for writing σʸ(𝜙,𝜶) after each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving σʸ(𝜙,𝜶) after each 𝐕𝐌𝐂 step on file ‹‹ opt_sigmaY_" << info << ".dat ››." << std::endl;
  if(!_file_opt_SigmaZ.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ opt_sigmaZ_" << info << ".dat ›› for writing σᶻ(𝜙,𝜶  ) after each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving σᶻ(𝜙,𝜶) after each 𝐕𝐌𝐂 step on file ‹‹ opt_sigmaZ_" << info << ".dat ››." << std::endl;

  _file_opt_Energy << "###########################################\n";
  _file_opt_Energy << "# Column Legend\n";
  _file_opt_Energy << "#\n";
  _file_opt_Energy << "#   1st:  the 𝐕𝐌𝐂-step identifier\n";
  _file_opt_Energy << "#   2nd:  progressive ⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆\n";
  _file_opt_Energy << "#   3rd:  progressive 𝜎[⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆]\n";
  _file_opt_Energy << "#   4th:  progressive ⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆\n";
  _file_opt_Energy << "#   5rd:  progressive 𝜎[⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆]\n";
  _file_opt_Energy << "#   6th:  progressive ⟨𝑬ᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_Energy << "#   7th:  progressive 𝜎[⟨𝑬ᴿ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_Energy << "#   8th:  progressive ⟨𝑬ᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_Energy << "#   9th:  progressive 𝜎[⟨𝑬ᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_Energy << "#   10th: standard ⟨𝑬ᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_Energy << "#   11th: standard ⟨𝑬ᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_Energy << "###########################################\n";

  _file_opt_SigmaX << "###########################################\n";
  _file_opt_SigmaX << "# Column Legend\n";
  _file_opt_SigmaX << "#\n";
  _file_opt_SigmaX << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_opt_SigmaX << "#   2nd: progressive ⟨σˣᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_SigmaX << "#   3rd: progressive 𝜎[⟨σˣᴿ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_SigmaX << "#   4th: progressive ⟨σˣᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_SigmaX << "#   5th: progressive 𝜎[⟨σˣᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_SigmaX << "###########################################\n";

  _file_opt_SigmaY << "###########################################\n";
  _file_opt_SigmaY << "# Column Legend\n";
  _file_opt_SigmaY << "#\n";
  _file_opt_SigmaY << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_opt_SigmaY << "#   2nd: progressive ⟨σʸᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_SigmaY << "#   3rd: progressive 𝜎[⟨σʸᴿ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_SigmaY << "#   4th: progressive ⟨σʸᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_SigmaY << "#   5th: progressive 𝜎[⟨σʸᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_SigmaY << "###########################################\n";

  _file_opt_SigmaZ << "###########################################\n";
  _file_opt_SigmaZ << "# Column Legend\n";
  _file_opt_SigmaZ << "#\n";
  _file_opt_SigmaZ << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_opt_SigmaZ << "#   2nd: progressive ⟨σᶻᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_SigmaZ << "#   3rd: progressive 𝜎[⟨σᶻᴿ(𝜙,𝜶)⟩_𝛹 ]\n";
  _file_opt_SigmaZ << "#   4th: progressive ⟨σᶻᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_opt_SigmaZ << "#   5th: progressive 𝜎[⟨σᶻᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_opt_SigmaZ << "###########################################\n";

}


void VMC_Sampler :: setFile_block_Obs(std::string info) {

  _write_block_Observables = true;
  _file_block_Energy.open("block_energy_" + info + ".dat");
  _file_block_SigmaX.open("block_sigmaX_" + info + ".dat");
  _file_block_SigmaY.open("block_sigmaY_" + info + ".dat");
  _file_block_SigmaZ.open("block_sigmaZ_" + info + ".dat");
  if(!_file_block_Energy.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ block_energy_" << info << ".dat ›› for writing all the block averages of E(𝜙,𝜶 ) during each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the block averages of E(𝜙,𝜶) during each 𝐕𝐌𝐂 step on file ‹‹ block_energy_" << info << ".dat ››." << std::endl;
  if(!_file_block_SigmaX.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ block_sigmaX_" << info << ".dat ›› for writing all the block averages of σˣ(𝜙,𝜶) during each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the block averages of σˣ(𝜙,𝜶) during each 𝐕𝐌𝐂 step on file ‹‹ block_sigmaX_" << info << ".dat ››." << std::endl;
  if(!_file_block_SigmaY.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ block_sigmaY_" << info << ".dat ›› for writing all the block averages of σʸ(𝜙,𝜶) during each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the block averages of σʸ(𝜙,𝜶) during each 𝐕𝐌𝐂 step on file ‹‹ block_sigmaY_" << info << ".dat ››." << std::endl;
  if(!_file_block_SigmaZ.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ block_sigmaZ_" << info << ".dat ›› for writing all the block averages of σᶻ(𝜙,𝜶) during each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the block averages of σᶻ(𝜙,𝜶) during each 𝐕𝐌𝐂 step on file ‹‹ block_sigmaZ_" << info << ".dat ››." << std::endl;

  _file_block_Energy << "############################################\n";
  _file_block_Energy << "# Column Legend\n";
  _file_block_Energy << "#\n";
  _file_block_Energy << "#   1st:  the 𝐕𝐌𝐂-step identifier\n";
  _file_block_Energy << "#   2nd:  the 𝐌𝐂-block identifier\n";
  _file_block_Energy << "#   3rd:  ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
  _file_block_Energy << "#   4th:  ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
  _file_block_Energy << "#   5th:  ⟨𝑬ᴿ(𝜙,𝜶)⟩ʲ𝓆 in block j\n";
  _file_block_Energy << "#   6th:  progressive ⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆\n";
  _file_block_Energy << "#   7th:  progressive 𝜎[⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆]\n";
  _file_block_Energy << "#   8th:  progressive ⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆\n";
  _file_block_Energy << "#   9th:  progressive 𝜎[⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆]\n";
  _file_block_Energy << "#   10th:  progressive ⟨𝑬ᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_Energy << "#   11th:  progressive 𝜎[⟨𝑬ᴿ(𝜙,𝜶)⟩_𝛹 ]\n";
  _file_block_Energy << "#   12th:  progressive ⟨𝑬ᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_Energy << "#   13th: progressive 𝜎[⟨𝑬ᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_block_Energy << "############################################\n";

  _file_block_SigmaX << "############################################\n";
  _file_block_SigmaX << "# Column Legend\n";
  _file_block_SigmaX << "#\n";
  _file_block_SigmaX << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_block_SigmaX << "#   2nd: the 𝐌𝐂-block identifier\n";
  _file_block_SigmaX << "#   3rd: progressive ⟨σˣᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_SigmaX << "#   4th: progressive 𝜎[⟨σˣᴿ(𝜙,𝜶)⟩_𝛹 ]\n";
  _file_block_SigmaX << "#   5th: progressive ⟨σˣᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_SigmaX << "#   6th: progressive 𝜎[⟨σˣᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_block_SigmaX << "############################################\n";

  _file_block_SigmaY << "############################################\n";
  _file_block_SigmaY << "# Column Legend\n";
  _file_block_SigmaY << "#\n";
  _file_block_SigmaY << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_block_SigmaY << "#   2nd: the 𝐌𝐂-block identifier\n";
  _file_block_SigmaY << "#   3rd: progressive ⟨σʸᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_SigmaY << "#   4th: progressive 𝜎[⟨σʸᴿ(𝜙,𝜶)⟩_𝛹 ]\n";
  _file_block_SigmaY << "#   5th: progressive ⟨σʸᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_SigmaY << "#   6th: progressive 𝜎[⟨σʸᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_block_SigmaY << "############################################\n";

  _file_block_SigmaZ << "############################################\n";
  _file_block_SigmaZ << "# Column Legend\n";
  _file_block_SigmaZ << "#\n";
  _file_block_SigmaZ << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_block_SigmaZ << "#   2nd: the 𝐌𝐂-block identifier\n";
  _file_block_SigmaZ << "#   3rd: progressive ⟨σᶻᴿ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_SigmaZ << "#   4th: progressive 𝜎[⟨σᶻᴿ(𝜙,𝜶)⟩_𝛹 ]\n";
  _file_block_SigmaZ << "#   5th: progressive ⟨σᶻᴵ(𝜙,𝜶)⟩_𝛹\n";
  _file_block_SigmaZ << "#   6th: progressive 𝜎[⟨σᶻᴵ(𝜙,𝜶)⟩_𝛹]\n";
  _file_block_SigmaZ << "############################################\n";

}


void VMC_Sampler :: setFile_opt_Params(std::string info) {

  _write_opt_Params = true;
  _file_opt_Params.open("optimized_parameters_" + info + ".wf");
  if(!_file_opt_Params.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ optimized_" << info << ".wf ›› for writing the optimized set of variational parameters 𝓥." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the optimized set of variational parameters 𝓥 on file ‹‹ optimized_" << info << ".wf ››." << std::endl;

  /*
  _file_opt_Params << "#####################################\n";
  _file_opt_Params << "# Column Legend\n";
  _file_opt_Params << "#\n";
  _file_opt_Params << "#   1st: 𝒱ᴿ\n";
  _file_opt_Params << "#   2nd: 𝒱ᴵ\n";
  _file_opt_Params << "#####################################\n";
  */

}


void VMC_Sampler :: setFile_all_Params(std::string info) {

  _write_all_Params = true;
  _file_all_Params.open("variational_manifold_" + info + ".wf");
  if(!_file_all_Params.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ variational_manifold_" << info << ".wf ›› for writing the set of variational parameters 𝓥 at the end of each 𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the set of variational parameters 𝓥 at the end of each 𝐕𝐌𝐂 step on file ‹‹ variational_manifold_" << info << ".wf ››." << std::endl;

  _file_all_Params << "#####################################\n";
  _file_all_Params << "# Column Legend\n";
  _file_all_Params << "#\n";
  _file_all_Params << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_all_Params << "#   2nd: 𝒱ᴿ\n";
  _file_all_Params << "#   3rd: 𝒱ᴵ\n";
  _file_all_Params << "#####################################\n";

}


void VMC_Sampler :: setFile_QGT_matrix(std::string info) {

  _write_QGT_matrix = true;
  _file_QGT_matrix.open("qgt_matrix_" + info + ".dat");
  if(!_file_QGT_matrix.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_matrix_" << info << ".dat ›› for writing the Quantum Geometric Tensor." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the QGT after each 𝐕𝐌𝐂 step on file ‹‹ qgt_matrix_" << info << ".dat ››." << std::endl;

  _file_QGT_matrix << "#######################################\n";
  _file_QGT_matrix << "# Column Legend\n";
  _file_QGT_matrix << "#\n";
  _file_QGT_matrix << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_QGT_matrix << "#   2nd: the Quantum Geometric Tensor\n";
  _file_QGT_matrix << "#######################################\n";

}


void VMC_Sampler :: setFile_QGT_cond(std::string info) {

  _write_QGT_cond = true;
  _file_QGT_cond.open("qgt_cond_" + info + ".dat");
  if(!_file_QGT_cond.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_cond_" << info << ".dat ›› for writing the Quantum Geometric Tensor condition number." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the QGT condition number after each 𝐕𝐌𝐂 step on file ‹‹ qgt_cond_" << info << ".dat ››." << std::endl;

  _file_QGT_cond << "###########################################################################\n";
  _file_QGT_cond << "# Column Legend\n";
  _file_QGT_cond << "#\n";
  _file_QGT_cond << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_QGT_cond << "#   2nd: the QGT condition number (real part) (no regularization)\n";
  _file_QGT_cond << "#   3rd: the QGT condition number (imaginary part) (no regularization)\n";
  _file_QGT_cond << "#   4th: the QGT condition number (real part) (with regularization)\n";
  _file_QGT_cond << "#   5th: the QGT condition number (imaginary part) (with regularization)\n";
  _file_QGT_cond << "###########################################################################\n";

}


void VMC_Sampler :: setFile_QGT_eigen(std::string info) {

  _write_QGT_eigen = true;
  _file_QGT_eigen.open("qgt_eigen_" + info + ".dat");
  if(!_file_QGT_eigen.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_eigen_" << info << ".dat ›› for writing the eigenvalues of the Quantum Geometric Tensor." << std::endl;
    std::abort();

  }
  else
    std::cout << " Saving the QGT eigenvalues after each 𝐕𝐌𝐂 step on file ‹‹ qgt_eigen_" << info << ".dat ››." << std::endl;

  _file_QGT_eigen << "#####################################\n";
  _file_QGT_eigen << "# Column Legend\n";
  _file_QGT_eigen << "#\n";
  _file_QGT_eigen << "#   1st: the 𝐕𝐌𝐂-step identifier\n";
  _file_QGT_eigen << "#   2nd: the QGT eigenvalues\n";
  _file_QGT_eigen << "#####################################\n";

}


void VMC_Sampler :: Write_Move_Statistics(unsigned int opt_step) {

  _file_Move_Statistics << opt_step + 1;
  _file_Move_Statistics << std::scientific;
  //_file_Move_Statistics << std::setprecision(10) << std::fixed;
  _file_Move_Statistics << "\t" << 100.0*_N_accepted_visible/_N_proposed_visible;
  if(_N_proposed_ket == 0)
    _file_Move_Statistics << "\t" << 0.0;
  else
    _file_Move_Statistics << "\t" << 100.0*_N_accepted_ket/_N_proposed_ket;
  if(_N_proposed_bra == 0)
    _file_Move_Statistics << "\t" << 0.0;
  else
    _file_Move_Statistics << "\t" << 100.0*_N_accepted_bra/_N_proposed_bra;
  if(_N_proposed_equal_site==0)
    _file_Move_Statistics << "\t" << 0.0;
  else
    _file_Move_Statistics << "\t" << 100.0*_N_accepted_equal_site/_N_proposed_equal_site;
  if(_N_proposed_visible_nn_site==0)
    _file_Move_Statistics << "\t" << 0.0;
  else
    _file_Move_Statistics << "\t" << 100.0*_N_accepted_visible_nn_site/_N_proposed_visible_nn_site;
  if(_N_proposed_hidden_nn_site==0)
    _file_Move_Statistics << "\t" << 0.0;
  else
    _file_Move_Statistics << "\t" << 100.0*_N_accepted_hidden_nn_site/_N_proposed_hidden_nn_site << std::endl;
  _file_Move_Statistics << std::endl;

}


void VMC_Sampler :: Write_MCMC_Config(unsigned int mcmc_step) {

  if(_write_MCMC_Config){

    _file_MCMC_Config << mcmc_step + 1;

    //Prints the visible configuration |𝒗⟩
    _file_MCMC_Config << "\t|𝒗 ⟩" << std::setw(4);
    for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

      for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
        _file_MCMC_Config << _configuration(j_row, j_col) << std::setw(4);
      _file_MCMC_Config << std::endl << "   " << std::setw(4);

    }

    //Prints the ket configuration |𝒉⟩
    if(_hidden_ket.is_empty())
      _file_MCMC_Config << "\t|𝒉 ⟩" << std::endl;
    else{

      _file_MCMC_Config << "\t|𝒉 ⟩" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
          _file_MCMC_Config << _hidden_ket(j_row, j_col) << std::setw(4);
        _file_MCMC_Config << std::endl << "   " << std::setw(4);

      }

    }

    //Prints the bra configuration ⟨𝒉ˈ|
    if(_hidden_bra.is_empty())
      _file_MCMC_Config << "\t⟨𝒉ˈ|" << std::endl;
    else{

      _file_MCMC_Config << "\t⟨𝒉ˈ|" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_bra.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_bra.n_cols; j_col++)
          _file_MCMC_Config << _hidden_bra(j_row, j_col) << std::setw(4);
        _file_MCMC_Config << std::endl;

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_final_Config(unsigned int opt_step) {

  if(_write_final_Config){

    _file_final_Config << opt_step + 1 << "\t|𝒗 ⟩" << std::setw(4);
    //Prints the visible configuration |𝒗 ⟩
    for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

      for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
        _file_final_Config << _configuration(j_row, j_col) << std::setw(4);
      _file_final_Config << std::endl << "   " << std::setw(4);

    }

    //Prints the ket configuration |𝒉 ⟩
    if(_hidden_ket.is_empty())
      _file_final_Config << "\t|𝒉 ⟩" << std::endl;
    else{

      _file_final_Config << "\t|𝒉 ⟩" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
          _file_final_Config << _hidden_ket(j_row, j_col) << std::setw(4);
        _file_final_Config << std::endl;;

      }

    }

    //Prints the bra configuration ⟨𝒉ˈ|
    if(_hidden_bra.is_empty())
      _file_final_Config << "\t⟨𝒉ˈ|" << std::endl;
    else{

      _file_final_Config << "\t⟨𝒉ˈ|" << std::setw(4);
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


void VMC_Sampler :: Write_opt_Params() {

  if(_write_opt_Params){

    if(!_if_shadow)
      _file_opt_Params << _vqs.n_visible() << "\n" << _vqs.density()*_vqs.n_visible() << std::endl;
    else
      _file_opt_Params << _vqs.n_visible() << "\n" << _vqs.phi() << std::endl;

    for(unsigned int p = 0; p < _vqs.n_alpha(); p++)
      _file_opt_Params << _vqs.alpha_at(p).real() << " " << _vqs.alpha_at(p).imag() << std::endl;

  }
  else
    return;

}


void VMC_Sampler :: Write_all_Params(unsigned int opt_step) {

  if(_write_all_Params){

    _file_all_Params << opt_step + 1 << " " << _vqs.phi().real() << " " << _vqs.phi().imag() << std::endl;
    for(unsigned int p = 0; p < _vqs.n_alpha(); p++)
      _file_all_Params << opt_step + 1 << " " << _vqs.alpha_at(p).real() << " " << _vqs.alpha_at(p).imag() << std::endl;
    //_file_all_Params << std::endl;

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_matrix(unsigned int opt_step) {

  if(_write_QGT_matrix){

    _file_QGT_matrix << opt_step + 1 << "\t";
    _file_QGT_matrix << std::setprecision(20) << std::fixed;
    if(_if_shadow){

      for(unsigned int j = 0; j < _Q.row(0).n_elem; j++)
        _file_QGT_matrix << _Q.row(0)(j).real() << " ";
      _file_QGT_matrix << std::endl;
      for(unsigned int d = 1; d < _Q.n_rows; d++){

        _file_QGT_matrix << "\t";
        for(unsigned int j = 0; j < _Q.row(d).n_elem; j++)
          _file_QGT_matrix << _Q.row(d)(j).real() << " ";
        _file_QGT_matrix << std::endl;

      }

    }
    else{

      for(unsigned int j = 0; j < _Q.row(0).n_elem; j++)
        _file_QGT_matrix << _Q.row(0)(j) << " ";
      _file_QGT_matrix << std::endl;
      for(unsigned int d = 1; d < _Q.n_rows; d++){

        _file_QGT_matrix << "\t";
        for(unsigned int j = 0; j < _Q.row(d).n_elem; j++)
          _file_QGT_matrix << _Q.row(d)(j) << " ";
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
      if(_if_QGT_reg)
        eigenval = eig_sym(real(_Q) + _eps * _I);
      else
        eigenval = eig_sym(real(_Q));
      for(unsigned int e = 0; e < eigenval.n_elem; e++)
        _file_QGT_eigen << opt_step + 1 << " " << eigenval(e) << "\n";

    }
    else{

      cx_vec eigenval;
      if(_if_QGT_reg)
        eigenval = eig_gen(_Q + _eps * _I);
      else
        eigenval = eig_gen(_Q);
      for(unsigned int e = 0; e < eigenval.n_elem; e++)
        _file_QGT_eigen << opt_step + 1 << " " << eigenval(e) << "\n";

    }

  }
  else
    return;

}


void VMC_Sampler :: CloseFile() {

  if(_write_Move_Statistics)
    _file_Move_Statistics.close();
  if(_write_MCMC_Config)
    _file_MCMC_Config.close();
  if(_write_final_Config)
    _file_final_Config.close();
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
  if(_write_all_Params)
    _file_all_Params.close();
  if(_write_opt_Params)
    _file_opt_Params.close();
  if(_write_QGT_matrix)
    _file_QGT_matrix.close();
  if(_write_QGT_cond)
    _file_QGT_cond.close();
  if(_write_QGT_eigen)
    _file_QGT_eigen.close();

}


void VMC_Sampler :: Reset() {

  /*##########################################################*/
  //  This function must be called every time a new 𝐕𝐌𝐂 step
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
  //  configuration |𝒮⟩ is sampled via the Metropolis-Hastings test,
  //  an instantaneous evaluation of a certain system properties, represented by
  //  a self-adjoint operator 𝔸, can be done by evaluating the Monte Carlo average
  //  of the instantaneous local observables 𝒜, defined as:
  //
  //        𝒜 ≡ 𝒜(𝒗) = Σ𝒗' ⟨𝒗|𝔸|𝒗'⟩•Ψ(𝒗',𝜶)/Ψ(𝒗,𝜶)        (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝒜 ≡ 𝒜(𝒗,𝒉) = Σ𝒗' ⟨𝒗|𝔸|𝒗'⟩•Φ(𝒗',𝒉,𝜶)/Φ(𝒗,𝒉,𝜶)  (𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  where the matrix elements ⟨𝒗|𝔸|𝒗'⟩ are the connections of the
  //  quantum observable operator 𝔸 related to the visible configuration |𝒗⟩ and
  //  the |𝒗'⟩ configurations are all the system configurations connected to |𝒗⟩.
  //  Whereupon, we can compute the Monte Carlo average value of 𝐀𝐍𝐘 quantum
  //  observable 𝔸 on the variational state as
  //
  //        ⟨𝔸⟩ = ⟨𝒜⟩             (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        ⟨𝔸⟩ = ≪𝒜ᴿ≫ + ⌈𝒜ᴵ⌋   (𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  Therefore, this function has the task of calculating and saving in memory the instantaneous
  //  values of the quantities of interest that allow to estimate, in a second moment, the (Monte Carlo)
  //  average properties, whenever a new configuration is sampled.
  //  To this end, _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭 and _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚 are matrices, whose rows keep in memory the
  //  instantaneous values of the various observables that we want to calculate (𝒜(𝒗,𝒉) and 𝒜(𝒗,𝒉ˈ)),
  //  and it will have as many columns as the number of points (i.e. the sampled configuration |𝒮⟩)
  //  that form the MCMC on which these instantaneous values are calculated.
  //  The function also calculates the values of the local operators
  //
  //        𝕆(𝒗,𝒉) = ∂𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))/∂𝛂
  //        𝕆(𝒗,𝒉ˈ) = ∂𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))/∂𝛂
  //
  //  related to the variational state on the current sampled configuration |𝒮⟩.
  //  The instantaneous values of 𝑐𝑜𝑠𝐼𝐼 and 𝑠𝑖𝑛𝐼𝐼 are stored in the rows of the matrix _𝐢𝐧𝐬𝐭𝐑𝐞𝐰𝐞𝐢𝐠𝐡𝐭 and
  //  all the above computations will be combined together in the 𝐄𝐬𝐭𝐢𝐦𝐚𝐭𝐞() function in order to
  //  obtained the desired Monte Carlo estimation.
  //
  //  N̲O̲T̲E̲: in the case of the Shadow wave function it may be necessary to make many more
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
  _H.FindConn(_configuration, _StatePrime, _Connections);  // ⟨𝒗|𝔸|𝒗'⟩ for all |𝒗'⟩

  //Function variables
  unsigned int n_props = _Connections.n_rows;  //Number of quantum observables
  _Observables.set_size(n_props, 1);  //Only sizing, this should be computed in 𝐄𝐬𝐭𝐢𝐦𝐚𝐭𝐞()
  Col <double> cosin(2, fill::zeros);  //Storage variable for cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')] and sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
  Col <std::complex <double>> A_ket(n_props, fill::zeros);  //Storage value for 𝒜(𝒗,𝒉)
  Col <std::complex <double>> A_bra(n_props, fill::zeros);  //Storage value for 𝒜(𝒗,𝒉ˈ)
  Col <std::complex <double>> O_ket(_vqs.n_alpha(), fill::zeros);
  Col <std::complex <double>> O_bra(_vqs.n_alpha(), fill::zeros);

  //Makes the Shadow measurement less noisy
  if(_if_extra_hidden_sum){

    //Extra sampling of the hidden variables
    if(_Nblks_extra == 0){

      std::cerr << " ##ValueError: not to use “block averaging” during the extra hidden sampling set _Nblks_extra = 𝟣." << std::endl;
      std::cerr << "   Failed to measure instantaneous quantum properties of the system." << std::endl;
      std::abort();

    }
    else if(_Nblks_extra == 1){  //No “block averaging”

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

            A_ket(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_ket);  // 𝒜(𝒗,𝒉)
            A_bra(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_bra);  // 𝒜(𝒗,𝒉')

          }

        }

      }
      cosin /= double(_Nextra);  //  ⟨⟨𝑐𝑜𝑠⟩ᵇˡᵏ⟩ & ⟨⟨𝑠𝑖𝑛⟩ᵇˡᵏ⟩
      A_ket /= double(_Nextra);  //  ⟨⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ⟩
      A_bra /= double(_Nextra);  //  ⟨⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ⟩
      O_ket /= double(_Nextra);  //  ⟨⟨𝕆(𝒗,𝒉)⟩ᵇˡᵏ⟩
      O_bra /= double(_Nextra);  //  ⟨⟨𝕆(𝒗,𝒉')⟩ᵇˡᵏ⟩

    }
    else{  //“block averaging”

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

              A_ket_blk(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_ket);  // 𝒜(𝒗,𝒉)
              A_bra_blk(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_bra);  // 𝒜(𝒗,𝒉')

            }

          }

        }
        cosin(0) += cos_blk / double(blk_size);  // ⟨𝑐𝑜𝑠⟩ᵇˡᵏ
        cosin(1) += sin_blk / double(blk_size);  // ⟨𝑠𝑖𝑛⟩ᵇˡᵏ
        A_ket += A_ket_blk / double(blk_size);  //  ⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ
        A_bra += A_bra_blk / double(blk_size);  //  ⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ
        O_ket += O_ket_blk / double(blk_size);  //  ⟨𝕆(𝒗,𝒉)⟩ᵇˡᵏ
        O_bra += O_bra_blk / double(blk_size);  //  ⟨𝕆(𝒗,𝒉')⟩ᵇˡᵏ

      }
      cosin /= double(_Nblks_extra);  //  ⟨⟨𝑐𝑜𝑠⟩ᵇˡᵏ⟩ & ⟨⟨𝑠𝑖𝑛⟩ᵇˡᵏ⟩
      A_ket /= double(_Nblks_extra);  //  ⟨⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ⟩
      A_bra /= double(_Nblks_extra);  //  ⟨⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ⟩
      O_ket /= double(_Nblks_extra);  //  ⟨⟨𝕆(𝒗,𝒉)⟩ᵇˡᵏ⟩
      O_bra /= double(_Nblks_extra);  //  ⟨⟨𝕆(𝒗,𝒉')⟩ᵇˡᵏ⟩

    }

  }
  else{

    //Computes cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')] and sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cosin(0) = _vqs.cosII(_configuration, _hidden_ket, _hidden_bra);
    cosin(1) = _vqs.sinII(_configuration, _hidden_ket, _hidden_bra);

    //Instantaneous evaluation of the quantum observables
    for(unsigned int Nobs = 0; Nobs < n_props; Nobs++){

      for(unsigned int mel = 0; mel < _Connections(Nobs).n_elem; mel++){

        A_ket(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_ket);  // 𝒜(𝒗,𝒉)
        A_bra(Nobs) += _Connections(Nobs, 0)(mel) * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime(Nobs, 0)(0, mel), _hidden_bra);  // 𝒜(𝒗,𝒉')

      }

    }

    //Instantaneous evaluation of the local operators
    _vqs.LocalOperators(_configuration, _hidden_ket, _hidden_bra);  //Computes 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉')
    O_ket = _vqs.O().col(0);
    O_bra = _vqs.O().col(1);

  }

  //Adds Monte Carlo statistics
  _instReweight.insert_cols(_instReweight.n_cols, cosin);  // ≡ instantaneous measure of the 𝑐𝑜𝑠 and of the 𝑠𝑖𝑛
  _instObs_ket.insert_cols(_instObs_ket.n_cols, A_ket);  // ≡ instantaneous measure of 𝒜(𝒗,𝒉)
  _instObs_bra.insert_cols(_instObs_bra.n_cols, A_bra);  // ≡ instantaneous measure of 𝒜(𝒗,𝒉')
  _instO_ket.insert_cols(_instO_ket.n_cols, O_ket);  // ≡ instantaneous measure of 𝕆(𝒗,𝒉)
  _instO_bra.insert_cols(_instO_bra.n_cols, O_bra);  // ≡ instantaneous measure of 𝕆(𝒗,𝒉')

}


void VMC_Sampler :: Estimate() {

  /*#############################################################################################*/
  //  This function is called at the end of the single VMC step and
  //  estimates the averages of the quantum observables
  //  as a Monte Carlo stochastic mean value on the choosen variational quantum state, i.e.:
  //
  //        ⟨𝔸⟩ = ⟨𝒜⟩             (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        ⟨𝔸⟩ = ≪𝒜ᴿ≫ + ⌈𝒜ᴵ⌋   (𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  with the relative uncertainties via the Blocking Method.
  //  We define the above special expectation value in the following way:
  //
  //        ≪◦≫ = 1/2•Σ𝒗Σ𝒉Σ𝒉ˈ𝓆(𝒗,𝒉,𝒉ˈ)•𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]•[◦(𝒗,𝒉) + ◦(𝒗,𝒉ˈ)]
  //            = 1/2•⟨𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]•[◦(𝒗,𝒉) + ◦(𝒗,𝒉ˈ)]⟩ / ⟨𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]⟩
  //        ⌈◦⌋ = 1/2•Σ𝒗Σ𝒉Σ𝒉ˈ𝓆(𝒗,𝒉,𝒉ˈ)•𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]•[◦(𝒗,𝒉ˈ) - ◦(𝒗,𝒉)]
  //            = 1/2•⟨𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]•[◦(𝒗,𝒉ˈ) - ◦(𝒗,𝒉)]⟩ / ⟨𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]⟩
  //
  //  in which the standard expectation value ⟨◦⟩ are calculated in a standard way with
  //  the Monte Carlo sampling of 𝓆(𝒗,𝒉,𝒉ˈ), and the normalization given by the cosine
  //  is due to the 𝐫𝐞𝐰𝐞𝐢𝐠𝐡𝐭𝐢𝐧𝐠 technique necessary to correctly estimate the various quantities.
  //  In the non-shadow case we have:
  //
  //        ≪◦≫ → ‹›, i.e. the standard Monte Carlo expectation value
  //        ⌈◦⌋ → 0
  //
  //  The instantaneous values along the single Markov chain necessary to make the Monte Carlo
  //  estimates just defined are computed by the 𝐌𝐞𝐚𝐬𝐮𝐫𝐞() function and are stored in the
  //  following data-members:
  //
  //        _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭  ‹--›  quantum observable 𝒜(𝒗,𝒉)
  //        _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚  ‹--›  quantum observable 𝒜(𝒗,𝒉')
  //        _𝐢𝐧𝐬𝐭𝐑𝐞𝐰𝐞𝐢𝐠𝐡𝐭  ‹--›  𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] & 𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]
  //        _𝐢𝐧𝐬𝐭𝐎_𝐤𝐞𝐭  ‹--›  𝕆(𝒗,𝒉)
  //        _𝐢𝐧𝐬𝐭𝐎_𝐛𝐫𝐚  ‹--›  𝕆(𝒗,𝒉')
  //
  //  The Quantum Geometric Tensor and the energy gradient required to optimize the variational
  //  parameters are also (stochastically) computed.
  /*#############################################################################################*/

  //Computes all necessary MC block estimates without yet adjusting with the reweighting ratio
  this -> compute_Reweighting_ratio();
  this -> compute_Quantum_observables();

  //Computes all stuff for the update of variational parameters
  if(!_if_vmc){

    this -> compute_O();
    this -> compute_QGTandGrad();
    this -> QGT_Check();

  }

}


void VMC_Sampler :: Write_Quantum_properties(unsigned int tdvmc_step) {

  /*############################################################*/
  //  We save on the output file the real and imaginary part
  //  with the relative uncertainties of the
  //  quantum observables via "block averaging": if everything
  //  has gone well, the imaginary part of the estimates of
  //  quantum operators MUST be statistically zero.
  /*############################################################*/

  //Computes progressive averages of the reweighting ratio with "block averaging" uncertainties
  Col <double> prog_cos = this -> compute_progressive_averages(_cosII);
  Col <double> err_cos = this -> compute_errorbar(_cosII);
  Col <double> prog_sin = this -> compute_progressive_averages(_sinII);
  Col <double> err_sin = this -> compute_errorbar(_sinII);

  if(!_if_shadow){

    //Computes progressive averages of quantum observables with "block averaging" uncertainties
    Col <std::complex <double>> prog_energy = this -> compute_progressive_averages(_Observables(0, 0));
    Col <std::complex <double>> err_energy = this -> compute_errorbar(_Observables(0, 0));
    Col <std::complex <double>> prog_Sx = this -> compute_progressive_averages(_Observables(1, 0));
    Col <std::complex <double>> err_Sx = this -> compute_errorbar(_Observables(1, 0));
    Col <std::complex <double>> prog_Sy = this -> compute_progressive_averages(_Observables(2, 0));
    Col <std::complex <double>> err_Sy = this -> compute_errorbar(_Observables(2, 0));
    Col <std::complex <double>> prog_Sz = this -> compute_progressive_averages(_Observables(3, 0));
    Col <std::complex <double>> err_Sz = this -> compute_errorbar(_Observables(3, 0));

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

        //Writes σ̂ˣ
        _file_block_SigmaX << std::setprecision(10) << std::fixed;
        _file_block_SigmaX << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
        _file_block_SigmaX << prog_Sx(block_ID).real() << "\t" << err_Sx(block_ID).real() << "\t";
        _file_block_SigmaX << prog_Sx(block_ID).imag() << "\t" << err_Sx(block_ID).imag() << "\t";
        _file_block_SigmaX << std::endl;

        //Writes block σ̂ʸ
        _file_block_SigmaY << std::setprecision(10) << std::fixed;
        _file_block_SigmaY << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
        _file_block_SigmaY << prog_Sy(block_ID).real() << "\t" << err_Sy(block_ID).real() << "\t";
        _file_block_SigmaY << prog_Sy(block_ID).imag() << "\t" << err_Sy(block_ID).imag() << "\t";
        _file_block_SigmaY << std::endl;

        //Writes block σ̂ᶻ
        _file_block_SigmaZ << std::setprecision(10) << std::fixed;
        _file_block_SigmaZ << tdvmc_step + 1 << "\t" << block_ID + 1 << "\t";
        _file_block_SigmaZ << prog_Sz(block_ID).real() << "\t" << err_Sz(block_ID).real() << "\t";
        _file_block_SigmaZ << prog_Sz(block_ID).imag() << "\t" << err_Sz(block_ID).imag() << "\t";
        _file_block_SigmaZ << std::endl;

      }

    }

    //Saves optimized quantum observables along the 𝐓𝐃𝐕𝐌𝐂
    if(_write_opt_Observables){

      // 𝐸(𝜙,𝜶) +/- 𝓈𝓉𝒹[𝐸(𝜙,𝜶)]
      _file_opt_Energy << std::setprecision(20) << std::fixed;
      _file_opt_Energy << tdvmc_step + 1 << "\t";
      _file_opt_Energy << prog_cos(_Nblks - 1) << "\t" << err_cos(_Nblks - 1) << "\t";
      _file_opt_Energy << prog_sin(_Nblks - 1) << "\t" << err_sin(_Nblks - 1) << "\t";
      _file_opt_Energy << prog_energy(_Nblks - 1).real() << "\t" << err_energy(_Nblks - 1).real() << "\t";
      _file_opt_Energy << prog_energy(_Nblks - 1).imag() << "\t" << err_energy(_Nblks - 1).imag() << "\t";
      _file_opt_Energy << _E.real() << "\t" << _E.imag();
      _file_opt_Energy << std::endl;

      // 𝝈(𝜙,𝜶) +/- 𝓈𝓉𝒹[𝝈(𝜙, 𝜶)]
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
    Col <double> shadow_energy = real(_Observables(0, 0)) / _cosII;  //Computes ⟨ℋ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
    Col <double> shadow_Sx = real(_Observables(1, 0)) / _cosII;  //Computes ⟨σ̂ˣ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
    Col <double> shadow_Sy = real(_Observables(2, 0)) / _cosII;  //Computes ⟨σ̂ʸ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
    Col <double> shadow_Sz = real(_Observables(3, 0)) / _cosII;  //Computes ⟨σ̂ᶻ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block

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

        //Writes σ̂ˣ
        _file_block_SigmaX << std::setprecision(10) << std::fixed;
        _file_block_SigmaX << tdvmc_step + 1 << " " << block_ID + 1 << " ";
        _file_block_SigmaX << prog_Sx(block_ID) << " " << err_Sx(block_ID) << " ";
        _file_block_SigmaX << 0.0 << " " << 0.0 << " ";
        _file_block_SigmaX << std::endl;

        //Writes block σ̂ʸ
        _file_block_SigmaY << std::setprecision(10) << std::fixed;
        _file_block_SigmaY << tdvmc_step + 1 << " " << block_ID + 1 << " ";
        _file_block_SigmaY << prog_Sy(block_ID) << " " << err_Sy(block_ID) << " ";
        _file_block_SigmaY << 0.0 << " " << 0.0 << " ";
        _file_block_SigmaY << std::endl;

        //Writes block σ̂ᶻ
        _file_block_SigmaZ << std::setprecision(10) << std::fixed;
        _file_block_SigmaZ << tdvmc_step + 1 << " " << block_ID + 1 << " ";
        _file_block_SigmaZ << prog_Sz(block_ID) << " " << err_Sz(block_ID) << " ";
        _file_block_SigmaZ << 0.0 << " " << 0.0 << " ";
        _file_block_SigmaZ << std::endl;

      }

    }

    //Saves optimized quantum observables along the 𝐓𝐃𝐕𝐌𝐂
    if(_write_opt_Observables){

      // 𝐸(𝜙,𝜶) +/- 𝓈𝓉𝒹[𝐸(𝜙,𝜶)]
      _file_opt_Energy << std::setprecision(20) << std::fixed;
      _file_opt_Energy << tdvmc_step + 1 << " ";
      _file_opt_Energy << prog_cos(_Nblks - 1) << " " << err_cos(_Nblks - 1) << " ";
      _file_opt_Energy << prog_sin(_Nblks - 1) << " " << err_sin(_Nblks - 1) << " ";
      _file_opt_Energy << prog_energy(_Nblks - 1) << " " << err_energy(_Nblks - 1) << " ";
      _file_opt_Energy << 0.0 << " " << 0.0 << " " << _E.real() << " " << _E.imag();
      _file_opt_Energy << std::endl;

      // 𝝈(𝜙,𝜶) +/- 𝓈𝓉𝒹[𝝈(𝜙, 𝜶)]
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
  //        ⟨𝔸⟩ᵇˡᵏ = ≪𝒜ᴿ≫ᵇˡᵏ + ⌈𝒜ᴵ⌋ᵇˡᵏ
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
  //        ≪𝒜ᴿ≫ᵇˡᵏ
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_Nblks));  //Sets the block length
  Col <double> blocks_angled_quantity(_Nblks);
  double angled_sum_in_each_block;

  //Computes Monte Carlo Shadow “angled” averages in each block ( ! without the reweighting ration ! )
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
  //        ⌈𝒜ᴵ⌋ᵇˡᵏ
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  unsigned int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_Nblks));  //Sets the block length
  Col <double> blocks_square_quantity(_Nblks);
  double square_sum_in_each_block;

  //Computes Monte Carlo Shadow “square” averages in each block ( ! without the reweighting ration ! )
  for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

    square_sum_in_each_block = 0.0;
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
      square_sum_in_each_block += _instReweight.row(1)(l) * (instantaneous_quantity_bra(l).imag() - instantaneous_quantity_ket(l).imag());
    square_sum_in_each_block *= 0.5;
    blocks_square_quantity(block_ID) = square_sum_in_each_block / double(blk_size);

  }

  return blocks_square_quantity;

}


void VMC_Sampler :: compute_Reweighting_ratio() {

  _cosII = this -> average_in_blocks(_instReweight.row(0));  //Computes ⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block, for j = 𝟣,…,𝖭ᵇˡᵏ
  _sinII = this -> average_in_blocks(_instReweight.row(1));  //Computes ⟨𝑠𝑖𝑛⟩ⱼᵇˡᵏ in each block, for j = 𝟣,…,𝖭ᵇˡᵏ

}


void VMC_Sampler :: compute_Quantum_observables() {

  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐄𝐧𝐞𝐫𝐠𝐲.
  //  We compute the stochastic average via the Blocking technique of
  //
  //        𝐸(𝜙,𝜶) = ⟨ℋ⟩ ≈ ⟨ℰ⟩            (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝐸(𝜙,𝜶) = ⟨ℋ⟩ ≈ ≪ℰᴿ≫ + ⌈ℰᴵ⌋   (𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  We remember that the matrix rows _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭(0) and _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚(0) contains
  //  the instantaneous values of the Hamiltonian operator along the MCMC, i.e.
  //  ℰ(𝒗,𝒉) and ℰ(𝒗,𝒉ˈ).
  /*#################################################################################*/
  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐒𝐢𝐧𝐠𝐥𝐞 𝐒𝐩𝐢𝐧 𝐎𝐛𝐬𝐞𝐫𝐯𝐚𝐛𝐥𝐞𝐬.
  //  We compute the stochastic average via the Blocking technique of
  //
  //        𝝈ˣ(𝜙,𝜶) = ⟨𝞼ˣ⟩ ≈ ⟨𝜎ˣ⟩             (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝝈ʸ(𝜙,𝜶) = ⟨𝞼ʸ⟩ ≈ ⟨𝜎ʸ⟩
  //        𝝈ᶻ(𝜙,𝜶) = ⟨𝞼ᶻ⟩ ≈ ⟨𝜎ᶻ⟩
  //        𝝈ˣ(𝜙,𝜶) = ⟨𝞼ˣ⟩ ≈ ≪𝜎ˣᴿ≫ + ⌈𝜎ˣᴵ⌋   (𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝝈ʸ(𝜙,𝜶) = ⟨𝞼ʸ⟩ ≈ ≪𝜎ʸᴿ≫ + ⌈𝜎ʸᴵ⌋
  //        𝝈ᶻ(𝜙,𝜶) = ⟨𝞼ᶻ⟩ ≈ ≪𝜎ᶻᴿ≫ + ⌈𝜎ᶻᴵ⌋
  //
  //  We remember that the matrix rows _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭(f) and _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚(f) contains
  //  the instantaneous values of the spin projection operator along the MCMC, i.e.
  //  𝜎(𝒗,𝒉) and 𝜎(𝒗,𝒉ˈ), with f = 1, 2, 3 and where {σ̂ᶠ} the are Pauli matrices
  //  in the computational basis.
  /*#################################################################################*/

  //Computes ⟨𝒪⟩ⱼᵇˡᵏ in each block
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

}


Col <double> VMC_Sampler :: compute_errorbar(const Col <double>& block_averages) const {

  /*################################################################*/
  //  Computes the statistical uncertainties of a certain quantity
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
  //  This calculation involves a real-valued quantity.
  /*################################################################*/

  //Function variables
  Col <double> errors(block_averages.n_elem);
  Col <double> squared_block_averages;  // ⟨◦⟩ⱼᵇˡᵏ • ⟨◦⟩ⱼᵇˡᵏ
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
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
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
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
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
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
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

      _O(lo_ID, 0) = this -> average_in_blocks(_instO_ket.row(lo_ID));  // ⟨𝕆ₖ⟩ⱼᵇˡᵏ
      _O(lo_ID, 1) = this -> average_in_blocks(conj(_instO_ket.row(lo_ID)));  // ⟨𝕆⋆ₖ⟩ⱼᵇˡᵏ

    }

  }
  else{

    for(unsigned int lo_ID = 0; lo_ID < _O.n_rows; lo_ID++){

      //Computes ≪𝕆ₖ≫ⱼᵇˡᵏ
      _O(lo_ID, 0).set_size(_Nblks);
      _O(lo_ID, 0).set_real(this -> Shadow_angled_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O(lo_ID, 0).set_imag(zeros(_Nblks));

      //Computes ⌈𝕆ₖ⌋ⱼᵇˡᵏ
      _O(lo_ID, 1).set_size(_Nblks);
      _O(lo_ID, 1).set_real(this -> Shadow_square_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O(lo_ID, 1).set_imag(zeros(_Nblks));

    }

  }

}


void VMC_Sampler :: compute_QGTandGrad() {

  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐐𝐮𝐚𝐧𝐭𝐮𝐦 𝐆𝐞𝐨𝐦𝐞𝐭𝐫𝐢𝐜 𝐓𝐞𝐧𝐬𝐨𝐫.
  //  We compute stochastically the 𝐐𝐆𝐓 defined as
  //
  //        ℚ = 𝙎ₘₙ                                  (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝙎ₘₙ ≈ ⟨𝕆⋆ₘ𝕆ₙ⟩ - ⟨𝕆⋆ₘ⟩•⟨𝕆ₙ⟩.
  //
  //        ℚ = 𝙎 + 𝘼•𝘽•𝘼                            (𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝙎ₘₙ ≈ ≪𝕆ₘ𝕆ₙ≫ - ≪𝕆ₘ≫•≪𝕆ₙ≫ - ⌈𝕆ₘ⌋⌈𝕆ₙ⌋
  //        𝘼ₘₙ ≈ -⌈𝕆ₘ𝕆ₙ⌋ + ⌈𝕆ₘ⌋≪𝕆ₙ≫ - ≪𝕆ₘ≫⌈𝕆ₙ⌋
  //        where 𝘽 is the inverse matrix of 𝙎.
  /*#################################################################################*/
  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐄𝐧𝐞𝐫𝐠𝐲 𝐆𝐫𝐚𝐝𝐢𝐞𝐧𝐭.
  //  We compute stochastically the Gradient which drive the optimization defined as
  //
  //        𝔽ₖ ≈ ⟨ℰ𝕆⋆ₖ⟩ - ⟨ℰ⟩•⟨𝕆⋆ₖ⟩                  (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //        𝔽ᴿ ≈ 𝞒 - 𝘼•𝘽•𝞨                           (𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝔽ᴵ ≈ 𝞨 + 𝘼•𝘽•𝞒
  //
  //  with
  //
  //        𝞒ₖ ≈ -⟨ℋ⟩•⌈𝕆ₖ⌋ + ≪𝕆ₖ•ℰᴵ≫ + ⌈𝕆ₖ•ℰᴿ⌋
  //        𝞨ₖ ≈ ⟨ℋ⟩•≪𝕆ₖ≫ + ⌈𝕆ₖ•ℰᴵ⌋ - ≪𝕆ₖ•ℰᴿ≫
  //
  //  where 𝘼 and 𝘽 are introduced before in the calculation of ℚ.
  /*#################################################################################*/

  //Function variables
  unsigned int n_alpha = _vqs.n_alpha();
  unsigned int blk_size = std::floor(double(_Nsweeps/_Nblks));  //Sets the block length
  _Q.zeros(n_alpha, n_alpha);
  _F.zeros(n_alpha);

  if(!_if_shadow){

    Col <std::complex <double>> mean_O(n_alpha);  // ⟨⟨𝕆ₖ⟩ᵇˡᵏ⟩
    Col <std::complex <double>> mean_O_star(n_alpha);  // ⟨⟨𝕆⋆ₖ⟩ᵇˡᵏ⟩
    std::complex <double> block_qgt, block_gradE;

    for(unsigned int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O(lo_ID) = mean(_O(lo_ID, 0));
      mean_O_star(lo_ID) = mean(_O(lo_ID, 1));

    }

    //Computes 𝐸(𝜙,𝜶) = ⟨ℋ⟩ stochastically without progressive errorbars
    _E = mean(_Observables(0, 0));  // ⟨⟨ℋ⟩ᵇˡᵏ⟩

    //Computes ℚ = 𝙎ₘₙ stochastically without progressive errorbars
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = 0; n < n_alpha; n++){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          block_qgt = 0.0;
          for(unsigned int l = block_ID * blk_size; l < (block_ID +  1) * blk_size; l++)
            block_qgt += std::conj(_instO_ket(m, l)) * _instO_ket(n, l);  //Accumulate 𝕆⋆ₘ𝕆ₙ in each block
          _Q(m, n) += block_qgt / double(blk_size) - mean_O_star(m) * mean_O(n);  // ⟨𝙎ₘₙ⟩ᵇˡᵏ

        }

      }

    }
    _Q /= double(_Nblks);  // ⟨ℚ⟩ ≈ ⟨⟨𝙎ₘₙ⟩ᵇˡᵏ⟩

    //Computes 𝔽 = 𝔽ₖ stochastically without progressive errorbars
    for(unsigned int k = 0; k < n_alpha; k++){

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_gradE = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
          block_gradE += _instObs_ket(0, l) * std::conj(_instO_ket(k, l));  //Accumulate ℰ𝕆⋆ₖ in each block
        _F(k) += block_gradE / double(blk_size) - _E * mean_O_star(k);  // ⟨𝔽ₖ⟩ᵇˡᵏ

      }

    }
    _F /= double(_Nblks);  // ⟨𝔽⟩ ≈ ⟨⟨𝔽ₖ⟩ᵇˡᵏ⟩

  }
  else{

    _mean_O_angled.set_size(n_alpha);  // ⟨≪𝕆ₖ≫ᵇˡᵏ⟩ with reweighting correction
    _mean_O_square.set_size(n_alpha);  // ⟨⌈𝕆ₖ⌋ᵇˡᵏ⟩ with reweighting correction
    Mat <double> S(n_alpha, n_alpha, fill::zeros);  // 𝙎ₘₙ ≈ ≪𝕆ₘ𝕆ₙ≫ - ≪𝕆ₘ≫•≪𝕆ₙ≫ - ⌈𝕆ₘ⌋⌈𝕆ₙ⌋
    Mat <double> A(n_alpha, n_alpha, fill::zeros);  // 𝘼ₘₙ ≈ -⌈𝕆ₘ𝕆ₙ⌋ + ⌈𝕆ₘ⌋≪𝕆ₙ≫ - ≪𝕆ₘ≫⌈𝕆ₙ⌋
    Col <double> Gamma(n_alpha, fill::zeros);  // 𝞒ₖ ≈ -⟨ℋ⟩•⌈𝕆ₖ⌋ + ≪𝕆ₖ•ℰᴵ≫ + ⌈𝕆ₖ•ℰᴿ⌋
    Col <double> Omega(n_alpha, fill::zeros);  // 𝞨ₖ ≈ ⟨ℋ⟩•≪𝕆ₖ≫ + ⌈𝕆ₖ•ℰᴵ⌋ - ≪𝕆ₖ•ℰᴿ≫
    double block_corr_angled, block_corr_square;
    double mean_cos = mean(_cosII);

    for(unsigned int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      _mean_O_angled(lo_ID) = mean(real(_O(lo_ID, 0))) / mean_cos;
      _mean_O_square(lo_ID) = mean(real(_O(lo_ID, 1))) / mean_cos;

    }

    //Computes 𝐸(𝜙,𝜶) = ⟨ℋ⟩ stochastically without progressive errorbars
    _E.real(mean(real(_Observables(0, 0))) / mean_cos);  // ⟨⟨ℋ⟩ᵇˡᵏ⟩ with reweighting correction
    _E.imag(0.0);

    //Computes ℚ = 𝙎 + 𝘼•𝘽•𝘼 stochastically without progressive errorbars
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = m; n < n_alpha; n++){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          block_corr_angled = 0.0;
          block_corr_square = 0.0;
          for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

            //Accumulate 𝕆ₘ𝕆ₙ in each block (angled part)
            block_corr_angled += _instReweight(0, l) * (_instO_ket(m, l).real() * _instO_bra(n, l).real() + _instO_bra(m, l).real() * _instO_ket(n, l).real());
            //Accumulate 𝕆ₘ𝕆ₙ in each block (square part)
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
    S /= double(_Nblks);  // ⟨⟨≪𝕆ₘ𝕆ₙ≫ᵇˡᵏ⟩⟩ without reweighting correction
    A /= double(_Nblks);  // ⟨⟨⌈𝕆ₘ𝕆ₙ⌋ᵇˡᵏ⟩⟩ without reweighting correction
    S /= mean_cos;
    A /= mean_cos;
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = 0; n < n_alpha; n++){

        S(m, n) -= (_mean_O_angled(m) * _mean_O_angled(n) + _mean_O_square(m) * _mean_O_square(n));  // ⟨𝙎ₘₙ⟩ with reweighting correction
        A(m, n) += (_mean_O_square(m) * _mean_O_angled(n) - _mean_O_angled(m) * _mean_O_square(n));  // ⟨𝘼ₘₙ⟩ with reweighting correction

      }

    }
    if(_if_QGT_reg)
      S = S + _eps * _I;
    Mat <double> AB = A * pinv(S);
    _Q.set_real(symmatu(S + AB * A));  // ⟨ℚ⟩ ≈ ⟨⟨𝙎 + 𝘼•𝘽•𝘼⟩ᵇˡᵏ⟩

    //Computes 𝔽 = {𝔽ᴿ, 𝔽ᴵ} stochastically without progressive errorbars
    for(unsigned int k = 0; k < n_alpha; k++){  //Computes ⟨𝞒ₖ⟩ᵇˡᵏ

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate 𝕆ₖ•ℰᴵ in each block (angled part)
          block_corr_angled += _instReweight(0, l) * (_instO_ket(k, l).real() * _instObs_bra(0, l).imag() + _instO_bra(k, l).real() * _instObs_ket(0, l).imag());
          //Accumulate 𝕆ₖ•ℰᴿ in each block (square part)
          block_corr_square += _instReweight(1, l) * (_instO_bra(k, l).real() * _instObs_ket(0, l).real() - _instO_ket(k, l).real() * _instObs_bra(0, l).real());

        }
        Gamma(k) += 0.5 * (block_corr_angled + block_corr_square) / double(blk_size);

      }

    }
    for(unsigned int k = 0; k < n_alpha; k++){  //Computes ⟨𝞨ₖ⟩ᵇˡᵏ

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate 𝕆ₖ•ℰᴿ in each block (angled part)
          block_corr_angled += _instReweight(0, l) * (_instO_ket(k, l).real() * _instObs_bra(0, l).real() + _instO_bra(k, l).real() * _instObs_ket(0, l).real());
          //Accumulate 𝕆ₖ•ℰᴵ in each block (square part)
          block_corr_square += _instReweight(1, l) * (_instO_bra(k, l).real() * _instObs_ket(0, l).imag() - _instO_ket(k, l).real() * _instObs_bra(0, l).imag());

        }
        Omega(k) += 0.5 * (block_corr_square - block_corr_angled) / double(blk_size);

      }

    }
    Gamma /= double(_Nblks);  // ⟨⟨𝞒ₖ⟩ᵇˡᵏ⟩ without reweighting correction
    Omega /= double(_Nblks);  // ⟨⟨𝞨ₖ⟩ᵇˡᵏ⟩ without reweighting correction
    Gamma /= mean_cos;
    Omega /=  mean_cos;
    Gamma -= _E.real() * _mean_O_square;  // ⟨𝞒ₖ⟩ with reweighting correction
    Omega += _E.real() * _mean_O_angled;  // ⟨𝞨ₖ⟩ with reweighting correction
    _F.set_real(Gamma - AB * Omega);  // ⟨𝔽ᴿ⟩ ≈ ⟨⟨𝞒 - 𝘼•𝘽•𝞨⟩ᵇˡᵏ⟩
    _F.set_imag(Omega + AB * Gamma);  // ⟨𝔽ᴵ⟩ ≈ ⟨⟨𝞨 + 𝘼•𝘽•𝞒⟩ᵇˡᵏ⟩

  }

}


void VMC_Sampler :: QGT_Check() {

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
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  representing it as the list of indeces of the visible flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    flipped_site.set_size(Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++)
      flipped_site(j, 0) = _rnd.Rannyu_INT(0, _Nspin-1);  //Choose a random spin to flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the visible variables have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐯𝐢𝐬𝐢𝐛𝐥𝐞 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
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

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _configuration(0, _flipped_site(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

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
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈ⟩
  //                or
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡 𝐡ˈⁿᵉʷ⟩
  //
  //  representing it as the list of indeces of the hidden flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  The function prevents from flipping the same site more than once.
  /*##############################################################################*/

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    flipped_hidden_site.set_size(Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++)
      flipped_hidden_site(j, 0) = _rnd.Rannyu_INT(0, _Nhidden-1);  //Choose a random spin to flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the hidden variables (ket) have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of auxiliary spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 hidden lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐡𝐢𝐝𝐝𝐞𝐧 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
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

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _hidden_ket(0, _flipped_ket_site(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the hidden variables (bra) have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of auxiliary spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 hidden lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐛𝐫𝐚 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
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

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _hidden_bra(0, _flipped_bra_site(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which the visible and the hidden variables have been
  //  tried to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins on
  //  𝐨𝐧 𝐭𝐡𝐞 𝐬𝐚𝐦𝐞 𝐥𝐚𝐭𝐭𝐢𝐜𝐞 𝐬i𝐭𝐞𝐬.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐯𝐢𝐬𝐢𝐛𝐥𝐞 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
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

        if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

          _configuration(0, _flipped_site(fs_row, 0)) *= -1;
          _hidden_ket(0, _flipped_site(fs_row, 0)) *= -1;
          _hidden_bra(0, _flipped_site(fs_row, 0)) *= -1;

        }
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

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
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  representing it as the list of indeces of the visible flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  If a certain lattice site is selected, 𝐢𝐭𝐬 𝐟𝐢𝐫𝐬𝐭 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫
  //  site it is automatically added to the list of flipped sites.
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Function variables
  unsigned int index_site;

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

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
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the visible variables have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins
  //  with their respective 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫 lattice site.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐧𝐧_𝐬𝐢𝐭𝐞 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
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

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _configuration(0, _flipped_site(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

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
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  representing it as the list of indeces of the hidden flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  If a certain lattice site is selected, 𝐢𝐭𝐬 𝐟𝐢𝐫𝐬𝐭 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫
  //  site it is automatically added to the list of flipped sites.
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Function variables
  unsigned int index_site_ket;
  unsigned int index_site_bra;

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

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
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the hidden variables (both ket and bra)
  //  have been tried to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of auxiliary spins
  //  with their respective 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫 lattice site.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 hidden lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐡𝐢𝐝𝐝𝐞𝐧_𝐧𝐧_𝐬𝐢𝐭𝐞 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩through the Metropolis-Hastings test.
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

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _hidden_ket(0, _flipped_ket_site(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

          /*
            .........
            .........
            .........
          */

        }

      }

      for(unsigned int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum bra configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _hidden_bra(0, _flipped_bra_site(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

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
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒,
  //
  //        |Sⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip. The selected sites will be in general different
  //  for the three different types of variables (𝒗, 𝐡, and 𝐡ˈ).
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


void VMC_Sampler :: VMC_Step() {

  /*###############################################################################################*/
  //  Runs the single optimization step.
  //  We perform the single Variational Monte Carlo optimization run using
  //  the following parameters:
  //
  //        • N̲ˢ̲ʷ̲ᵉ̲ᵉ̲ᵖ̲: is the number of Monte Carlo sweeps.
  //                  In each single MC sweep a bunch of spins is considered,
  //                  randomly chosen and whose dimension is expressed by the variable N̲ᶠ̲ˡ̲ⁱ̲ᵖ̲ˢ̲,
  //                  and it is tried to flip this bunch of spins with the probability defined
  //                  by the Metropolis-Hastings algorithm; this operation is repeated a certain
  //                  number of times in the single sweep, where this certain number is defined
  //                  by the variables M̲; once the new proposed configuration is accepted or not,
  //                  instantaneous quantum properties are measured on that state, and the single
  //                  sweep ends; different Monte Carlo moves are applied in different situations,
  //                  involving all or only some of the visible and/or hidden variables;
  //
  //        • e̲q̲ᵗ̲ⁱ̲ᵐ̲ᵉ̲: is the number of Monte Carlo steps, i.e. the number
  //                  of sweeps to be employed in the thermalization phase
  //                  of the system (i.e., the phase in which new quantum
  //                  configurations are sampled but nothing is measured;
  //
  //        • N̲ᵇ̲ˡ̲ᵏ̲ˢ̲: is the number of blocks to be used in the estimation of the
  //                 Monte Carlo quantum averages and uncertainties of the observables
  //                 via the Blocking method;
  //
  //  The single VMC run allows us to move a single step in the variational
  //  parameter optimization procedure.
  /*###############################################################################################*/

  //Initialization and Equilibration
  if(_if_restart_from_config)
    this -> Init_Config(_configuration, _hidden_ket, _hidden_bra);
  else
    this -> Init_Config();
  for(unsigned int eq_step = 0; eq_step < _Neq; eq_step++)
    this -> Make_Sweep();

  //Monte Carlo measurement
  for(unsigned int mcmc_step = 0; mcmc_step < _Nsweeps; mcmc_step++){

    this -> Make_Sweep();  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ (i.e. a new point of the mcmc)
    this -> Measure();  //Measure quantum properties on the new sampled system configuration |𝒮ⁿᵉʷ⟩
    this -> Write_MCMC_Config(mcmc_step);  //Records the sampled |𝒮ⁿᵉʷ⟩

  }

  //Computes the quantum averages
  this -> Estimate();

}


void VMC_Sampler :: Euler() {

  /*#########################################################################*/
  //  Updates the variational parameters (𝜙,𝜶) according to the choosen
  //  𝐓𝐃𝐕𝐌𝐂 equations of motion through the Euler integration method.
  //  The equations for the parameters optimization are:
  //
  //        ==================
  //          𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌
  //        ==================
  //          • 𝐈𝐦𝐚𝐠𝐢𝐧𝐚𝐫𝐲-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝒊𝐓𝐃𝐕𝐌𝐂)
  //              𝕊(τ)•𝜶̇(τ) = - 𝔽(τ)
  //          • 𝐑𝐞𝐚𝐥-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝐓𝐃𝐕𝐌𝐂)
  //              𝕊(𝑡)•𝜶̇(𝑡) =  - 𝑖 • 𝔽(𝑡)
  //
  //        ============
  //          𝒮𝒽𝒶𝒹ℴ𝓌
  //        ============
  //          • 𝐈𝐦𝐚𝐠𝐢𝐧𝐚𝐫𝐲-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝒊𝐓𝐃𝐕𝐌𝐂)
  //              ℚ(τ) • 𝜶̇ᴿ(τ) = 𝔽ᴵ(τ)
  //              ℚ(τ) • 𝜶̇ᴵ(τ) = - 𝔽ᴿ(τ)
  //              𝜙̇ᴿ(τ) = - 𝜶̇ᴿ(τ) • ≪𝕆≫ - 𝜶̇ᴵ(τ) • ⌈𝕆⌋ - ⟨ℋ⟩
  //              𝜙̇ᴵ(τ) = + 𝜶̇ᴿ(τ) • ⌈𝕆⌋ - 𝜶̇ᴵ(τ) • ≪𝕆≫
  //          • 𝐑𝐞𝐚𝐥-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝐓𝐃𝐕𝐌𝐂)
  //              ℚ(𝑡) • 𝜶̇ᴿ(𝑡) = 𝔽ᴿ(𝑡)
  //              ℚ(𝑡) • 𝜶̇ᴵ(𝑡) = 𝔽ᴵ(𝑡)
  //              𝜙̇ᴿ(𝑡) = - 𝜶̇ᴿ(𝑡) • ≪𝕆≫ - 𝜶̇ᴵ(𝑡) • ⌈𝕆⌋
  //              𝜙̇ᴵ(𝑡) = + 𝜶̇ᴿ(𝑡) • ⌈𝕆⌋ - 𝜶̇ᴵ(𝑡) • ≪𝕆≫ - ⟨ℋ⟩
  //
  //  where in the 𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌 case we assume 𝜙 = 0.
  //  In the Euler method we obtain the new parameters in the following way:
  //  𝒾𝒻
  //
  //        𝛼̇(𝑡) = 𝒻{𝛼(𝑡)}
  //
  //  𝓉𝒽ℯ𝓃
  //
  //        𝛼(𝑡+𝑑𝑡) = 𝛼(𝑡) + 𝑑𝑡 • 𝒻{𝛼(𝑡)}
  //
  //  where 𝒻{𝛼(𝑡)} is numerically integrated by using the 𝐬𝐨𝐥𝐯𝐞() method
  //  of the C++ Armadillo library.
  /*#########################################################################*/

  if(!_if_vmc){

      /*################*/
     /*  𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_dot;

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          alpha_dot = solve(_Q + _eps * _I, - _i * _F);
        else
          alpha_dot = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          alpha_dot = solve(_Q + _eps * _I, - _F);
        else
          alpha_dot = solve(_Q, - _F);

      }

      //Updates the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++)
        _vqs.set_alpha_at(prms, _vqs.alpha_at(prms) + _delta * alpha_dot(prms));

    }

      /*############*/
     /*  𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      Col <double> alpha_dot_re;
      Col <double> alpha_dot_im;
      double phi_dot_re;
      double phi_dot_im;

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_real_at(prms, _vqs.alpha_at(prms).real() + _delta * alpha_dot_re(prms));
        _vqs.set_alpha_imag_at(prms, _vqs.alpha_at(prms).imag() + _delta * alpha_dot_im(prms));

      }
      if(_vqs.if_phi_neq_zero()){

        _vqs.set_phi_real(_vqs.phi().real() + _delta * phi_dot_re);
        _vqs.set_phi_imag(_vqs.phi().imag() + _delta * phi_dot_im);

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Heun() {

  /*###############################################################*/
  //  The Heun method is a so-called predictor-corrector method,
  //  which achieves a second order accuracy.
  //  In the Heun method we first obtain the auxiliary updates
  //  of the variational parameters
  //
  //        𝜶̃(𝑡 + 𝛿𝑡) = 𝜶(𝑡) + 𝛿𝑡•𝒻{𝛼(𝑡)}
  //
  //  as in the Euler method. We remember that
  //
  //        𝛼̇(𝑡) = 𝒻{𝛼(𝑡)}.
  //
  //  These updates are used to performed a second optimization
  //  step via the 𝐕𝐌𝐂_𝐒𝐭𝐞𝐩() function, and then obtained a second
  //  order updates as
  //
  //        𝜶(𝑡 + 𝛿𝑡) = 𝜶(𝑡) + 1/2•𝛿𝑡•[𝒻{𝛼(𝑡)} + f{𝜶̃(𝑡 + 𝛿𝑡)}].
  //
  //  The first 𝐕𝐌𝐂 step in this integration is performed in the
  //  main program.
  /*###############################################################*/

  if(!_if_vmc){

      /*################*/
     /*  𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_t = _vqs.alpha();  // 𝜶(𝑡)
      Col <std::complex <double>> alpha_dot_t;  // 𝛼̇(𝑡) = 𝒻{𝛼(𝑡)}
      Col <std::complex <double>> alpha_dot_tilde_t;  // f{𝜶̃(𝑡 + 𝛿𝑡)}

      /**************/
      /* FIRST STEP */
      /**************/
      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          alpha_dot_t = solve(_Q + _eps * _I, - _i * _F);
        else
          alpha_dot_t = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          alpha_dot_t = solve(_Q + _eps * _I, - _F);
        else
          alpha_dot_t = solve(_Q, - _F);

      }

      //Updates the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_at(prms, alpha_t(prms) + _delta * alpha_dot_t(prms));  // 𝜶̃(𝑡 + 𝛿𝑡)

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          alpha_dot_tilde_t = solve(_Q + _eps * _I, - _i * _F);
        else
          alpha_dot_tilde_t = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          alpha_dot_tilde_t = solve(_Q + _eps * _I, - _F);
        else
          alpha_dot_tilde_t = solve(_Q, - _F);

      }

      //Final update of the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_at(prms, alpha_t(prms) + 0.5 * _delta * (alpha_dot_t(prms) + alpha_dot_tilde_t(prms)));  // 𝜶(𝑡 + 𝛿𝑡)

      }

    }

      /*############*/
     /*  𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // 𝜙ᴿ(𝑡)
      double phi_t_im = _vqs.phi().imag();  // 𝜙ᴵ(𝑡)
      Col <double> alpha_t_re = real(_vqs.alpha());  // 𝜶ᴿ(𝑡)
      Col <double> alpha_t_im = imag(_vqs.alpha());  // 𝜶ᴵ(𝑡)
      Col <double> alpha_dot_t_re;  // 𝛼̇ᴿ(𝑡) = 𝒻{𝛼ᴿ(𝑡)}
      Col <double> alpha_dot_t_im;  // 𝛼̇ᴵ(𝑡) = 𝒻{𝛼ᴵ(𝑡)}
      double phi_dot_t_re;  // 𝜙̇ᴿ(𝑡)
      double phi_dot_t_im;  // 𝜙̇ᴵ(𝑡)
      Col <double> alpha_dot_tilde_t_re;
      Col <double> alpha_dot_tilde_t_im;
      double phi_dot_tilde_re;
      double phi_dot_tilde_im;

      /**************/
      /* FIRST STEP */
      /**************/
      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_real_at(prms, alpha_t_re(prms) + _delta * alpha_dot_t_re(prms));
        _vqs.set_alpha_imag_at(prms, alpha_t_im(prms) + _delta * alpha_dot_t_im(prms));

      }
      if(_vqs.if_phi_neq_zero()){

        _vqs.set_phi_real(phi_t_re + _delta * phi_dot_t_re);
        _vqs.set_phi_imag(phi_t_im + _delta * phi_dot_t_im);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_real_at(prms, alpha_t_re(prms) + 0.5 * _delta * (alpha_dot_t_re(prms) + alpha_dot_tilde_t_re(prms)));
        _vqs.set_alpha_imag_at(prms, alpha_t_im(prms) + 0.5 * _delta * (alpha_dot_t_im(prms) + alpha_dot_tilde_t_im(prms)));

      }
      if(_vqs.if_phi_neq_zero()){

        _vqs.set_phi_real(phi_t_re + 0.5 * _delta * (phi_dot_t_re + phi_dot_tilde_re));
        _vqs.set_phi_imag(phi_t_im + 0.5 * _delta * (phi_dot_t_im + phi_dot_tilde_im));

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: RK4() {

  /*############################################################################*/
  //  The fourth order Runge Kutta method (𝐑𝐊𝟒) is a one-step explicit
  //  method that achieves a fourth-order accuracy by evaluating the
  //  function 𝒻{𝛼(𝑡)} four times at each time-step.
  //  It is defined as follows:
  //
  //        𝛼ₖ(𝑡 + 𝛿ₜ) = 𝛼ₖ(𝑡) + 𝟣/𝟨•𝛿ₜ•[κ𝟣 + κ𝟤 + κ𝟥 + κ𝟦]
  //
  //  where we have defined
  //
  //        κ𝟣 = 𝒻{𝛼(𝑡)}
  //        κ𝟤 = 𝒻{𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣}
  //        κ𝟥 = 𝒻{𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤}
  //        κ𝟦 = 𝒻{𝛼(𝑡) + 𝛿ₜ•κ𝟥}.
  //
  //  We remember that
  //
  //        𝛼̇(𝑡) = 𝒻{𝛼(𝑡)}.
  //
  //  The first 𝐕𝐌𝐂 step in this integration is performed in the main program.
  /*############################################################################*/

  if(!_if_vmc){

      /*################*/
     /*  𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_t = _vqs.alpha();  // 𝜶(𝑡)
      Col <std::complex <double>> k1;  // κ𝟣 = 𝒻{𝛼(𝑡)}
      Col <std::complex <double>> k2;  // κ𝟤 = 𝒻{𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣}
      Col <std::complex <double>> k3;  // κ𝟥 = 𝒻{𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤}
      Col <std::complex <double>> k4;  // κ𝟦 = 𝒻{𝛼(𝑡) + 𝛿ₜ•κ𝟥}

      /**************/
      /* FIRST STEP */
      /**************/
      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k1 = solve(_Q + _eps * _I, - _i * _F);
        else
          k1 = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k1 = solve(_Q + _eps * _I, - _F);
        else
          k1 = solve(_Q, - _F);

      }

      //Updates the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_at(prms, alpha_t(prms) + 0.5 * _delta * k1(prms));  // 𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐕𝐌𝐂 step with parameters 𝛼(𝑡) → 𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k2 = solve(_Q + _eps * _I, - _i * _F);
        else
          k2 = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k2 = solve(_Q + _eps * _I, - _F);
        else
          k2 = solve(_Q, - _F);

      }

      //Final update of the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_at(prms, alpha_t(prms) + 0.5 * _delta * k2(prms));  // 𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤

      }

      /**************/
      /* THIRD STEP */
      /**************/
      //Makes a second 𝐕𝐌𝐂 step with parameters 𝛼(𝑡) → 𝛼(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k3 = solve(_Q + _eps * _I, - _i * _F);
        else
          k3 = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k3 = solve(_Q + _eps * _I, - _F);
        else
          k3 = solve(_Q, - _F);

      }

      //Final update of the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_at(prms, alpha_t(prms) + _delta * k3(prms));  // 𝛼(𝑡) + 𝛿ₜ•κ𝟥

      }

      /***************/
      /* FOURTH STEP */
      /***************/
      //Makes a second 𝐕𝐌𝐂 step with parameters 𝛼(𝑡) → 𝛼(𝑡) + 𝛿ₜ•κ𝟥
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k4 = solve(_Q + _eps * _I, - _i * _F);
        else
          k4 = solve(_Q, - _i * _F);

      }
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

        if(_if_QGT_reg)
          k4 = solve(_Q + _eps * _I, - _F);
        else
          k4 = solve(_Q, - _F);

      }

      //Final update of the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_at(prms, alpha_t(prms) + (1.0/6.0) * _delta * (k1(prms) + k2(prms) + k3(prms) + k4(prms)));  // 𝛼ₖ(𝑡 + 𝛿ₜ)

      }

    }

      /*############*/
     /*  𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // 𝜙ᴿ(𝑡)
      double phi_t_im = _vqs.phi().imag();  // 𝜙ᴵ(𝑡)
      Col <double> alpha_t_re = real(_vqs.alpha());  // 𝜶ᴿ(𝑡)
      Col <double> alpha_t_im = imag(_vqs.alpha());  // 𝜶ᴵ(𝑡)
      Col <double> k1_re;  // κ𝟣ᴿ = 𝒻{𝛼ᴿ(𝑡)}
      Col <double> k1_im;  // κ𝟣ᴵ = 𝒻{𝛼ᴵ(𝑡)}
      Col <double> k2_re;  // κ𝟤ᴿ = 𝒻{𝛼ᴿ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣ᴿ}
      Col <double> k2_im;  // κ𝟤ᴵ = 𝒻{𝛼ᴵ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣ᴵ}
      Col <double> k3_re;  // κ𝟥ᴿ = 𝒻{𝛼ᴿ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤ᴿ}
      Col <double> k3_im;  // κ𝟥ᴵ = 𝒻{𝛼ᴵ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤ᴵ}
      Col <double> k4_re;  // κ𝟦ᴿ = 𝒻{𝛼ᴿ(𝑡) + 𝛿ₜ•κ𝟥ᴿ}
      Col <double> k4_im;  // κ𝟦ᴵ = 𝒻{𝛼ᴵ(𝑡) + 𝛿ₜ•κ𝟥ᴵ}
      double phi_k1_re, phi_k2_re, phi_k3_re, phi_k4_re;
      double phi_k1_im, phi_k2_im, phi_k3_im, phi_k4_im;

      /**************/
      /* FIRST STEP */
      /**************/
      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_real_at(prms, alpha_t_re(prms) + 0.5 * _delta * k1_re(prms));
        _vqs.set_alpha_imag_at(prms, alpha_t_im(prms) + 0.5 * _delta * k1_im(prms));

      }
      if(_vqs.if_phi_neq_zero()){

        _vqs.set_phi_real(phi_t_re + 0.5 * _delta * phi_k1_re);
        _vqs.set_phi_imag(phi_t_im + 0.5 * _delta * phi_k1_im);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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

      //Final update of the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_real_at(prms, alpha_t_re(prms) + 0.5 * _delta * k2_re(prms));
        _vqs.set_alpha_imag_at(prms, alpha_t_im(prms) + 0.5 * _delta * k2_im(prms));

      }
      if(_vqs.if_phi_neq_zero()){

        _vqs.set_phi_real(phi_t_re + 0.5 * _delta * phi_k2_re);
        _vqs.set_phi_imag(phi_t_im + 0.5 * _delta * phi_k2_re);

      }

      /**************/
      /* THIRD STEP */
      /**************/
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step();

      //Solves the appropriate equations of motion
      if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
      else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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

      //Final update of the variational parameters
      for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

        _vqs.set_alpha_real_at(prms, alpha_t_re(prms) + _delta * k3_re(prms));
        _vqs.set_alpha_imag_at(prms, alpha_t_im(prms) + _delta * k3_im(prms));

      }
      if(_vqs.if_phi_neq_zero()){

        _vqs.set_phi_real(phi_t_re + _delta * phi_k3_re);
        _vqs.set_phi_imag(phi_t_im + _delta * phi_k3_re);

      }

    /***************/
    /* FOURTH STEP */
    /***************/
    //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
    this -> Reset_Moves_Statistics();
    this -> Reset();
    this -> VMC_Step();

    //Solves the appropriate equations of motion
    if(_if_real_time){  // 𝐓𝐃𝐕𝐌𝐂

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
    else{  // 𝒊𝐓𝐃𝐕𝐌𝐂

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
    for(unsigned int prms = 0; prms < _vqs.n_alpha(); prms++){

      _vqs.set_alpha_real_at(prms, alpha_t_re(prms) + (1.0/6.0) * _delta * (k1_re(prms) + k2_re(prms) + k3_re(prms) + k4_re(prms)));
      _vqs.set_alpha_imag_at(prms, alpha_t_im(prms) + (1.0/6.0) * _delta * (k1_im(prms) + k2_im(prms) + k3_im(prms) + k4_im(prms)));

    }
    if(_vqs.if_phi_neq_zero()){

      _vqs.set_phi_real(phi_t_re + (1.0/6.0) * _delta * (phi_k1_re + phi_k2_re + phi_k3_re + phi_k4_re));
      _vqs.set_phi_imag(phi_t_im + (1.0/6.0) * _delta * (phi_k1_im + phi_k2_im + phi_k3_im + phi_k4_im));

    }

  }

  }
  else
    return;



}


#endif
