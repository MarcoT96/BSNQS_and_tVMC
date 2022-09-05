#ifndef __SAMPLER__
#define __SAMPLER__


/*********************************************************************************************************/
/********************************  Variational Monte Carlo Sampler  **************************************/
/*********************************************************************************************************/
/*

  We create a Variational Quantum Monte Carlo (𝐕𝐌𝐂) sampler as a C++ class, which is able to
  optimize a generic 𝒮𝒽𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 (a variational quantum state 𝐯𝐪𝐬) in order to study a
  generic Lattice Quantum System (𝐋𝐐𝐒).
  The main goal of the sampler is to optimize the parameters that uniquely characterize the 𝐯𝐪𝐬
  to obtain the ground state of the given Hamiltonian; once found the ground state, it is
  possible to study the real-time dynamics of the system after performing a quantum quench on a
  certain coupling constant.

  The optimization described above takes place within a stochastic setting, in which the
  procedure leads to the resolution of the following equations of motion for the variational
  parameters 𝛂 (𝐭𝐕𝐌𝐂 Equations of Motion):

            Σₖ α̇ₖ {αⱼ, αₖ} = ∂𝙀[𝛂] / ∂αⱼ      (𝐭𝐕𝐌𝐂)
            Σₖ α̇ₖ {αⱼ, αₖ} = - 𝑖 • ∂𝙀[𝛂] / ∂αⱼ   (𝑖-𝐭𝐕𝐌𝐂)

  where the ground state properties are recovered with an imaginaty time evolution

            𝒕 → 𝝉 = 𝑖𝒕.

  This class is also able to apply the above technique to a 𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏, where
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
    WaveFunction& _vqs;  //The variational wave function |Ψ(𝜙,𝛂)⟩
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
    Row <double> _instSquareMag;  //Measured values of the square magnetization on the configuration |𝒗⟩ along the MCMC
    Mat <double> _instSpinSpinCorr;  //Measured values of spin-spin correlation on the configuration |𝒗⟩ along the MCMC
    Mat <std::complex <double>> _instO_ket;  //Measured local operators 𝓞(𝒗,𝒉) along the MCMC
    Mat <std::complex <double>> _instO_bra;  //Measured local operators 𝓞(𝒗,𝒉ˈ) along the MCMC

    //Simulation options variables
    bool _if_shadow;  //Chooses the shadow or the non-shadow algorithm
    bool _if_hidden_off;  //Chooses to shut down the auxiliary variable in a Shadow vqs
    bool _if_vmc;  //Chooses to make a single simple 𝐕𝐌𝐂 without parameters optimization
    bool _if_imag_time;  //Chooses imaginary-time dinamics, i.e. 𝐭𝐕𝐌𝐂 with 𝛕 = 𝑖𝐭
    bool _if_real_time;  //Chooses real-time dynamics
    bool _if_QGT_reg;  //Chooses to regularize the Quantum Geometric Tensor by adding a bias
    bool _if_extra_hidden_sum;  //Increases the sampling of |𝒉⟩ and ⟨𝒉ˈ| during the single measure
    bool _if_restart_from_config;  //Chooses to initialize the initial point of the MCMC from a previously optimized visible configuration |𝒗⟩

    //Simulation parameters of the single 𝐕𝐌𝐂 step
    unsigned int _reg_method;  //Chooses how to regularize ℚ
    unsigned int _Nsweeps;  //Number of Monte Carlo sweeps (i.e. #MC-steps of the single 𝐭𝐕𝐌𝐂 step)
    unsigned int _Nblks;  //Number of blocks to estimate uncertainties
    unsigned int _Neq;  //Number of Monte Carlo equilibration steps to do at the beginning of the single 𝐕𝐌𝐂 step
    unsigned int _M;  //Number of spin-flips moves to perform in the single sweep
    unsigned int _Nflips;  //Number of spin-flips in each spin-flips move
    unsigned int _Nextra;  //Number of extra MC-steps involving only the hidden sampling
    unsigned int _Nblks_extra;  //Number of blocks in the extra hidden sampling
    double _p_equal_site;  //Probability for the equal site Monte Carlo move
    double _p_visible_nn;  //Probability for the visible nearest neighbor Monte Carlo move
    double _p_hidden_nn;  //Probability for the hidden nearest neighbor Monte Carlo move

    //𝐭𝐕𝐌𝐂 variables
    double _delta;  //The value of the integration step 𝛿𝑡
    double _eps;  //The value of the Quantum Geometric Tensor bias ε
    Col <double> _cosII;  //The block averages of the non-zero reweighting ratio part ⟨cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]⟩ⱼᵇˡᵏ
    Col <double> _sinII;  //The block averages of the (theoretically)-zero reweighting ratio part ⟨sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]⟩ⱼᵇˡᵏ
    Col <double> _global_cosII;
    Col <double> _global_sinII;
    field <Col <std::complex <double>>> _Observables;  //The block averages of the quantum observables computed along the MCMC ⟨𝒪⟩ⱼᵇˡᵏ
    Col <double> _squareMag;  //The block averages of the square magnetization computed along the MCMC ⟨(𝗠 ᶻ)^2⟩ⱼᵇˡᵏ
    Mat <double> _SpinSpinCorr;  //The block averages of the spin-spin correlation along the z-axis as a function of distance computed along the MCMC ⟨𝗖ⱼₖ(𝙧)⟩ⱼᵇˡᵏ
    field <Col <std::complex <double>>> _O;  //The block averages of the local operators computed along the MCMC ⟨𝓞ₖ⟩ⱼᵇˡᵏ, for k = 𝟣,…,nᵃˡᵖʰᵃ
    field <Col <std::complex <double>>> _global_Observables; //Collects the statistics for _Observables among all the nodes
    Col <double> _globalMz2;  //Collects the statistics for _squareMag among all the nodes
    Mat <double> _globalCofr;  //Collects the statistics for _SpinSpinCorr among all the nodes
    Col <std::complex <double>> _mean_O;  // ⟨⟨𝓞ₖ⟩ᵇˡᵏ⟩
    Col <std::complex <double>> _mean_O_star;  // ⟨⟨𝓞⋆ₖ⟩ᵇˡᵏ⟩
    Col <double> _mean_O_angled;  // ⟨≪𝓞≫ᵇˡᵏ⟩
    Col <double> _mean_O_square;  // ⟨⌈𝓞⌋ᵇˡᵏ⟩
    std::complex <double> _E;  // The standard stochastic average of ⟨Ĥ⟩ (without block averaging)
    Mat <std::complex <double>> _Q;  //The Quantum Geometric Tensor ℚ
    Col <std::complex <double>> _F;  //The energy Gradient 𝔽 acting on 𝛂

    //Print options and related files
    bool _write_Move_Statistics;  //Writes the acceptance statistics along the single MCMC
    bool _write_MCMC_Config;  //Writes the sampled |𝒮⟩ along the single MCMC
    bool _write_final_Config;  //Writes the last sampled |𝒮⟩ of each 𝐕𝐌𝐂 step
    bool _write_opt_Observables;  //Writes optimized Monte Carlo estimates of quantum observables at the end of each 𝐕𝐌𝐂 step
    bool _write_block_Observables;  //Writes the observables averages in each block of the MCMC, for each 𝐕𝐌𝐂 step
    bool _write_opt_Params;  //Writes the optimized set 𝓥ᵒᵖᵗ of the variational wave function at the end of the 𝐭𝐕𝐌𝐂
    bool _write_all_Params;  //Writes the set of optimized 𝓥 of the variational wave function after each 𝐕𝐌𝐂 step
    bool _write_QGT_matrix;  //Writes the Quantum Geometric Tensor matrix of each 𝐕𝐌𝐂 step
    bool _write_QGT_cond;  //Writes the condition number of the Quantum Geometric Tensor matrix of each 𝐕𝐌𝐂 step
    bool _write_QGT_eigen;  //Writes the Quantum Geometric Tensor eigenvalues of each 𝐕𝐌𝐂 step
    std::ofstream _file_Move_Statistics;
    std::ofstream _file_MCMC_Config;
    std::ofstream _file_final_Config;
    std::ofstream _file_opt_Energy;
    std::ofstream _file_opt_SigmaX;
    std::ofstream _file_opt_SzSz;
    std::ofstream _file_opt_SzSzCorr;
    std::ofstream _file_block_Energy;
    std::ofstream _file_block_SigmaX;
    std::ofstream _file_block_SzSz;
    std::ofstream _file_block_SzSzCorr;
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
    WaveFunction& vqs() const {return _vqs;}  //Returns the reference to the 𝒜𝓃𝓈𝒶𝓉𝓏
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
    Mat <std::complex <double>> InstObs_ket() const {return _instObs_ket;}  //Returns all the measured values of 𝒪ˡᵒᶜ(𝒗,𝒉) after a single 𝐕𝐌𝐂 run
    Mat <std::complex <double>> InstObs_bra() const {return _instObs_bra;}  //Returns all the measured values of 𝒪ˡᵒᶜ(𝒗,𝒉') after a single 𝐕𝐌𝐂 run
    Mat <std::complex <double>> InstO_ket() const {return _instO_ket;}  //Returns all the measured local operators 𝓞(𝒗,𝒉) after a single 𝐕𝐌𝐂 run
    Mat <std::complex <double>> InstO_bra() const {return _instO_bra;}  //Returns all the measured local operators 𝓞(𝒗,𝒉') after a single 𝐕𝐌𝐂 run
    Mat <double> InstNorm() const {return _instReweight;}  //Returns all the measured values of 𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] and 𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] after a single 𝐕𝐌𝐂 run
    double delta() const {return _delta;}  //Returns the integration step parameter 𝛿𝑡 used in the dynamics solver
    double QGT_bias() const {return _eps;}  //Returns the regularization bias of the Quantum Geometric Tensor
    Col <double> cos() const {return _global_cosII;}
    Col <double> sin() const {return _global_sinII;}
    field <Col <std::complex <double>>> Observables() const {return _global_Observables;}
    Mat <std::complex <double>> QGT() const {return _Q;}  //Returns the Monte Carlo estimate of the QGT
    Col <std::complex <double>> F() const {return _F;}  //Returns the Monte Carlo estimate of the energy gradient
    Col <std::complex <double>> O() const {return _mean_O;}
    Col <std::complex <double>> O_star() const {return _mean_O_star;}
    Col <double> _O_angled() const {return _mean_O_angled;}  //Returns the Monte Carlo estimate of the vector of ≪𝓞ₖ≫
    Col <double> _O_square() const {return _mean_O_square;}  //Returns the Monte Carlo estimate of the vector of ⌈𝓞ₖ⌋
    std::complex <double> E() const {return _E;}  //Returns the Monte Carlo estimate of the energy ⟨Ĥ⟩

    //Initialization functions
    void Init_Config(const Mat <int>& initial_visible=Mat <int>(),  //Initializes the quantum configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
                     const Mat <int>& initial_ket=Mat <int>(),
                     const Mat <int>& initial_bra=Mat <int>(),
                     bool zeroMag=true);
    void ShutDownHidden() {_if_hidden_off = true;}  //Shuts down the hidden variables
    void setImagTimeDyn(double delta=0.01);  //Chooses the imaginary-time 𝐭𝐕𝐌𝐂 algorithm
    void setRealTimeDyn(double delta=0.01);  //Chooses the real-time 𝐭𝐕𝐌𝐂 algorithm
    void choose_reg_method(unsigned int method_flag) {_reg_method = method_flag;}
    void setQGTReg(double eps=0.000001);  //Chooses to regularize the QGT
    void setExtraHiddenSum(unsigned int, unsigned int);  //Chooses to make the MC observables less noisy
    void setRestartFromConfig() {_if_restart_from_config = true;}  //Chooses the restart option at the beginning of the MCMC
    void setStepParameters(unsigned int, unsigned int, unsigned int,           //Sets the Monte Carlo parameters for the single 𝐕𝐌𝐂 step
                           unsigned int, unsigned int, double, double, double,
                           int);

    //Print options functions
    void setFile_Move_Statistics(std::string, int);
    void setFile_MCMC_Config(std::string, int);
    void setFile_final_Config(std::string, MPI_Comm);
    void setFile_opt_Energy(std::string, int);
    void setFile_opt_Obs(std::string, int);
    void setFile_block_Energy(std::string, int);
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
    void Measure();  //Measurement of the istantaneous observables along a single 𝐕𝐌𝐂 run
    void Estimate(MPI_Comm);  //Monte Carlo estimates of the quantum observable averages
    void Write_Quantum_properties(unsigned int, int);  //Write on appropriate files all the required system quantum properties

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
    void VMC_Step(MPI_Comm);  //Performs a single 𝐕𝐌𝐂 step

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

    std::cout << "#Define the 𝐕𝐌𝐂 sampler of the variational quantum state |Ψ(𝜙, 𝛂)⟩." << std::endl;
    std::cout << " The sampler is defined on a " << _vqs.type_of_ansatz() << " architecture designed for Lattice Quantum Systems." << std::endl;

  }

  /*#######################################################*/
  //  Creates and initializes the Random Number Generator
  //  Each process involved in the parallelization of
  //  the executable code reads a different pair of
  //  numbers from the Primes file, according to its rank.
  /*#######################################################*/
  if(rank == 0) std::cout << " Create and initialize the random number generator." << std::endl;
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes");
  if(Primes.is_open())
    for(unsigned int p = 0; p <= rank; p++) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed2.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Sets the simulation option variables
  if(_vqs.type_of_ansatz() == "Shadow") _if_shadow = true;
  else _if_shadow = false;
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

  if(rank == 0) std::cout << " 𝐕𝐌𝐂 sampler correctly initialized." << std::endl;

}


void VMC_Sampler :: print_configuration() const {  //Useful for debugging

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
  //  or to initialize the configuration by providing an acceptable 𝐢𝐧𝐢𝐭𝐢𝐚𝐥_* for the variables.
  //  If the boolean data-member 𝐢𝐟_𝒉𝒊𝒅𝒅𝒆𝒏_𝐨𝐟𝐟 is true, the hidden variables are all initialized
  //  and fixed to zero, i.e. they are turned off in order to make the 𝒮𝒽𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 a simple
  //  𝒜𝓃𝓈𝒶𝓉𝓏 deprived of the auxiliary variables.
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

    if(!_if_restart_from_config){  //Restarts from a random configuration |𝒮⟩

      _configuration.set_size(1, _Nspin);
      if(_if_shadow){

        _hidden_ket.set_size(1, _Nhidden);
        _hidden_bra.set_size(1, _Nhidden);

      }

    }
    else{  //Restarts from a previously sampled configuration |𝒮⟩

      _configuration = initial_visible;
      if(_if_shadow){

        if(initial_ket.is_empty()) _hidden_ket.set_size(1, _Nhidden);
        else _hidden_ket = initial_ket;
        if(initial_bra.is_empty()) _hidden_bra.set_size(1, _Nhidden);
        else _hidden_bra = initial_bra;

      }

    }

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
        _configuration.at(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

    }
    //Performs a check on the magnetization
    if(zeroMag){  //Default case

      if(!_Nspin % 2){

        std::cerr << " ##SizeError: Cannot initialize a random spin configuration with zero magnetization for an odd number of spins." << std::endl;
        std::cerr << "   Failed to initialize the starting point of the Markov Chain." << std::endl;
        std::abort();

      }
      int tempMag = 1;
      while(tempMag != 0){

        tempMag = 0;
        for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
            tempMag += _configuration.at(j_row, j_col);

        }
        if(tempMag > 0){

          int rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows - 1);  //Select a random spin-UP
          int rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols - 1);
          while(_configuration.at(rs_row, rs_col) < 0){

            rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows - 1);
            rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols - 1);

          }
          _configuration.at(rs_row, rs_col) = -1;  //Flip that spin-UP in order to decrease the positive magnetization
          tempMag -= 1;

        }
        else if(tempMag < 0){

          int rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows - 1);  //Select a random spin-DOWN
          int rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols - 1);
          while(_configuration.at(rs_row, rs_col) > 0){

            rs_row = _rnd.Rannyu_INT(0, _configuration.n_rows - 1);
            rs_col = _rnd.Rannyu_INT(0, _configuration.n_cols - 1);

          }
          _configuration.at(rs_row, rs_col) = 1;  //Flip that spin-DOWN in order to increase the negative magnetization
          tempMag += 1;

        }

      }

    }

  }

  //Initializes |𝐡⟩ and ⟨𝐡ˈ| randomly
  if(_if_shadow){

    if(_if_hidden_off){

      _hidden_ket.fill(0);
      _hidden_bra.fill(0);

    }  //Shuts down the auxiliary variables
    else{

      if(initial_ket.is_empty()){

        //Randomly chooses spin up or spin down
        for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
            _hidden_ket.at(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

        }

      }
      if(initial_bra.is_empty()){

        //Randomly chooses spin up or spin down
        for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
            _hidden_bra.at(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

        }

      }

    }

  }

  //Initializes the variational quantum state
  _vqs.Init_on_Config(_configuration);

}


void VMC_Sampler :: setImagTimeDyn(double delta){

  /*#############################################################*/
  //  Allows to update the variational parameters by integration
  //  (with an ODE integrator) of the equation of motion in
  //  imaginary time
  //
  //        𝒕 → 𝝉 = 𝑖𝒕
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
    std::cout << " \tNumber of spin sweeps in the single 𝐕𝐌𝐂 step:  " << _Nsweeps << std::endl;
    std::cout << " \tNumber of blocks in the single 𝐕𝐌𝐂 step:  " << _Nblks << std::endl;
    std::cout << " \tEquilibration steps in the single 𝐕𝐌𝐂 step:  " << _Neq << std::endl;
    std::cout << " \tNumber of spin-flips moves in the single 𝐌𝐂 sweep:  " << _M << std::endl;
    std::cout << " \tNumber of spin-flip in the single spin-flips move:  " << _Nflips << std::endl;
    std::cout << " \tProbability for the equal-site 𝐌𝐂-move:  " << _p_equal_site * 100.0 << " %" << std::endl;
    std::cout << " \tProbability for the nearest-neighbors visible 𝐌𝐂-move:  " << _p_visible_nn * 100.0 << " %" << std::endl;
    std::cout << " \tProbability for the nearest-neighbors hidden 𝐌𝐂-move:  " << _p_hidden_nn * 100.0 << " %" << std::endl;
    if(_if_extra_hidden_sum){

      std::cout << " \tNumber of extra hidden sampling performed within each instantaneous measurement:  "  << _Nextra << std::endl;
      std::cout << " \tNumber of block for the extra hidden sampling statistics:  " << _Nblks_extra << std::endl;

    }
    std::cout << " \tIntegration step parameter:  " << _delta << std::endl;
    if(_if_QGT_reg){

      if(_reg_method == 0) std::cout << " \tDiagonal QGT regularization with ε = " << _eps << std::endl << std::endl;
      else if(_reg_method == 1) std::cout << " \tMoore-Penrose pseudo-inverse QGT regularization" << std::endl << std::endl;
      else{

        std::cerr << " ##ValueError: choosen regularization method not available!" << std::endl;
        std::cerr << "   Failed to set the parameters for the single simulation step." << std::endl;
        std::abort();

      }

    }

  }

}


void VMC_Sampler :: setFile_Move_Statistics(std::string info, int rank) {

  _write_Move_Statistics = true;
  if(rank == 0){

    _file_Move_Statistics.open("Move_Statistics_" + info + ".dat");
    if(!_file_Move_Statistics.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ Move_Statistics_" << info << ".dat ›› for writing the acceptance statistics at the end of the single 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the acceptance statistics of the moves at the end of the single 𝐭𝐕𝐌𝐂 step on file ‹‹ Move_Statistics_" << info << ".dat ››." << std::endl;

    _file_Move_Statistics << "###########################################################################################################\n";
    _file_Move_Statistics << "# Column Legend\n";
    _file_Move_Statistics << "#\n";
    _file_Move_Statistics << "#   1st: the 𝐭𝐕𝐌𝐂 step identifier\n";
    _file_Move_Statistics << "#   2nd: the sampling acceptance probability (%) of |𝒗⟩\n";
    _file_Move_Statistics << "#   3rd: the sampling acceptance probability (%) of |𝒉⟩\n";
    _file_Move_Statistics << "#   4th: the sampling acceptance probability (%) of ⟨𝒉ˈ|\n";
    _file_Move_Statistics << "#   5th: the sampling acceptance probability (%) of |𝒗 𝒉 𝒉ˈ⟩ moved on equal sites\n";
    _file_Move_Statistics << "#   6th: the sampling acceptance probability (%) of |𝒗⟩ moved on nearest-neighbor sites\n";
    _file_Move_Statistics << "#   7th: the sampling acceptance probability (%) of |𝒉⟩ and ⟨𝒉ˈ| moved on nearest-neighbor sites\n";
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

      std::cerr << " ##FileError: Cannot open the file ‹‹ CONFIG/MCMC_config_" << info << ".dat ›› for writing the sampled configurations along a single MCMC." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the sampled configurations along a single MCMC on file ‹‹ CONFIG/MCMC_config_" << info << ".dat ››." << std::endl;
    _file_MCMC_Config << "####################################################\n";
    _file_MCMC_Config << "# Column Legend\n";
    _file_MCMC_Config << "#\n";
    _file_MCMC_Config << "#   1st: the 𝐌𝐂-step identifier\n";
    _file_MCMC_Config << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
    _file_MCMC_Config << "####################################################\n";

  }

}


void VMC_Sampler :: setFile_final_Config(std::string info, MPI_Comm common) {

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  _write_final_Config = true;

  MPI_Barrier(common);

  //Creates the output directory by checking if CONFIG folder already exists
  if(rank == 0)
    if(!is_directory("./CONFIG") || !exists("./CONFIG")) create_directory("./CONFIG");

  _file_final_Config.open("./CONFIG/final_config_" + info + "_node_" + std::to_string(rank) + ".dat");
  if(!_file_final_Config.good()){

    std::cerr << " ##FileError: Cannot open the file ‹‹ CONFIG/final_config_" << info << "_node_" << rank << ".dat ›› for writing the final configurations at the end of each 𝐭𝐕𝐌𝐂 step." << std::endl;
    std::abort();

  }
  else
    if(rank == 0) std::cout << " Saving the final configurations sampled at the end of each 𝐭𝐕𝐌𝐂 step on file ‹‹ CONFIG/final_config_" << info << "_node_*.dat ››." << std::endl;

  _file_final_Config << "########################################################\n";
  _file_final_Config << "# Column Legend\n";
  _file_final_Config << "#\n";
  _file_final_Config << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
  _file_final_Config << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
  _file_final_Config << "########################################################\n";

}


void VMC_Sampler :: setFile_opt_Energy(std::string info, int rank){

  if(rank == 0){

    _file_opt_Energy.open("opt_energy_" + info + ".dat");

    if(!_file_opt_Energy.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ opt_energy_" << info << ".dat ›› for writing E(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving E(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ opt_energy_" << info << ".dat ››." << std::endl;

    _file_opt_Energy << "##############################################################################\n";
    _file_opt_Energy << "# Column Legend\n";
    _file_opt_Energy << "#\n";
    _file_opt_Energy << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_opt_Energy << "#   2nd:  estimate of ⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆\n";
    _file_opt_Energy << "#   3rd:  error on ⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆\n";
    _file_opt_Energy << "#   4th:  estimate of ⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆\n";
    _file_opt_Energy << "#   5rd:  error on ⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆\n";
    _file_opt_Energy << "#   6th:  estimate of 𝑬ᴿ(𝜙,𝛂)\n";
    _file_opt_Energy << "#   7th:  error on 𝑬ᴿ(𝜙,𝛂)\n";
    _file_opt_Energy << "#   8th:  estimate of 𝑬ᴵ(𝜙,𝛂)\n";
    _file_opt_Energy << "#   9th:  error on 𝑬ᴵ(𝜙,𝛂)\n";
    _file_opt_Energy << "#   10th: standard 𝐌𝐂 average (without block averaging) of 𝑬ᴿ(𝜙,𝛂)\n";
    _file_opt_Energy << "#   11th: standard 𝐌𝐂 average (without block averaging) of 𝑬ᴵ(𝜙,𝛂)\n";
    _file_opt_Energy << "##############################################################################\n";

  }

}


void VMC_Sampler :: setFile_opt_Obs(std::string info, int rank) {

  _write_opt_Observables = true;
  if(rank == 0){

    _file_opt_SigmaX.open("opt_sigmaX_" + info + ".dat");
    _file_opt_SzSz.open("opt_square_mag_" + info + ".dat");
    _file_opt_SzSzCorr.open("opt_Cofr_" + info + ".dat");

    if(!_file_opt_SigmaX.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ opt_sigmaX_" << info << ".dat ›› for writing σˣ(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving σˣ(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ opt_sigmaX_" << info << ".dat ››." << std::endl;
    if(!_file_opt_SzSz.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ opt_square_mag_" << info << ".dat ›› for writing (𝗠 ᶻ)^2(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving (𝗠 ᶻ)^2(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ opt_square_mag_" << info << ".dat ››." << std::endl;
    if(!_file_opt_SzSzCorr.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ opt_Cofr_" << info << ".dat ›› for writing the 𝗖ᶻ(𝙧) correlation function after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving 𝗖ᶻ(𝙧) correlation function after each 𝐭𝐕𝐌𝐂 step on file ‹‹ opt_Cofr_" << info << ".dat ››." << std::endl;

    _file_opt_SigmaX << "################################################\n";
    _file_opt_SigmaX << "# Column Legend\n";
    _file_opt_SigmaX << "#\n";
    _file_opt_SigmaX << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_opt_SigmaX << "#   2nd:  estimate of ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
    _file_opt_SigmaX << "#   3rd:  error on ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
    _file_opt_SigmaX << "#   4th:  estimate of ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
    _file_opt_SigmaX << "#   5th:  error on ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
    _file_opt_SigmaX << "################################################\n";

    _file_opt_SzSz << "##############################################\n";
    _file_opt_SzSz << "# Column Legend\n";
    _file_opt_SzSz << "#\n";
    _file_opt_SzSz << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_opt_SzSz << "#   2nd:  estimate of (𝗠 ᶻ)^2(𝜙,𝛂)\n";
    _file_opt_SzSz << "#   3rd:  error on (𝗠 ᶻ)^2(𝜙,𝛂)\n";
    _file_opt_SzSz << "##############################################\n";

    _file_opt_SzSzCorr << "##############################################\n";
    _file_opt_SzSzCorr << "# Column Legend\n";
    _file_opt_SzSzCorr << "#\n";
    _file_opt_SzSzCorr << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_opt_SzSzCorr << "#   2nd:  spin distance 𝙧 = |𝙭 - 𝙮|\n";
    _file_opt_SzSzCorr << "#   3rd:  estimate of 𝗖ᶻ(𝙧)\n";
    _file_opt_SzSzCorr << "#   4th:  error on 𝗖ᶻ(𝙧)\n";
    _file_opt_SzSzCorr << "##############################################\n";

  }

}


void VMC_Sampler :: setFile_block_Energy(std::string info, int rank){

  if(rank == 0){

    _file_block_Energy.open("block_energy_" + info + ".dat");

    if(!_file_block_Energy.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_energy_" << info << ".dat ›› for writing all the block averages of E(𝜙,𝛂) during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the block averages of E(𝜙,𝛂) during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_energy_" << info << ".dat ››." << std::endl;

    if(!_if_shadow){

      _file_block_Energy << "######################################################\n";
      _file_block_Energy << "# Column Legend\n";
      _file_block_Energy << "#\n";
      _file_block_Energy << "#   1st:   the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_block_Energy << "#   2nd:   the 𝐌𝐂-block identifier\n";
      _file_block_Energy << "#   3rd:   ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   4th:   error on ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   5th:   ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   6th:   error on ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   7th:   ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   8th:   error on ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   9th:   ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   10th:  error on ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "######################################################\n";

    }
    else{

      _file_block_Energy << "######################################################\n";
      _file_block_Energy << "# Column Legend\n";
      _file_block_Energy << "#\n";
      _file_block_Energy << "#   1st:   the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_block_Energy << "#   2nd:   the 𝐌𝐂-block identifier\n";
      _file_block_Energy << "#   3rd:   ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   4th:   error on ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   5th:   ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   6th:   error on ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_block_Energy << "#   7th:   shadow (without reweighting) ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   8th:   shadow (without reweighting) ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   9th:   ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   10th:  error on ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   11th:  ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "#   12th:  error on ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_block_Energy << "######################################################\n";

    }

  }

}


void VMC_Sampler :: setFile_block_Obs(std::string info, int rank) {

  _write_block_Observables = true;
  if(rank == 0){

    _file_block_SigmaX.open("block_sigmaX_" + info + ".dat");
    _file_block_SzSz.open("block_square_mag_" + info + ".dat");
    _file_block_SzSzCorr.open("block_Cofr_" + info + ".dat");

    if(!_file_block_SigmaX.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_sigmaX_" << info << ".dat ›› for writing all the block averages of σˣ during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the block averages of σˣ during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_sigmaX_" << info << ".dat ››." << std::endl;
    if(!_file_block_SzSz.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_square_mag_" << info << ".dat ›› for writing all the block averages of (𝗠 ᶻ)^2 during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the block averages of (𝗠 ᶻ)^2 during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_square_mag_" << info << ".dat ››." << std::endl;
    if(!_file_block_SzSzCorr.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_Cofr_" << info << ".dat ›› for writing all the block averages of the 𝗖ᶻ(𝙧) correlation function during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the block averages of the 𝗖ᶻ(𝙧) correlation function during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_Cofr_" << info << ".dat ››." << std::endl;

    if(!_if_shadow){

      _file_block_SigmaX << "#####################################################################\n";
      _file_block_SigmaX << "# Column Legend\n";
      _file_block_SigmaX << "#\n";
      _file_block_SigmaX << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_block_SigmaX << "#   2nd:  the 𝐌𝐂-block identifier\n";
      _file_block_SigmaX << "#   3rd:  ℜ𝓮{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_block_SigmaX << "#   4th:  progressive error ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
      _file_block_SigmaX << "#   5th:  ℑ𝓶{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_block_SigmaX << "#   6th:  progressive error on ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
      _file_block_SigmaX << "#####################################################################\n";

    }
    else{

      _file_block_SigmaX << "#####################################################################\n";
      _file_block_SigmaX << "# Column Legend\n";
      _file_block_SigmaX << "#\n";
      _file_block_SigmaX << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_block_SigmaX << "#   2nd:  the 𝐌𝐂-block identifier\n";
      _file_block_SigmaX << "#   3rd:  shadow (without reweighting) ℜ𝓮{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_block_SigmaX << "#   4th:  shadow (without reweighting) ℑ𝓶{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_block_SigmaX << "#   5th:  ℜ𝓮{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_block_SigmaX << "#   6th:  progressive error ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
      _file_block_SigmaX << "#   7th:  ℑ𝓶{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_block_SigmaX << "#   8th:  progressive error on ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
      _file_block_SigmaX << "#####################################################################\n";

    }

    _file_block_SzSz << "#################################################\n";
    _file_block_SzSz << "# Column Legend\n";
    _file_block_SzSz << "#\n";
    _file_block_SzSz << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_block_SzSz << "#   2nd:  the 𝐌𝐂-block identifier\n";
    _file_block_SzSz << "#   3rd:  (𝗠 ᶻ)^2ʲ𝓆 in block j\n";
    _file_block_SzSz << "#   4th:  progressive error (𝗠 ᶻ)^2(𝜙,𝛂)\n";
    _file_block_SzSz << "#################################################\n";

    _file_block_SzSzCorr << "##############################################################\n";
    _file_block_SzSzCorr << "# Column Legend\n";
    _file_block_SzSzCorr << "#\n";
    _file_block_SzSzCorr << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_block_SzSzCorr << "#   2nd:  the 𝐌𝐂-block identifier\n";
    _file_block_SzSzCorr << "#   3rd:  spin distance 𝙧 = |𝙭 - 𝙮|\n";
    _file_block_SzSzCorr << "#   4th:  ⟨𝗖ᶻ(𝙧)⟩ʲ𝓆 in block j at distance 𝙧\n";
    _file_block_SzSzCorr << "#   5th:  progressive error on 𝗖ᶻ(𝙧)(𝜙,𝛂) at distance 𝙧\n";
    _file_block_SzSzCorr << "##############################################################\n";

  }

}


void VMC_Sampler :: setFile_opt_Params(std::string info, int rank) {

  _write_opt_Params = true;
  if(rank == 0){

    _file_opt_Params.open("optimized_parameters_" + info + ".wf");
    if(!_file_opt_Params.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ optimized_" << info << ".wf ›› for writing the optimized set of variational parameters 𝓥." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the optimized set of variational parameters 𝓥 on file ‹‹ optimized_" << info << ".wf ››." << std::endl;

  }

}


void VMC_Sampler :: setFile_all_Params(std::string info, int rank) {

  _write_all_Params = true;
  if(rank == 0){

    _file_all_Params.open("variational_manifold_" + info + ".wf");
    if(!_file_all_Params.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ variational_manifold_" << info << ".wf ›› for writing the set of variational parameters 𝓥 at the end of each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the set of variational parameters 𝓥 at the end of each 𝐭𝐕𝐌𝐂 step on file ‹‹ variational_manifold_" << info << ".wf ››." << std::endl;

    _file_all_Params << "########################################\n";
    _file_all_Params << "# Column Legend\n";
    _file_all_Params << "#\n";
    _file_all_Params << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_all_Params << "#   2nd: 𝒱ᴿ\n";
    _file_all_Params << "#   3rd: 𝒱ᴵ\n";
    _file_all_Params << "########################################\n";

    _file_all_Params << 0;
    _file_all_Params << std::setprecision(8) << std::fixed;
    _file_all_Params << "\t" << _vqs.phi().real() << "\t" << _vqs.phi().imag() << std::endl;
    for(unsigned int p = 0; p < _vqs.n_alpha(); p++)
      _file_all_Params << 0 << "\t" << _vqs.alpha_at(p).real() << "\t" << _vqs.alpha_at(p).imag() << std::endl;

  }

}


void VMC_Sampler :: setFile_QGT_matrix(std::string info, int rank) {

  _write_QGT_matrix = true;
  if(rank == 0){

    _file_QGT_matrix.open("qgt_matrix_" + info + ".dat");
    if(!_file_QGT_matrix.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_matrix_" << info << ".dat ›› for writing the Quantum Geometric Tensor." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the QGT after each 𝐭𝐕𝐌𝐂 step on file ‹‹ qgt_matrix_" << info << ".dat ››." << std::endl;

    _file_QGT_matrix << "#######################################\n";
    _file_QGT_matrix << "# Column Legend\n";
    _file_QGT_matrix << "#\n";
    _file_QGT_matrix << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_QGT_matrix << "#   2nd: the Quantum Geometric Tensor\n";
    _file_QGT_matrix << "#######################################\n";

  }

}


void VMC_Sampler :: setFile_QGT_cond(std::string info, int rank) {

  _write_QGT_cond = true;
  if(rank == 0){

    _file_QGT_cond.open("qgt_cond_" + info + ".dat");
    if(!_file_QGT_cond.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_cond_" << info << ".dat ›› for writing the Quantum Geometric Tensor condition number." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the QGT condition number after each 𝐭𝐕𝐌𝐂 step on file ‹‹ qgt_cond_" << info << ".dat ››." << std::endl;

    if(_vqs.type_of_ansatz() == "Neural Network"){

      _file_QGT_cond << "###########################################################################\n";
      _file_QGT_cond << "# Column Legend\n";
      _file_QGT_cond << "#\n";
      _file_QGT_cond << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_cond << "#   2nd: the QGT condition number (real part) (no regularization)\n";
      _file_QGT_cond << "#   3rd: the QGT condition number (imaginary part) (no regularization)\n";
      _file_QGT_cond << "#   4th: the QGT condition number (real part) (with regularization)\n";
      _file_QGT_cond << "#   5th: the QGT condition number (imaginary part) (with regularization)\n";
      _file_QGT_cond << "###########################################################################\n";

    }
    else{

      _file_QGT_cond << "###########################################################################\n";
      _file_QGT_cond << "# Column Legend\n";
      _file_QGT_cond << "#\n";
      _file_QGT_cond << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_cond << "#   2nd: the QGT condition number (no regularization)\n";
      _file_QGT_cond << "#   3th: the QGT condition number (with regularization)\n";
      _file_QGT_cond << "###########################################################################\n";

    }

  }

}


void VMC_Sampler :: setFile_QGT_eigen(std::string info, int rank) {

  _write_QGT_eigen = true;
  if(rank == 0){

    _file_QGT_eigen.open("qgt_eigen_" + info + ".dat");
    if(!_file_QGT_eigen.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_eigen_" << info << ".dat ›› for writing the eigenvalues of the Quantum Geometric Tensor." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the QGT eigenvalues after each 𝐭𝐕𝐌𝐂 step on file ‹‹ qgt_eigen_" << info << ".dat ››." << std::endl;

    _file_QGT_eigen << "#####################################\n";
    _file_QGT_eigen << "# Column Legend\n";
    _file_QGT_eigen << "#\n";
    _file_QGT_eigen << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_QGT_eigen << "#   2nd: the QGT eigenvalues\n";
    _file_QGT_eigen << "#####################################\n";

  }

}


void VMC_Sampler :: Write_Move_Statistics(unsigned int tvmc_step, MPI_Comm common) {

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

    _file_Move_Statistics << tvmc_step + 1;
    //_file_Move_Statistics << std::scientific;
    _file_Move_Statistics << std::setprecision(2) << std::fixed;
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

      //Prints the visible configuration |𝒗⟩
      _file_MCMC_Config << "\t\t|𝒗 ⟩" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
          _file_MCMC_Config << _configuration.at(j_row, j_col) << std::setw(4);
        _file_MCMC_Config << std::endl << "   " << std::setw(4);

      }

      //Prints the ket configuration |𝒉⟩
      if(_hidden_ket.is_empty()) _file_MCMC_Config << "\t\t|𝒉 ⟩" << std::endl;
      else{

        _file_MCMC_Config << "\t\t|𝒉 ⟩" << std::setw(4);
        for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
            _file_MCMC_Config << _hidden_ket.at(j_row, j_col) << std::setw(4);
          _file_MCMC_Config << std::endl << "   " << std::setw(4);

        }

      }

      //Prints the bra configuration ⟨𝒉ˈ|
      if(_hidden_bra.is_empty()) _file_MCMC_Config << "\t\t⟨𝒉ˈ|" << std::endl;
      else{

        _file_MCMC_Config << "\t\t⟨𝒉ˈ|" << std::setw(4);
        for(unsigned int j_row = 0; j_row < _hidden_bra.n_rows; j_row++){

          for(unsigned int j_col = 0; j_col < _hidden_bra.n_cols; j_col++)
            _file_MCMC_Config << _hidden_bra.at(j_row, j_col) << std::setw(4);
          _file_MCMC_Config << std::endl;

        }

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_final_Config(unsigned int tvmc_step) {

  if(_write_final_Config){

    _file_final_Config << tvmc_step + 1 << "\t\t|𝒗 ⟩" << std::setw(4);
    //Prints the visible configuration |𝒗 ⟩
    for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++){

      for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++)
        _file_final_Config << _configuration.at(j_row, j_col) << std::setw(4);
      _file_final_Config << std::endl << "   " << std::setw(4);

    }

    //Prints the ket configuration |𝒉 ⟩
    if(_hidden_ket.is_empty()) _file_final_Config << "\t\t|𝒉 ⟩" << std::endl;
    else{

      _file_final_Config << "\t\t|𝒉 ⟩" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_ket.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_ket.n_cols; j_col++)
          _file_final_Config << _hidden_ket.at(j_row, j_col) << std::setw(4);
        _file_final_Config << std::endl;;

      }

    }

    //Prints the bra configuration ⟨𝒉ˈ|
    if(_hidden_bra.is_empty()) _file_final_Config << "\t\t⟨𝒉ˈ|" << std::endl;
    else{

      _file_final_Config << "\t\t⟨𝒉ˈ|" << std::setw(4);
      for(unsigned int j_row = 0; j_row < _hidden_bra.n_rows; j_row++){

        for(unsigned int j_col = 0; j_col < _hidden_bra.n_cols; j_col++)
          _file_final_Config << _hidden_bra.at(j_row, j_col) << std::setw(4);
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

      if(_vqs.type_of_ansatz()  == "Neural Network") _file_opt_Params << _vqs.n_visible() << "\n" << _vqs.density()*_vqs.n_visible() << std::endl;
      else _file_opt_Params << _vqs.n_visible() << std::endl;
      if(_vqs.if_phi_neq_zero()) _file_opt_Params << _vqs.phi() << std::endl;
      for(unsigned int p = 0; p < _vqs.n_alpha(); p++) _file_opt_Params << _vqs.alpha_at(p) << std::endl;

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_all_Params(unsigned int tvmc_step, int rank) {

  if(_write_all_Params){

    if(rank == 0){

      _file_all_Params << tvmc_step + 1;
      _file_all_Params << std::setprecision(8) << std::fixed;
      _file_all_Params << "\t" << _vqs.phi().real() << "\t" << _vqs.phi().imag() << std::endl;
      for(unsigned int p = 0; p < _vqs.n_alpha(); p++)
        _file_all_Params << tvmc_step + 1 << "\t" << _vqs.alpha_at(p).real() << "\t" << _vqs.alpha_at(p).imag() << std::endl;

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_matrix(unsigned int tvmc_step) {

  if(_write_QGT_matrix){

    _file_QGT_matrix << tvmc_step + 1 << "\t\t";
    _file_QGT_matrix << std::setprecision(10) << std::fixed;
    if(_vqs.type_of_ansatz() != "Neural Network"){

      for(unsigned int j = 0; j < _Q.row(0).n_elem; j++) _file_QGT_matrix << _Q.row(0)[j].real() << "  ";
      _file_QGT_matrix << std::endl;
      for(unsigned int d = 1; d < _Q.n_rows; d++){

        _file_QGT_matrix << "\t\t";
        for(unsigned int j = 0; j < _Q.row(d).n_elem; j++) _file_QGT_matrix << _Q.row(d)[j].real() << "  ";
        _file_QGT_matrix << std::endl;

      }

    }
    else{

      for(unsigned int j = 0; j < _Q.row(0).n_elem; j++) _file_QGT_matrix << _Q.row(0)[j] << "  ";
      _file_QGT_matrix << std::endl;
      for(unsigned int d = 1; d < _Q.n_rows; d++){

        _file_QGT_matrix << "\t\t";
        for(unsigned int j = 0; j < _Q.row(d).n_elem; j++) _file_QGT_matrix << _Q.row(d)[j] << "  ";
        _file_QGT_matrix << std::endl;

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_cond(unsigned int tvmc_step) {

  if(_write_QGT_cond){

    _file_QGT_cond << tvmc_step + 1 << "\t\t";
    _file_QGT_cond << std::setprecision(0) << std::fixed;
    if(_vqs.type_of_ansatz() != "Neural Network"){

      double C;
      if(_if_QGT_reg == true && _reg_method == 0){

        C = cond(real(_Q));
        _file_QGT_cond << C << "\t";
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
      if(_if_QGT_reg == true && _reg_method == 0){

        C = cond(_Q);
        _file_QGT_cond << C.real() << "\t" << C.imag() << "\t";
        C = cond(_Q + _eps * _I);
        _file_QGT_cond << C.real() << "\t" << C.imag() << std::endl;

      }
      else{

        C = cond(_Q);
        _file_QGT_cond << C.real() << "\t" << C.imag() << std::endl;

      }

    }

  }
  else
    return;

}


void VMC_Sampler :: Write_QGT_eigen(unsigned int tvmc_step) {

  if(_write_QGT_eigen){

    if(_vqs.type_of_ansatz() != "Neural Network"){

      vec eigenval;
      if(_if_QGT_reg == true && _reg_method == 0){

        if(eig_sym(eigenval, real(_Q) + _eps * _I) == true){

          for(unsigned int e = 0; e < eigenval.n_elem; e++) _file_QGT_eigen << tvmc_step + 1 << "\t" << eigenval[e] << "\n";

        }
        else _file_QGT_eigen << tvmc_step + 1 << "\t" << "Armadillo decomposition fails!\n";

      }
      else eigenval = eig_sym(real(_Q));
      for(unsigned int e = 0; e < eigenval.n_elem; e++) _file_QGT_eigen << tvmc_step + 1 << "\t" << eigenval[e] << "\n";

    }
    else{

      cx_vec eigenval;
      if(_if_QGT_reg) eigenval = eig_gen(_Q + _eps * _I);
      else eigenval = eig_gen(_Q);
      for(unsigned int e = 0; e < eigenval.n_elem; e++) _file_QGT_eigen << tvmc_step + 1 << "\t" << eigenval[e] << "\n";

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
    _file_block_Energy.close();
    _file_opt_Energy.close();
    if(_write_block_Observables){

      _file_block_SigmaX.close();
      _file_block_SzSz.close();
      _file_block_SzSzCorr.close();

    }
    if(_write_opt_Observables){

      _file_opt_SigmaX.close();
      _file_opt_SzSz.close();
      _file_opt_SzSzCorr.close();

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

  /*###########################################################*/
  //  This function must be called every time a new 𝐭𝐕𝐌𝐂 step
  //  is about to begin. In fact, it performs an appropriate
  //  initialization of all the variables necessary for the
  //  stochastic estimation of the quantum observables.
  /*###########################################################*/

  _instReweight.reset();
  _instO_ket.reset();
  _instO_bra.reset();
  _instObs_ket.reset();
  _instObs_bra.reset();
  _instSquareMag.reset();
  _instSpinSpinCorr.reset();

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
  //        𝒜 ≡ 𝒜(𝒗) = Σ𝒗' ⟨𝒗|𝔸|𝒗'⟩ • Ψ(𝒗',𝛂)/Ψ(𝒗,𝛂)        (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝒜 ≡ 𝒜(𝒗,𝒉) = Σ𝒗' ⟨𝒗|𝔸|𝒗'⟩ • Φ(𝒗',𝒉,𝛂)/Φ(𝒗,𝒉,𝛂)  (𝒮𝒽𝒶𝒹ℴ𝓌)
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
  //        𝓞(𝒗,𝒉) = ∂𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂)) / ∂𝛂
  //        𝓞(𝒗,𝒉ˈ) = ∂𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂)) / ∂𝛂
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
  unsigned int n_props;  //Number of quantum observables to be computed via 𝐌𝐂
  if(_write_block_Observables || _write_opt_Observables) n_props = _Connections.n_rows;
  else n_props = 1;  //Only energy computation
  Row <double> magnetization;  //Storage variable for (𝗠 ᶻ)^2
  Col <double> Cofr;  //Storage variable for 𝗖ᶻ(𝙧)
  _Observables.set_size(n_props, 1);  //Only sizing, this should be computed in 𝐄𝐬𝐭𝐢𝐦𝐚𝐭𝐞()
  _global_Observables.set_size(n_props, 1);  //Only sizing, this should be computed later
  Col <double> cosin(2, fill::zeros);  //Storage variable for cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')] and sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
  Col <std::complex <double>> A_ket(n_props, fill::zeros);  //Storage variable for 𝒜(𝒗,𝒉)
  Col <std::complex <double>> A_bra(n_props, fill::zeros);  //Storage variable for 𝒜(𝒗,𝒉ˈ)
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
        cosin[0] += _vqs.cosII(_configuration, _hidden_ket, _hidden_bra);
        cosin[1] += _vqs.sinII(_configuration, _hidden_ket, _hidden_bra);
        _vqs.LocalOperators(_configuration, _hidden_ket, _hidden_bra);
        O_ket += _vqs.O().col(0);
        O_bra += _vqs.O().col(1);
        for(unsigned int Nobs = 0; Nobs < n_props; Nobs++){

          for(unsigned int mel = 0; mel < _Connections[Nobs].n_elem; mel++){

            A_ket[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _hidden_ket);  // 𝒜(𝒗,𝒉)
            A_bra[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _hidden_bra);  // 𝒜(𝒗,𝒉')

          }

        }

      }
      cosin /= double(_Nextra);  //  ⟨⟨𝑐𝑜𝑠⟩ᵇˡᵏ⟩ & ⟨⟨𝑠𝑖𝑛⟩ᵇˡᵏ⟩
      A_ket /= double(_Nextra);  //  ⟨⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ⟩
      A_bra /= double(_Nextra);  //  ⟨⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ⟩
      O_ket /= double(_Nextra);  //  ⟨⟨𝓞(𝒗,𝒉)⟩ᵇˡᵏ⟩
      O_bra /= double(_Nextra);  //  ⟨⟨𝓞(𝒗,𝒉')⟩ᵇˡᵏ⟩

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

            for(unsigned int mel = 0; mel < _Connections[Nobs].n_elem; mel++){

              A_ket_blk[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _hidden_ket);  // 𝒜(𝒗,𝒉)
              A_bra_blk[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _hidden_bra);  // 𝒜(𝒗,𝒉')

            }

          }

        }
        cosin[0] += cos_blk / double(blk_size);  // ⟨𝑐𝑜𝑠⟩ᵇˡᵏ
        cosin[1] += sin_blk / double(blk_size);  // ⟨𝑠𝑖𝑛⟩ᵇˡᵏ
        A_ket += A_ket_blk / double(blk_size);  //  ⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ
        A_bra += A_bra_blk / double(blk_size);  //  ⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ
        O_ket += O_ket_blk / double(blk_size);  //  ⟨𝓞(𝒗,𝒉)⟩ᵇˡᵏ
        O_bra += O_bra_blk / double(blk_size);  //  ⟨𝓞(𝒗,𝒉')⟩ᵇˡᵏ

      }
      cosin /= double(_Nblks_extra);  //  ⟨⟨𝑐𝑜𝑠⟩ᵇˡᵏ⟩ & ⟨⟨𝑠𝑖𝑛⟩ᵇˡᵏ⟩
      A_ket /= double(_Nblks_extra);  //  ⟨⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ⟩
      A_bra /= double(_Nblks_extra);  //  ⟨⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ⟩
      O_ket /= double(_Nblks_extra);  //  ⟨⟨𝓞(𝒗,𝒉)⟩ᵇˡᵏ⟩
      O_bra /= double(_Nblks_extra);  //  ⟨⟨𝓞(𝒗,𝒉')⟩ᵇˡᵏ⟩

    }

  }

  else{

    //Computes cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')] and sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cosin[0] = _vqs.cosII(_configuration, _hidden_ket, _hidden_bra);
    cosin[1] = _vqs.sinII(_configuration, _hidden_ket, _hidden_bra);

    //Instantaneous evaluation of the quantum observables
    for(unsigned int Nobs = 0; Nobs < n_props; Nobs++){

      for(unsigned int mel = 0; mel < _Connections[Nobs].n_elem; mel++){

        A_ket[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _hidden_ket);  // 𝒜(𝒗,𝒉)
        A_bra[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _hidden_bra);  // 𝒜(𝒗,𝒉')

      }

    }

    //Instantaneous evaluation of the local operators
    _vqs.LocalOperators(_configuration, _hidden_ket, _hidden_bra);  //Computes 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉')
    O_ket = _vqs.O().col(0);
    O_bra = _vqs.O().col(1);

  }

  //Computes diagonal observables (𝗠 ᶻ)^2 and 𝗖ᶻ(𝙧)
  if(_write_block_Observables || _write_opt_Observables){

    magnetization.zeros(1);
    Cofr.zeros(int(_Nspin / 2) + 1);

    //Instantaneous squared magnetization (𝗠 ᶻ)^2 = (Σⱼ 𝜎ⱼᶻ)(Σₖ 𝜎ₖᶻ)
    for(unsigned int j_row = 0; j_row < _configuration.n_rows; j_row++)
      for(unsigned int j_col = 0; j_col < _configuration.n_cols; j_col++) magnetization[0] += double(_configuration.at(j_row, j_col));

    //Instantaneous 𝗖ᶻ(𝙧)
    if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

      for(unsigned int j = 0; j < _Nspin; j++){

        for(unsigned int k = 0; k <= int(_Nspin / 2); k++)
          if(j + k < _Nspin) Cofr[k] += _configuration.at(0, j) * _configuration.at(0, j + k);

      }

    }
    else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

      /*
        .............
        .............
        .............
      */

    }

    _instSquareMag.insert_cols(_instSquareMag.n_cols, magnetization % magnetization);  // ≡ instantaneous measure of (𝗠 ᶻ)^2
    _instSpinSpinCorr.insert_cols(_instSpinSpinCorr.n_cols, Cofr);  // ≡ instantaneous measure of 𝗖ᶻ(𝙧)

  }

  //Adds Monte Carlo statistics
  _instReweight.insert_cols(_instReweight.n_cols, cosin);  // ≡ instantaneous measure of the 𝑐𝑜𝑠 and of the 𝑠𝑖𝑛
  _instObs_ket.insert_cols(_instObs_ket.n_cols, A_ket);  // ≡ instantaneous measure of 𝒜(𝒗,𝒉)
  _instObs_bra.insert_cols(_instObs_bra.n_cols, A_bra);  // ≡ instantaneous measure of 𝒜(𝒗,𝒉')
  _instO_ket.insert_cols(_instO_ket.n_cols, O_ket);  // ≡ instantaneous measure of 𝓞(𝒗,𝒉)
  _instO_bra.insert_cols(_instO_bra.n_cols, O_bra);  // ≡ instantaneous measure of 𝓞(𝒗,𝒉')

}


void VMC_Sampler :: Estimate(MPI_Comm common) {

  /*#############################################################################################*/
  //  This function is called at the end of the single 𝐭𝐕𝐌𝐂 step and
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
  //        _𝐢𝐧𝐬𝐭𝐎_𝐤𝐞𝐭  ‹--›  𝓞(𝒗,𝒉)
  //        _𝐢𝐧𝐬𝐭𝐎_𝐛𝐫𝐚  ‹--›  𝓞(𝒗,𝒉')
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


void VMC_Sampler :: Write_Quantum_properties(unsigned int tvmc_step, int rank) {

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
      Col <std::complex <double>> prog_energy = this -> compute_progressive_averages(_global_Observables.at(0, 0));
      Col <std::complex <double>> err_energy = this -> compute_errorbar(_global_Observables.at(0, 0));
      Col <std::complex <double>> prog_Sx;
      Col <std::complex <double>> err_Sx;
      if(_write_opt_Observables || _write_block_Observables){

        prog_Sx = this -> compute_progressive_averages(_global_Observables.at(1, 0));
        err_Sx = this -> compute_errorbar(_global_Observables.at(1, 0));

      }

      //Writes energy
      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        _file_block_Energy << std::setprecision(10) << std::fixed;
        _file_block_Energy << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
        _file_block_Energy << prog_cos[block_ID] << "\t" << err_cos[block_ID] << "\t";
        _file_block_Energy << prog_sin[block_ID] << "\t" << err_sin[block_ID] << "\t";
        _file_block_Energy << prog_energy[block_ID].real() << "\t" << err_energy[block_ID].real() << "\t";
        _file_block_Energy << prog_energy[block_ID].imag() << "\t" << err_energy[block_ID].imag() << "\t";
        _file_block_Energy << std::endl;

      }
      _file_opt_Energy << std::setprecision(5) << std::fixed;
      _file_opt_Energy << tvmc_step + 1 << "\t";
      _file_opt_Energy << prog_cos[_Nblks - 1] << "\t" << err_cos[_Nblks - 1] << "\t";
      _file_opt_Energy << prog_sin[_Nblks - 1] << "\t" << err_sin[_Nblks - 1] << "\t";
      _file_opt_Energy << std::setprecision(18) << std::fixed;
      _file_opt_Energy << prog_energy[_Nblks - 1].real() << "\t" << err_energy[_Nblks - 1].real() << "\t";
      _file_opt_Energy << prog_energy[_Nblks - 1].imag() << "\t" << err_energy[_Nblks - 1].imag() << "\t";
      _file_opt_Energy << _E.real() << "\t" << _E.imag();
      _file_opt_Energy << std::endl;

      //Writes system properties on files
      if(_write_block_Observables){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          //Writes σ̂ˣ
          _file_block_SigmaX << std::setprecision(10) << std::fixed;
          _file_block_SigmaX << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_SigmaX << prog_Sx[block_ID].real() << "\t" << err_Sx[block_ID].real() << "\t";
          _file_block_SigmaX << prog_Sx[block_ID].imag() << "\t" << err_Sx[block_ID].imag() << "\t";
          _file_block_SigmaX << std::endl;

        }

      }

      //Saves optimized quantum observables along the 𝐭𝐕𝐌𝐂
      if(_write_opt_Observables){

        // 𝝈(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝝈(𝜙, 𝛂)]
        _file_opt_SigmaX << std::setprecision(20) << std::fixed;
        _file_opt_SigmaX << tvmc_step + 1 << "\t";
        _file_opt_SigmaX << prog_Sx[_Nblks - 1].real() << "\t" << err_Sx[_Nblks - 1].real() << "\t";
        _file_opt_SigmaX << prog_Sx[_Nblks - 1].imag() << "\t" << err_Sx[_Nblks - 1].imag() << "\t";
        _file_opt_SigmaX << std::endl;

      }

    }
    else{

      //Computes the true Shadow observable via reweighting ratio in each block
      Col <double> shadow_energy = real(_global_Observables(0, 0)) / _global_cosII;  //Computes ⟨Ĥ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
      Col <double> shadow_Sx;
      Col <double> prog_Sx;
      Col <double> err_Sx;

      //Computes progressive averages of quantum observables with "block averaging" uncertainties
      Col <double> prog_energy = this -> compute_progressive_averages(shadow_energy);
      Col <double> err_energy = this -> compute_errorbar(shadow_energy);
      if(_write_opt_Observables || _write_block_Observables){

        shadow_Sx = real(_global_Observables(1, 0)) / _global_cosII;  //Computes ⟨σ̂ˣ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
        prog_Sx = this -> compute_progressive_averages(shadow_Sx);
        err_Sx = this -> compute_errorbar(shadow_Sx);

      }

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        //Writes energy
        _file_block_Energy << std::setprecision(8) << std::fixed;
        _file_block_Energy << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
        _file_block_Energy << prog_cos[block_ID] << "\t" << err_cos[block_ID] << "\t";
        _file_block_Energy << prog_sin[block_ID] << "\t" << err_sin[block_ID] << "\t";
        _file_block_Energy << std::setprecision(12) << std::fixed;
        _file_block_Energy << real(_global_Observables.at(0, 0)[block_ID]) << "\t" << imag(_global_Observables.at(0, 0)[block_ID]) << "\t";
        _file_block_Energy << prog_energy[block_ID] << "\t" << err_energy[block_ID] << "\t";
        _file_block_Energy << 0.0 << "\t" << 0.0 << "\t";
        _file_block_Energy << std::endl;

      }
      // 𝐸(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝐸(𝜙,𝛂)]
      _file_opt_Energy << std::setprecision(8) << std::fixed;
      _file_opt_Energy << tvmc_step + 1 << "\t";
      _file_opt_Energy << prog_cos[_Nblks - 1] << "\t" << err_cos[_Nblks - 1] << "\t";
      _file_opt_Energy << prog_sin[_Nblks - 1] << "\t" << err_sin[_Nblks - 1] << "\t";
      _file_opt_Energy << std::setprecision(12) << std::fixed;
      _file_opt_Energy << prog_energy[_Nblks - 1] << "\t" << err_energy[_Nblks - 1] << "\t";
      _file_opt_Energy << 0.0 << "\t" << 0.0 << "\t" << _E.real() << " " << _E.imag();
      _file_opt_Energy << std::endl;


      //Writes all system properties computations on files
      if(_write_block_Observables){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          //Writes σ̂ˣ
          _file_block_SigmaX << std::setprecision(15) << std::fixed;
          _file_block_SigmaX << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_SigmaX << real(_global_Observables.at(1, 0)[block_ID]) << "\t" << imag(_global_Observables.at(1, 0)[block_ID]) << "\t";
          _file_block_SigmaX << prog_Sx(block_ID) << "\t" << err_Sx(block_ID) << "\t";
          _file_block_SigmaX << 0.0 << "\t" << 0.0 << "\t";
          _file_block_SigmaX << std::endl;

        }

      }

      //Saves optimized quantum observables along the 𝐭𝐕𝐌𝐂
      if(_write_opt_Observables){

        // 𝝈(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝝈(𝜙, 𝛂)]
        _file_opt_SigmaX << std::setprecision(20) << std::fixed;
        _file_opt_SigmaX << tvmc_step + 1 << "\t";
        _file_opt_SigmaX << prog_Sx[_Nblks - 1] << "\t" << err_Sx[_Nblks - 1] << "\t";
        _file_opt_SigmaX << 0.0 << "\t" << 0.0 << "\t";
        _file_opt_SigmaX << std::endl;

      }

    }

    if(_write_block_Observables || _write_opt_Observables){

      Col <double> prog_Mz2 = this -> compute_progressive_averages(_globalMz2);
      Col <double> err_Mz2 = this -> compute_errorbar(_globalMz2);
      Mat <double> prog_Cofr(_Nblks, _SpinSpinCorr.n_cols);
      Mat <double> err_Cofr(_Nblks, _SpinSpinCorr.n_cols);
      for(unsigned int d = 0; d < prog_Cofr.n_cols; d++){

        prog_Cofr.col(d) = this -> compute_progressive_averages(_globalCofr.col(d));
        err_Cofr.col(d) = this -> compute_errorbar(_globalCofr.col(d));

      }

      if(_write_block_Observables){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          //Writes (𝗠 ᶻ)^2
          _file_block_SzSz << std::setprecision(10) << std::fixed;
          _file_block_SzSz << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_block_SzSz << prog_Mz2[block_ID] << "\t" << err_Mz2[block_ID];
          _file_block_SzSz << std::endl;

          //Writes 𝜎ᶻ𝜎ᶻ(𝙧)
          for(unsigned int r = 0; r < prog_Cofr.n_cols; r++){

            _file_block_SzSzCorr << std::setprecision(10) << std::fixed;
            _file_block_SzSzCorr << tvmc_step + 1 << "\t" << block_ID + 1 << "\t" << r << "\t";
            _file_block_SzSzCorr << prog_Cofr.at(block_ID, r) << "\t" << err_Cofr.at(block_ID, r);
            _file_block_SzSzCorr << std::endl;

          }

        }

      }
      if(_write_opt_Observables){

        //Writes (𝗠 ᶻ)^2
        _file_opt_SzSz << std::setprecision(20) << std::fixed;
        _file_opt_SzSz << tvmc_step + 1 << "\t";
        _file_opt_SzSz << prog_Mz2[_Nblks - 1] << "\t" << err_Mz2[_Nblks - 1];
        _file_opt_SzSz << std::endl;

        //Writes 𝜎ᶻ𝜎ᶻ(𝙧)
        for(unsigned int r = 0; r < prog_Cofr.n_cols; r++){

          _file_opt_SzSzCorr << std::setprecision(20) << std::fixed;
          _file_opt_SzSzCorr << tvmc_step + 1 << "\t" << r << "\t";
          _file_opt_SzSzCorr << prog_Cofr.at(_Nblks - 1, r) << "\t" << err_Cofr.at(_Nblks - 1, r);
          _file_opt_SzSzCorr << std::endl;

        }

      }

    }

    if(!_if_vmc){

      this -> Write_QGT_matrix(tvmc_step);
      this -> Write_QGT_cond(tvmc_step);
      this -> Write_QGT_eigen(tvmc_step);

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
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++) sum_in_each_block += instantaneous_quantity[l];
    blocks_quantity[block_ID] = sum_in_each_block / double(blk_size);

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
    for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++) sum_in_each_block += instantaneous_quantity[l];
    blocks_quantity[block_ID] = sum_in_each_block / double(blk_size);

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

      sum_in_each_block += _instReweight.row(0)[l] * (instantaneous_quantity_ket[l].real() + instantaneous_quantity_bra[l].real());
      sum_in_each_block += _instReweight.row(1)[l] * (instantaneous_quantity_bra[l].imag() - instantaneous_quantity_ket[l].imag());

    }
    sum_in_each_block *= 0.5;
    blocks_quantity[block_ID] = sum_in_each_block / double(blk_size);

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
      angled_sum_in_each_block += _instReweight.row(0)[l] * (instantaneous_quantity_ket[l].real() + instantaneous_quantity_bra[l].real());
    angled_sum_in_each_block *= 0.5;
    blocks_angled_quantity[block_ID] = angled_sum_in_each_block / double(blk_size);

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
      square_sum_in_each_block += _instReweight.row(1)[l] * (instantaneous_quantity_bra[l].imag() - instantaneous_quantity_ket[l].imag());
    square_sum_in_each_block *= 0.5;
    blocks_square_quantity[block_ID] = square_sum_in_each_block / double(blk_size);

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
  _cosII = this -> average_in_blocks(_instReweight.row(0));  //Computes ⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block, for j = 𝟣,…,𝖭ᵇˡᵏ
  _sinII = this -> average_in_blocks(_instReweight.row(1));  //Computes ⟨𝑠𝑖𝑛⟩ⱼᵇˡᵏ in each block, for j = 𝟣,…,𝖭ᵇˡᵏ

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
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐄𝐧𝐞𝐫𝐠𝐲.
  //  We compute the stochastic average via the Blocking technique of
  //
  //        𝐸(𝜙,𝛂) = ⟨Ĥ⟩ ≈ ⟨ℰ⟩            (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝐸(𝜙,𝛂) = ⟨Ĥ⟩ ≈ ≪ℰᴿ≫ + ⌈ℰᴵ⌋   (𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  We remember that the matrix rows _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭(0) and _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚(0) contains
  //  the instantaneous values of the Hamiltonian operator along the MCMC, i.e.
  //  ℰ(𝒗,𝒉) and ℰ(𝒗,𝒉ˈ).
  /*#################################################################################*/
  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐒𝐢𝐧𝐠𝐥𝐞 𝐒𝐩𝐢𝐧 𝐎𝐛𝐬𝐞𝐫𝐯𝐚𝐛𝐥𝐞𝐬.
  //  We compute the stochastic average via the Blocking technique of
  //
  //        𝝈ˣ(𝜙,𝛂) = ⟨𝞼ˣ⟩ ≈ ⟨𝜎ˣ⟩             (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝝈ˣ(𝜙,𝛂) = ⟨𝞼ˣ⟩ ≈ ≪𝜎ˣᴿ≫ + ⌈𝜎ˣᴵ⌋   (𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //  and so on for the others quantum properties.
  /*#################################################################################*/

  //MPI variables for parallelization
  int rank,size;
  MPI_Comm_size(common, &size);
  MPI_Comm_rank(common, &rank);

  //Computes ⟨𝒪⟩ⱼᵇˡᵏ in each block
  for(unsigned int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++) _global_Observables.at(n_obs, 0).zeros(_Nblks);

  if(!_if_shadow){

    for(unsigned int n_obs = 0; n_obs < _Observables.n_rows; n_obs++)
      _Observables.at(n_obs, 0) = this -> average_in_blocks(_instObs_ket.row(n_obs));

  }
  else{

    for(unsigned int n_obs = 0; n_obs < _Observables.n_rows; n_obs++){

      _Observables.at(n_obs, 0).set_size(_Nblks);
      _Observables.at(n_obs, 0).set_real(this -> Shadow_average_in_blocks(_instObs_ket.row(n_obs), _instObs_bra.row(n_obs)));
      _Observables.at(n_obs, 0).set_imag(zeros(_Nblks));

    }

  }

  MPI_Barrier(common);

  //Shares block averages among all the nodes
  for(unsigned int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++)
    MPI_Reduce(_Observables.at(n_obs, 0).begin(), _global_Observables.at(n_obs, 0).begin(), _Nblks, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);

  if(rank == 0)
    for(unsigned int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++) _global_Observables.at(n_obs, 0) /= double(size);

  //Computes ⟨(𝗠 ᶻ)^2⟩ⱼᵇˡᵏ and ⟨𝗖ⱼₖ(𝙧)⟩ⱼᵇˡᵏ in each block
  if(_write_block_Observables || _write_opt_Observables){

    _SpinSpinCorr.set_size(_Nblks, _instSpinSpinCorr.n_rows);
    _globalMz2.zeros(_Nblks);
    _globalCofr.zeros(_Nblks, _instSpinSpinCorr.n_rows);

    _squareMag = this -> average_in_blocks(_instSquareMag);
    for(unsigned int r = 0; r < _SpinSpinCorr.n_cols; r++) _SpinSpinCorr.col(r) = this -> average_in_blocks(_instSpinSpinCorr.row(r));

    MPI_Barrier(common);

    //Shares block averages among all the nodes
    MPI_Reduce(_squareMag.begin(), _globalMz2.begin(), _Nblks, MPI_DOUBLE, MPI_SUM, 0, common);
    MPI_Reduce(_SpinSpinCorr.begin(), _globalCofr.begin(), _Nblks * _instSpinSpinCorr.n_rows, MPI_DOUBLE, MPI_SUM, 0, common);

    if(rank == 0){

      _globalMz2 /= double(size);
      _globalCofr /= double(size);

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

      sum_ave += block_averages[j];
      sum_ave_squared += squared_block_averages[j];

    }
    sum_ave /= double(block_ID + 1);
    sum_ave_squared /= double(block_ID + 1);
    if(block_ID == 0) errors[block_ID] = 0.0;
    else errors[block_ID] = std::sqrt(std::abs(sum_ave_squared - sum_ave * sum_ave) / (double(block_ID)));

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
    for(unsigned int j = 0; j < (block_ID + 1); j++) sum_ave += block_averages[j];
    sum_ave /= double(block_ID + 1);
    prog_ave[block_ID] = sum_ave;

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

      _O.at(lo_ID, 0) = this -> average_in_blocks(_instO_ket.row(lo_ID));  // ⟨𝓞ₖ⟩ⱼᵇˡᵏ
      _O.at(lo_ID, 1) = this -> average_in_blocks(conj(_instO_ket.row(lo_ID)));  // ⟨𝓞⋆ₖ⟩ⱼᵇˡᵏ

    }

  }
  else{

    for(unsigned int lo_ID = 0; lo_ID < _O.n_rows; lo_ID++){

      //Computes ≪𝓞ₖ≫ⱼᵇˡᵏ
      _O.at(lo_ID, 0).set_size(_Nblks);
      _O.at(lo_ID, 0).set_real(this -> Shadow_angled_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O.at(lo_ID, 0).set_imag(zeros(_Nblks));

      //Computes ⌈𝓞ₖ⌋ⱼᵇˡᵏ
      _O.at(lo_ID, 1).set_size(_Nblks);
      _O.at(lo_ID, 1).set_real(this -> Shadow_square_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O.at(lo_ID, 1).set_imag(zeros(_Nblks));

    }

  }

}


void VMC_Sampler :: compute_QGTandGrad(MPI_Comm common) {

  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐐𝐮𝐚𝐧𝐭𝐮𝐦 𝐆𝐞𝐨𝐦𝐞𝐭𝐫𝐢𝐜 𝐓𝐞𝐧𝐬𝐨𝐫.
  //  We compute stochastically the 𝐐𝐆𝐓 defined as
  //
  //        ℚ = 𝙎ₘₙ                                  (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝙎ₘₙ ≈ ⟨𝓞⋆ₘ𝓞ₙ⟩ - ⟨𝓞⋆ₘ⟩•⟨𝓞ₙ⟩.
  //
  //        ℚ = 𝙎 + 𝘼•𝘽•𝘼                            (𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝙎ₘₙ ≈ ≪𝓞ₘ𝓞ₙ≫ - ≪𝓞ₘ≫•≪𝓞ₙ≫ - ⌈𝓞ₘ⌋⌈𝓞ₙ⌋
  //        𝘼ₘₙ ≈ -⌈𝓞ₘ𝓞ₙ⌋ + ⌈𝓞ₘ⌋≪𝓞ₙ≫ - ≪𝓞ₘ≫⌈𝓞ₙ⌋
  //        where 𝘽 is the inverse matrix of 𝙎.
  /*#################################################################################*/
  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐄𝐧𝐞𝐫𝐠𝐲 𝐆𝐫𝐚𝐝𝐢𝐞𝐧𝐭.
  //  We compute stochastically the Gradient which drive the optimization defined as
  //
  //        𝔽ₖ ≈ ⟨ℰ𝓞⋆ₖ⟩ - ⟨ℰ⟩•⟨𝓞⋆ₖ⟩                  (𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌)
  //
  //        𝔽ᴿ ≈ 𝞒 - 𝘼•𝘽•𝞨                           (𝒮𝒽𝒶𝒹ℴ𝓌)
  //        𝔽ᴵ ≈ 𝞨 + 𝘼•𝘽•𝞒
  //
  //  with
  //
  //        𝞒ₖ ≈ -⟨Ĥ⟩•⌈𝓞ₖ⌋ + ≪𝓞ₖ•ℰᴵ≫ + ⌈𝓞ₖ•ℰᴿ⌋
  //        𝞨ₖ ≈ ⟨Ĥ⟩•≪𝓞ₖ≫ + ⌈𝓞ₖ•ℰᴵ⌋ - ≪𝓞ₖ•ℰᴿ≫
  //
  //  where 𝘼 and 𝘽 are introduced before in the calculation of ℚ.
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

    Col <std::complex <double>> mean_O(n_alpha);  // ⟨⟨𝓞ₖ⟩ᵇˡᵏ⟩
    Col <std::complex <double>> mean_O_star(n_alpha);  // ⟨⟨𝓞⋆ₖ⟩ᵇˡᵏ⟩
    std::complex <double> block_qgt, block_gradE;

    //Computes 𝐸(𝜙,𝛂) = ⟨Ĥ⟩ stochastically without progressive errorbars
    std::complex <double> E = mean(_Observables.at(0, 0));

    for(unsigned int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O[lo_ID] = mean(_O.at(lo_ID, 0));
      mean_O_star[lo_ID] = mean(_O.at(lo_ID, 1));

    }

    //Computes ℚ = 𝙎ₘₙ stochastically without progressive errorbars
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = 0; n < n_alpha; n++){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          block_qgt = 0.0;
          for(unsigned int l = block_ID * blk_size; l < (block_ID +  1) * blk_size; l++)
            block_qgt += std::conj(_instO_ket.at(m, l)) * _instO_ket.at(n, l);  //Accumulate 𝓞⋆ₘ𝓞ₙ in each block
          Q.at(m, n) += block_qgt / double(blk_size);  // ⟨𝙎ₘₙ⟩ᵇˡᵏ

        }

      }

    }
    Q /= double(_Nblks);  // ⟨ℚ⟩ ≈ ⟨⟨𝙎ₘₙ⟩ᵇˡᵏ⟩
    Q = Q - kron(mean_O_star, mean_O.st());

    //Computes 𝔽 = 𝔽ₖ stochastically without progressive errorbars
    for(unsigned int k = 0; k < n_alpha; k++){

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_gradE = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
          block_gradE += _instObs_ket.at(0, l) * std::conj(_instO_ket.at(k, l));  //Accumulate ℰ𝓞⋆ₖ in each block
        F[k] += block_gradE / double(blk_size);  // ⟨𝔽ₖ⟩ᵇˡᵏ

      }

    }
    F /= double(_Nblks);  // ⟨𝔽⟩ ≈ ⟨⟨𝔽ₖ⟩ᵇˡᵏ⟩
    F = F - E * mean_O_star;

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

    }

  }
  else{

    Col <double> mean_O_angled(n_alpha);  // ⟨≪𝓞ₖ≫ᵇˡᵏ⟩ with reweighting correction
    Col <double> mean_O_square(n_alpha);  // ⟨⌈𝓞ₖ⌋ᵇˡᵏ⟩ with reweighting correction
    Mat <double> S(n_alpha, n_alpha, fill::zeros);  // 𝙎ₘₙ ≈ ≪𝓞ₘ𝓞ₙ≫ - ≪𝓞ₘ≫•≪𝓞ₙ≫ - ⌈𝓞ₘ⌋⌈𝓞ₙ⌋
    Mat <double> A(n_alpha, n_alpha, fill::zeros);  // 𝘼ₘₙ ≈ -⌈𝓞ₘ𝓞ₙ⌋ + ⌈𝓞ₘ⌋≪𝓞ₙ≫ - ≪𝓞ₘ≫⌈𝓞ₙ⌋
    Mat <double> AB;
    Col <double> Gamma(n_alpha, fill::zeros);  // 𝞒ₖ ≈ -⟨Ĥ⟩•⌈𝓞ₖ⌋ + ≪𝓞ₖ•ℰᴵ≫ + ⌈𝓞ₖ•ℰᴿ⌋
    Col <double> Omega(n_alpha, fill::zeros);  // 𝞨ₖ ≈ ⟨Ĥ⟩•≪𝓞ₖ≫ + ⌈𝓞ₖ•ℰᴵ⌋ - ≪𝓞ₖ•ℰᴿ≫
    double block_corr_angled, block_corr_square;
    double mean_cos = mean(_cosII);

    for(unsigned int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O_angled[lo_ID] = mean(real(_O.at(lo_ID, 0))) / mean_cos;
      mean_O_square[lo_ID] = mean(real(_O.at(lo_ID, 1))) / mean_cos;

    }

    //Computes 𝐸(𝜙,𝛂) = ⟨Ĥ⟩ stochastically without progressive errorbars
    std::complex <double> E;
    E.real(mean(real(_Observables.at(0, 0))) / mean_cos);  // ⟨⟨Ĥ⟩ᵇˡᵏ⟩ with reweighting correction
    E.imag(0.0);

    //Computes ℚ = 𝙎 + 𝘼•𝘽•𝘼 stochastically without progressive errorbars
    for(unsigned int m = 0; m < n_alpha; m++){

      for(unsigned int n = m; n < n_alpha; n++){

        for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

          block_corr_angled = 0.0;
          block_corr_square = 0.0;
          for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

            //Accumulate 𝓞ₘ𝓞ₙ in each block (angled part)
            block_corr_angled += _instReweight.at(0, l) * (_instO_ket.at(m, l).real() * _instO_bra.at(n, l).real() + _instO_bra.at(m, l).real() * _instO_ket.at(n, l).real());
            //Accumulate 𝓞ₘ𝓞ₙ in each block (square part)
            if(m != n)
              block_corr_square += _instReweight.at(1, l) * (_instO_bra.at(m, l).real() * _instO_ket.at(n, l).real() - _instO_ket.at(m, l).real() * _instO_bra.at(n, l).real());


          }
          if(m == n)
            S.at(m, m) += 0.5 * block_corr_angled / double(blk_size);  //Computes the diagonal elements of S
          else{

            S.at(m, n) += 0.5 * block_corr_angled / double(blk_size);  //This is a symmetric matrix, so we calculate only the upper triangular matrix
            S.at(n, m) = S.at(m, n);
            A.at(m, n) -= 0.5 * block_corr_square / double(blk_size);  //This is an anti-symmetric matrix, so we calculate only the upper triangular matrix
            A.at(n, m) = (-1.0) * A.at(m, n);

          }

        }

      }

    }
    S /= double(_Nblks);  // ⟨⟨≪𝓞ₘ𝓞ₙ≫ᵇˡᵏ⟩⟩ without reweighting correction
    A /= double(_Nblks);  // ⟨⟨⌈𝓞ₘ𝓞ₙ⌋ᵇˡᵏ⟩⟩ without reweighting correction
    S /= mean_cos;
    A /= mean_cos;
    S = S - kron(mean_O_angled, mean_O_angled.t()) + kron(mean_O_square, mean_O_square.t());
    A = A + kron(mean_O_square, mean_O_angled.t()) - kron(mean_O_angled, mean_O_square.t());
    if(_if_QGT_reg){

      if(_reg_method == 0) AB = A * (S + _eps * _I).i();  // 0 → Diagonal regularization
      else if(_reg_method == 1)  AB =  A * pinv(S);  // 1 → Moore-Penrose pseudo-inverse

    }
    Q.set_real(symmatu(S + AB * A));  // ⟨ℚ⟩ ≈ ⟨⟨𝙎 + 𝘼•𝘽•𝘼⟩ᵇˡᵏ⟩

    //Computes 𝔽 = {𝔽ᴿ, 𝔽ᴵ} stochastically without progressive errorbars
    for(unsigned int k = 0; k < n_alpha; k++){  //Computes ⟨𝞒ₖ⟩ᵇˡᵏ

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate 𝓞ₖ•ℰᴵ in each block (angled part)
          block_corr_angled += _instReweight.at(0, l) * (_instO_ket.at(k, l).real() * _instObs_bra.at(0, l).imag() + _instO_bra.at(k, l).real() * _instObs_ket.at(0, l).imag());
          //Accumulate 𝓞ₖ•ℰᴿ in each block (square part)
          block_corr_square += _instReweight.at(1, l) * (_instO_bra.at(k, l).real() * _instObs_ket.at(0, l).real() - _instO_ket.at(k, l).real() * _instObs_bra.at(0, l).real());

        }
        Gamma[k] += 0.5 * (block_corr_angled + block_corr_square) / double(blk_size);

      }

    }
    for(unsigned int k = 0; k < n_alpha; k++){  //Computes ⟨𝞨ₖ⟩ᵇˡᵏ

      for(unsigned int block_ID = 0; block_ID < _Nblks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(unsigned int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate 𝓞ₖ•ℰᴿ in each block (angled part)
          block_corr_angled += _instReweight.at(0, l) * (_instO_ket.at(k, l).real() * _instObs_bra.at(0, l).real() + _instO_bra.at(k, l).real() * _instObs_ket.at(0, l).real());
          //Accumulate 𝓞ₖ•ℰᴵ in each block (square part)
          block_corr_square += _instReweight.at(1, l) * (_instO_bra.at(k, l).real() * _instObs_ket.at(0, l).imag() - _instO_ket.at(k, l).real() * _instObs_bra.at(0, l).imag());

        }
        Omega[k] += 0.5 * (block_corr_square - block_corr_angled) / double(blk_size);

      }

    }
    Gamma /= double(_Nblks);  // ⟨⟨𝞒ₖ⟩ᵇˡᵏ⟩ without reweighting correction
    Omega /= double(_Nblks);  // ⟨⟨𝞨ₖ⟩ᵇˡᵏ⟩ without reweighting correction
    Gamma /= mean_cos;
    Omega /=  mean_cos;
    Gamma -= E.real() * mean_O_square;  // ⟨𝞒ₖ⟩ with reweighting correction
    Omega += E.real() * mean_O_angled;  // ⟨𝞨ₖ⟩ with reweighting correction
    F.set_real(Gamma - AB * Omega);  // ⟨𝔽ᴿ⟩ ≈ ⟨⟨𝞒 - 𝘼•𝘽•𝞨⟩ᵇˡᵏ⟩
    F.set_imag(Omega + AB * Gamma);  // ⟨𝔽ᴵ⟩ ≈ ⟨⟨𝞨 + 𝘼•𝘽•𝞒⟩ᵇˡᵏ⟩

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

      if(real(_Q).is_symmetric() == false) std::cerr << "  ##EstimationError: the Quantum Geometric Tensor must be symmetric!" << std::endl;
      else return;

    }
    else{

      if(_Q.is_hermitian() == false) std::cerr << "  ##EstimationError: the Quantum Geometric Tensor must be hermitian!" << std::endl;
      else return;

    }

  }

}


void VMC_Sampler :: is_asymmetric(const Mat <double>& A) const {  //Helpful in debugging

  unsigned int failed = 0;

  for(unsigned int m = 0; m < A.n_rows; m++){

    for(unsigned int n = m; n < A.n_cols; n++)
      if(A.at(m, n) != (-1.0) * A.at(n, m)) failed++;

  }

  if(failed != 0) std::cout << "The matrix is not anti-Symmetric." << std::endl;
  else return;

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
      flipped_site.at(j, 0) = _rnd.Rannyu_INT(0, _Nspin-1);  //Choose a random spin to flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_site);
  if(test.n_elem == flipped_site.n_rows) return true;
  else return false;

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
          _configuration.at(0, _flipped_site.at(fs_row, 0)) *= -1;
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
  else return;

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
      flipped_hidden_site.at(j, 0) = _rnd.Rannyu_INT(0, _Nhidden-1);  //Choose a random spin to flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_hidden_site);
  if(test.n_elem == flipped_hidden_site.n_rows) return true;
  else return false;

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
      for(unsigned int fs_row = 0; fs_row < _flipped_ket_site.n_rows; fs_row++){  //Move the quantum ket configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _hidden_ket.at(0, _flipped_ket_site.at(fs_row, 0)) *= -1;
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
  else return;

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
      for(unsigned int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum bra configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _hidden_bra.at(0, _flipped_bra_site.at(fs_row, 0)) *= -1;
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
  else return;

}


void VMC_Sampler :: Move_equal_site(unsigned int Nflips) {

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

          _configuration.at(0, _flipped_site.at(fs_row, 0)) *= -1;
          _hidden_ket.at(0, _flipped_site.at(fs_row, 0)) *= -1;
          _hidden_bra.at(0, _flipped_site.at(fs_row, 0)) *= -1;

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
  else return;

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

    flipped_visible_nn_site.set_size(2 * Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++){

      if(_H.if_pbc()) index_site = _rnd.Rannyu_INT(0, _Nspin-1);
      else index_site  = _rnd.Rannyu_INT(0, _Nspin-2);
      flipped_visible_nn_site.at(j, 0) = index_site;  //Choose a random spin to flip

      //Adds the right nearest neighbor lattice site
      if(_H.if_pbc()){

        if(index_site == _Nspin-1) flipped_visible_nn_site.at(j + 1, 0) = 0;  //Pbc
        else flipped_visible_nn_site.at(j + 1, 0) = index_site + 1;

      }
      else flipped_visible_nn_site.at(j + 1) = index_site + 1;

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
  if(test.n_elem == flipped_visible_nn_site.n_rows) return true;
  else return false;

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
          _configuration.at(0, _flipped_site.at(fs_row, 0)) *= -1;
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
  else return;

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

    flipped_ket_site.set_size(2 * Nflips, 1);
    flipped_bra_site.set_size(2 * Nflips, 1);
    for(unsigned int j = 0; j < Nflips; j++){

      if(_H.if_pbc()){

          index_site_ket = _rnd.Rannyu_INT(0, _Nspin - 1);
          index_site_bra = _rnd.Rannyu_INT(0, _Nspin - 1);

      }
      else{

        index_site_ket  = _rnd.Rannyu_INT(0, _Nspin - 2);
        index_site_bra = _rnd.Rannyu_INT(0, _Nspin - 2);

      }
      flipped_ket_site.at(j, 0) = index_site_ket;  //Choose a random spin to flip
      flipped_bra_site.at(j, 0) = index_site_bra;  //Choose a random spin to flip

      //Adds the right nearest neighbor lattice site
      if(_H.if_pbc()){

        if(index_site_ket == _Nspin - 1) flipped_ket_site.at(j + 1, 0) = 0;  //Pbc
        if(index_site_bra == _Nspin - 1) flipped_bra_site.at(j + 1, 0) = 0;  //Pbc
        else{

          flipped_ket_site.at(j + 1, 0) = index_site_ket + 1;
          flipped_bra_site.at(j + 1, 0) = index_site_bra + 1;

        }

      }
      else{

        flipped_ket_site.at(j + 1) = index_site_ket + 1;
        flipped_bra_site.at(j + 1) = index_site_bra + 1;

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
  if(test_ket.n_elem == flipped_ket_site.n_rows && test_bra.n_elem == flipped_bra_site.n_rows) return true;
  else return false;

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
          _hidden_ket.at(0, _flipped_ket_site.at(fs_row, 0)) *= -1;
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
          _hidden_bra.at(0, _flipped_bra_site.at(fs_row, 0)) *= -1;
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
  else return;

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
    if(_rnd.Rannyu() < _p_equal_site) this -> Move_equal_site(_Nflips);
    if(_rnd.Rannyu() < _p_hidden_nn) this -> Move_hidden_nn_site(_Nflips);

  }
  if(_rnd.Rannyu() < _p_visible_nn) this -> Move_visible_nn_site(_Nflips);

}


void VMC_Sampler :: Make_Sweep() {

  for(unsigned int n_bunch = 0; n_bunch < _M; n_bunch++) this -> Move();

}


void VMC_Sampler :: VMC_Step(MPI_Comm common) {

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
  //  The single 𝐕𝐌𝐂 run allows us to move a single step in the variational
  //  parameter optimization procedure.
  /*###############################################################################################*/

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  //Initialization and Equilibration
  if(_if_restart_from_config) this -> Init_Config(_configuration, _hidden_ket, _hidden_bra);
  else this -> Init_Config();
  for(unsigned int eq_step = 0; eq_step < _Neq; eq_step++) this -> Make_Sweep();

  //Monte Carlo measurement
  for(unsigned int mcmc_step = 0; mcmc_step < _Nsweeps; mcmc_step++){  //The Monte Carlo Markov Chain

    this -> Make_Sweep();  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ (i.e. a new point of the mcmc)
    this -> Measure();  //Measure quantum properties on the new sampled system configuration |𝒮ⁿᵉʷ⟩
    this -> Write_MCMC_Config(mcmc_step, rank);  //Records the sampled |𝒮ⁿᵉʷ⟩

  }

  //Computes the quantum averages
  this -> Estimate(common);

}


void VMC_Sampler :: Euler(MPI_Comm common) {

  /*#########################################################################*/
  //  Updates the variational parameters (𝜙,𝛂) according to the choosen
  //  𝐭𝐕𝐌𝐂 equations of motion through the Euler integration method.
  //  The equations for the parameters optimization are:
  //
  //        ==================
  //          𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌
  //        ==================
  //          • 𝐈𝐦𝐚𝐠𝐢𝐧𝐚𝐫𝐲-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝒊-𝐭𝐕𝐌𝐂)
  //              𝕊(τ)•𝛂̇(τ) = - 𝔽(τ)
  //          • 𝐑𝐞𝐚𝐥-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝐭𝐕𝐌𝐂)
  //              𝕊(𝑡)•𝛂̇(𝑡) =  - 𝑖 • 𝔽(𝑡)
  //
  //        ============
  //          𝒮𝒽𝒶𝒹ℴ𝓌
  //        ============
  //          • 𝐈𝐦𝐚𝐠𝐢𝐧𝐚𝐫𝐲-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝒊-𝐭𝐕𝐌𝐂)
  //              ℚ(τ) • 𝛂̇ᴿ(τ) = 𝔽ᴵ(τ)
  //              ℚ(τ) • 𝛂̇ᴵ(τ) = - 𝔽ᴿ(τ)
  //              𝜙̇ᴿ(τ) = - 𝛂̇ᴿ(τ) • ≪𝓞≫ - 𝛂̇ᴵ(τ) • ⌈𝓞⌋ - ⟨Ĥ⟩
  //              𝜙̇ᴵ(τ) = + 𝛂̇ᴿ(τ) • ⌈𝓞⌋ - 𝛂̇ᴵ(τ) • ≪𝓞≫
  //          • 𝐑𝐞𝐚𝐥-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝐭𝐕𝐌𝐂)
  //              ℚ(𝑡) • 𝛂̇ᴿ(𝑡) = 𝔽ᴿ(𝑡)
  //              ℚ(𝑡) • 𝛂̇ᴵ(𝑡) = 𝔽ᴵ(𝑡)
  //              𝜙̇ᴿ(𝑡) = - 𝛂̇ᴿ(𝑡) • ≪𝓞≫ - 𝛂̇ᴵ(𝑡) • ⌈𝓞⌋
  //              𝜙̇ᴵ(𝑡) = + 𝛂̇ᴿ(𝑡) • ⌈𝓞⌋ - 𝛂̇ᴵ(𝑡) • ≪𝓞≫ - ⟨Ĥ⟩
  //
  //  where in the 𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌 case we assume 𝜙 = 0.
  //  In the Euler method we obtain the new parameters in the following way:
  //  𝒾𝒻
  //
  //        α̇(𝑡) = 𝒻{α(𝑡)}
  //
  //  𝓉𝒽ℯ𝓃
  //
  //        α(𝑡+𝑑𝑡) = α(𝑡) + 𝑑𝑡 • 𝒻{α(𝑡)}
  //
  //  where 𝒻{α(𝑡)} is numerically integrated by using the 𝐬𝐨𝐥𝐯𝐞() method
  //  of the C++ Armadillo library.
  /*#########################################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      Col <std::complex <double>> new_alpha(_vqs.n_alpha());  // α(𝑡+𝑑𝑡)
      std::complex <double> new_phi;  // 𝜙(𝑡+𝑑𝑡)
      if(rank == 0){

        //Function variables
        Col <std::complex <double>> alpha_dot;
        std::complex <double> phi_dot;

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            // 0 → Diagonal regularization
            if(_reg_method == 0) alpha_dot = - _i * (_Q + _eps * _I).i() * _F;

            // 1 → Moore-Penrose pseudo-inverse
            else if(_reg_method == 1) alpha_dot = - _i * pinv(_Q) * _F;

          }
          else alpha_dot = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero()) phi_dot = as_scalar(- alpha_dot.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            // 0 → Diagonal regularization
            if(_reg_method == 0) alpha_dot = solve(_Q + _eps * _I, (-1.0) * _F);

            // 1 → Moore-Penrose pseudo-inverse
            else if(_reg_method == 1) alpha_dot = - pinv(_Q) * _F;

          }
          else alpha_dot = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero()) phi_dot = _i * as_scalar(alpha_dot.st() * _mean_O) - _E.real();

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
     /*  𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      Col <std::complex <double>> new_alpha(_vqs.n_alpha());  // α(𝑡+𝑑𝑡)
      std::complex <double> new_phi;  // 𝜙(𝑡+𝑑𝑡)
      if(rank == 0){

        //Function variables
        Col <double> alpha_dot_re;
        Col <double> alpha_dot_im;
        double phi_dot_re;
        double phi_dot_im;

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            // 0 → Diagonal regularization
            if(_reg_method == 0){

              alpha_dot_re = (real(_Q) + _eps * _I).i() * real(_F);
              alpha_dot_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }

            // 1 → Moore-Penrose pseudo-inverse
            else if(_reg_method == 1){

              alpha_dot_re = pinv(real(_Q)) * real(_F);
              alpha_dot_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            // 0 → Diagonal regularization
            if(_reg_method == 0){

              alpha_dot_re = (real(_Q) + _eps * _I).i() * imag(_F);
              alpha_dot_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }

            // 1 → Moore-Penrose pseudo-inverse
            else if(_reg_method == 1){

              alpha_dot_re = pinv(real(_Q)) * imag(_F);
              alpha_dot_im = - pinv(real(_Q)) * real(_F);

            }

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
  else return;

}


void VMC_Sampler :: Heun(MPI_Comm common) {

  /*###############################################################*/
  //  The Heun method is a so-called predictor-corrector method,
  //  which achieves a second order accuracy.
  //  In the Heun method we first obtain the auxiliary updates
  //  of the variational parameters
  //
  //        𝛂̃(𝑡 + 𝛿𝑡) = 𝛂(𝑡) + 𝛿𝑡•𝒻{α(𝑡)}
  //
  //  as in the Euler method. We remember that
  //
  //        α̇(𝑡) = 𝒻{α(𝑡)}.
  //
  //  These updates are used to performed a second optimization
  //  step via the 𝐕𝐌𝐂_𝐒𝐭𝐞𝐩() function, and then obtained a second
  //  order updates as
  //
  //        𝛂(𝑡 + 𝛿𝑡) = 𝛂(𝑡) + 1/2•𝛿𝑡•[𝒻{α(𝑡)} + f{𝛂̃(𝑡 + 𝛿𝑡)}].
  //
  //  The first 𝐕𝐌𝐂 step in this integration is performed in the
  //  main program.
  /*###############################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_t = _vqs.alpha();  // 𝛂(𝑡)
      Col <std::complex <double>> alpha_dot_t;  // α̇(𝑡) = 𝒻{α(𝑡)}
      Col <std::complex <double>> alpha_dot_tilde_t;  // f{𝛂̃(𝑡 + 𝛿𝑡)}
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
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) alpha_dot_t = - _i * (_Q + _eps * _I).i() * _F;
            else if(_reg_method  == 1) alpha_dot_t = - _i * pinv(_Q) * _F;

          }
          else alpha_dot_t = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero()) phi_dot_t = as_scalar(- alpha_dot_t.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) alpha_dot_t = - (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) alpha_dot_t = - pinv(_Q) * _F;

          }
          else alpha_dot_t = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero()) phi_dot_t = _i * as_scalar(alpha_dot_t.st() * _mean_O) - _E.real();

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
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) alpha_dot_tilde_t = - _i * (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) alpha_dot_tilde_t = - _i * pinv(_Q) * _F;

          }
          else alpha_dot_tilde_t = solve(_Q, - _i * _F);
          if(_vqs.if_phi_neq_zero()) phi_dot_tilde_t = as_scalar(- alpha_dot_tilde_t.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) alpha_dot_tilde_t = - (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) alpha_dot_tilde_t = - pinv(_Q) * _F;

          }
          else alpha_dot_tilde_t = solve(_Q, - _F);
          if(_vqs.if_phi_neq_zero()) phi_dot_tilde_t = _i * as_scalar(alpha_dot_tilde_t.st() * _mean_O) - _E.real();

        }

        //Final update of the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * (alpha_dot_t + alpha_dot_tilde_t);  // 𝛂(𝑡 + 𝛿𝑡)
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
     /*  𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // 𝜙ᴿ(𝑡)
      double phi_t_im = _vqs.phi().imag();  // 𝜙ᴵ(𝑡)
      Col <double> alpha_t_re = real(_vqs.alpha());  // 𝛂ᴿ(𝑡)
      Col <double> alpha_t_im = imag(_vqs.alpha());  // 𝛂ᴵ(𝑡)
      Col <double> alpha_dot_t_re;  // α̇ᴿ(𝑡) = 𝒻{αᴿ(𝑡)}
      Col <double> alpha_dot_t_im;  // α̇ᴵ(𝑡) = 𝒻{αᴵ(𝑡)}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      double phi_dot_t_re;  // 𝜙̇ᴿ(𝑡)
      double phi_dot_t_im;  // 𝜙̇ᴵ(𝑡)
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
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              alpha_dot_t_re = (real(_Q) + _eps * _I).i() * real(_F);
              alpha_dot_t_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_reg_method == 1){

              alpha_dot_t_re = pinv(real(_Q)) * real(_F);
              alpha_dot_t_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              alpha_dot_t_re = (real(_Q) + _eps * _I).i() * imag(_F);
              alpha_dot_t_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_reg_method == 1){

              alpha_dot_t_re = pinv(real(_Q)) * imag(_F);
              alpha_dot_t_im = - pinv(real(_Q)) * real(_F);

            }

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
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              alpha_dot_tilde_t_re = (real(_Q) + _eps * _I).i() * real(_F);
              alpha_dot_tilde_t_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_reg_method == 1){

              alpha_dot_tilde_t_re = pinv(real(_Q)) * real(_F);
              alpha_dot_tilde_t_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method ==  0){

              alpha_dot_tilde_t_re = (real(_Q) + _eps * _I).i() * imag(_F);
              alpha_dot_tilde_t_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_reg_method == 1){

              alpha_dot_tilde_t_re = pinv(real(_Q)) * imag(_F);
              alpha_dot_tilde_t_im = - pinv(real(_Q)) * real(_F);

            }

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
  else return;

}


void VMC_Sampler :: RK4(MPI_Comm common) {

  /*############################################################################*/
  //  The fourth order Runge Kutta method (𝐑𝐊𝟒) is a one-step explicit
  //  method that achieves a fourth-order accuracy by evaluating the
  //  function 𝒻{α(𝑡)} four times at each time-step.
  //  It is defined as follows:
  //
  //        αₖ(𝑡 + 𝛿ₜ) = αₖ(𝑡) + 𝟣/𝟨•𝛿ₜ•[κ𝟣 + κ𝟤 + κ𝟥 + κ𝟦]
  //
  //  where we have defined
  //
  //        κ𝟣 = 𝒻{α(𝑡)}
  //        κ𝟤 = 𝒻{α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣}
  //        κ𝟥 = 𝒻{α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤}
  //        κ𝟦 = 𝒻{α(𝑡) + 𝛿ₜ•κ𝟥}.
  //
  //  We remember that
  //
  //        α̇(𝑡) = 𝒻{α(𝑡)}.
  //
  //  The first 𝐕𝐌𝐂 step in this integration is performed in the main program.
  /*############################################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  𝓃ℴ𝓃-𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      Col <std::complex <double>> alpha_t = _vqs.alpha();  // 𝛂(𝑡)
      std::complex <double> phi_t = _vqs.phi();  // 𝜙(𝑡)
      Col <std::complex <double>> k1;  // κ𝟣 = 𝒻{α(𝑡)}
      Col <std::complex <double>> k2;  // κ𝟤 = 𝒻{α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣}
      Col <std::complex <double>> k3;  // κ𝟥 = 𝒻{α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤}
      Col <std::complex <double>> k4;  // κ𝟦 = 𝒻{α(𝑡) + 𝛿ₜ•κ𝟥}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      std::complex <double> phi_k1, phi_k2, phi_k3, phi_k4;
      std::complex <double> new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k1 = - _i * (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k1 = - _i * pinv(_Q) * _F;

          }
          else k1 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero()) phi_k1 = as_scalar(- k1.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k1 = - (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k1 = - pinv(_Q) * _F;

          }
          else k1 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero()) phi_k1 = _i * as_scalar(k1.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * k1;  // α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣
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
      //Makes a second 𝐕𝐌𝐂 step with parameters α(𝑡) → α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k2 = -_i * (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k2 = -_i * pinv(_Q) * _F;

          }
          else k2 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero()) phi_k2 = as_scalar(- k2.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k2 = - (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k2 = - pinv(_Q) * _F;

          }
          else k2 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero()) phi_k2 = _i * as_scalar(k2.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * k2;  // α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤
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
      //Makes a second 𝐕𝐌𝐂 step with parameters α(𝑡) → α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k3 = -_i * (_Q + _eps * _I).i() * _F;
            else  if(_reg_method == 1) k3 = -_i * pinv(_Q) * _F;

          }
          else k3 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero()) phi_k3 = as_scalar(- k3.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k3 = - (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k3 = - pinv(_Q) * _F;

          }
          else k3 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero()) phi_k3 = _i * as_scalar(k3.st() * _mean_O) - _E.real();

        }

        //Updates the variational parameters
        new_alpha = alpha_t + _delta * k3;  // α(𝑡) + 𝛿ₜ•κ𝟥
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
      //Makes a second 𝐕𝐌𝐂 step with parameters α(𝑡) → α(𝑡) + 𝛿ₜ•κ𝟥
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k4 = -_i * (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k4 = -_i * pinv(_Q) * _F;

          }
          else k4 = solve(_Q, - _i * _F);

          if(_vqs.if_phi_neq_zero()) phi_k4 = as_scalar(- k4.st() * _mean_O) - _i * _E.real();

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0) k4 = - (_Q + _eps * _I).i() * _F;
            else if(_reg_method == 1) k4 = - pinv(_Q) * _F;

          }
          else k4 = solve(_Q, - _F);

          if(_vqs.if_phi_neq_zero()) phi_k4 = _i * as_scalar(k4.st() * _mean_O) - _E.real();

        }

        //Final update of the variational parameters
        new_alpha = alpha_t + (1.0/6.0) * _delta * (k1 + k2 + k3 + k4);  // αₖ(𝑡 + 𝛿ₜ)
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
     /*  𝒮𝒽𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // 𝜙ᴿ(𝑡)
      double phi_t_im = _vqs.phi().imag();  // 𝜙ᴵ(𝑡)
      Col <double> alpha_t_re = real(_vqs.alpha());  // 𝛂ᴿ(𝑡)
      Col <double> alpha_t_im = imag(_vqs.alpha());  // 𝛂ᴵ(𝑡)
      Col <double> k1_re;  // κ𝟣ᴿ = 𝒻{αᴿ(𝑡)}
      Col <double> k1_im;  // κ𝟣ᴵ = 𝒻{αᴵ(𝑡)}
      Col <double> k2_re;  // κ𝟤ᴿ = 𝒻{αᴿ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣ᴿ}
      Col <double> k2_im;  // κ𝟤ᴵ = 𝒻{αᴵ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣ᴵ}
      Col <double> k3_re;  // κ𝟥ᴿ = 𝒻{αᴿ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤ᴿ}
      Col <double> k3_im;  // κ𝟥ᴵ = 𝒻{αᴵ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤ᴵ}
      Col <double> k4_re;  // κ𝟦ᴿ = 𝒻{αᴿ(𝑡) + 𝛿ₜ•κ𝟥ᴿ}
      Col <double> k4_im;  // κ𝟦ᴵ = 𝒻{αᴵ(𝑡) + 𝛿ₜ•κ𝟥ᴵ}
      Col <std::complex <double>> new_alpha(_vqs.n_alpha());
      double phi_k1_re, phi_k2_re, phi_k3_re, phi_k4_re;
      double phi_k1_im, phi_k2_im, phi_k3_im, phi_k4_im;
      std::complex <double> new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k1_re = (real(_Q) + _eps * _I).i() * real(_F);
              k1_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_reg_method == 1){

              k1_re = pinv(real(_Q)) * real(_F);
              k1_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k1_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k1_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_reg_method == 1){

              k1_re = pinv(real(_Q)) * imag(_F);
              k1_im = - pinv(real(_Q)) * real(_F);

            }

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
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k2_re = (real(_Q) + _eps * _I).i() * real(_F);
              k2_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_reg_method == 1){

              k2_re = pinv(real(_Q)) * real(_F);
              k2_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k2_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k2_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_reg_method == 1){

              k2_re = pinv(real(_Q)) * imag(_F);
              k2_im = - pinv(real(_Q)) * real(_F);

            }

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
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k3_re = (real(_Q) + _eps * _I).i() * real(_F);
              k3_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_reg_method == 1){

              k3_re = pinv(real(_Q)) * real(_F);
              k3_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k3_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k3_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_reg_method == 1){

              k3_re = pinv(real(_Q)) * imag(_F);
              k3_im = - pinv(real(_Q)) * real(_F);

            }

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
      //Makes a second 𝐕𝐌𝐂 step at time 𝑡 + 𝛿𝑡
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> VMC_Step(common);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k4_re = (real(_Q) + _eps * _I).i() * real(_F);
              k4_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_reg_method == 1){

              k4_re = pinv(real(_Q)) * real(_F);
              k4_im = pinv(real(_Q)) * imag(_F);

            }

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
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_reg){

            if(_reg_method == 0){

              k4_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k4_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_reg_method == 1){

              k4_re = pinv(real(_Q)) * imag(_F);
              k4_im = - pinv(real(_Q)) * real(_F);

            }

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
  else return;

}


#endif
