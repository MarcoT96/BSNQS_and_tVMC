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


#ifndef __SAMPLER__
#define __SAMPLER__


/**********************************************************************************************************/
/**********************************  𝑽𝒂𝒓𝒊𝒂𝒕𝒊𝒐𝒏𝒂𝒍 𝑴𝒐𝒏𝒕𝒆 𝑪𝒂𝒓𝒍𝒐 𝑺𝒂𝒎𝒑𝒍𝒆𝒓  ****************************************/
/*********************************************************************************************************/
/*

  We create a Variational Quantum Monte Carlo (𝐕𝐌𝐂) sampler as a C++ class, which is able to
  evolve in time a generic 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 (a variational quantum state vqs) in order to study a
  generic lattice quantum system (𝐋𝐐𝐒).
  The main goal of the sampler is to optimize the parameters that uniquely characterize the vqs
  to obtain the ground state of a given Hamiltonian; once found the ground state, it is
  possible to study the real-time quantum dynamics of the system after performing a suddden quench
  on a certain coupling constant.

  The optimization described above takes place within a stochastic setting, in which the
  procedure leads to the resolution of the following equations of motion for the variational
  parameters 𝛂 (𝐭𝐕𝐌𝐂 equations of motion):

            Σₖ α̇ₖ {αⱼ, αₖ} = ∂𝙀[𝛂] / ∂αⱼ      (𝐭𝐕𝐌𝐂)
            Σₖ α̇ₖ {αⱼ, αₖ} = - 𝑖 • ∂𝙀[𝛂] / ∂αⱼ   (𝑖-𝐭𝐕𝐌𝐂)

  where the ground state properties are recovered with an imaginary-time evolution

            𝒕 → 𝝉 = 𝑖𝒕.

  This class is also able to apply the above technique to a 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏, where
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
#include <algorithm>  // <-- std::max()
#include <armadillo>  // <-- arma::Mat, arma::Col, arma::Row, arma::field
#include "random.cpp"  // <-- Random
#include "ansatz.cpp"  // <-- WaveFunction
#include "model.cpp"  // <-- SpinHamiltonian


using namespace arma;
using namespace std::__fs::filesystem;  //Use std::experimental::filesystem if you are in @tolab


class VMC_Sampler {

  private:

    //Variables of the quantum problem
    WaveFunction& _vqs;  //The variational wave function |Ψ(𝜙,𝛂)⟩
    SpinHamiltonian& _H;  //The Spin Hamiltonian Ĥ
    const int _L;  //Number of quantum degrees of freedom in the system

    //Constant data-members
    const cx_double _i;  //The imaginary unit 𝑖
    const mat _I;  //The real identity matrix 𝟙

    //Random device
    Random _rnd;

    //Quantum configuration variables |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
    const int _n_shadows;  //Number of auxiliary quantum variables
    Mat <int> _configuration;  //Current 𝓇ℯ𝒶𝑙 configuration of the system |𝒗⟩ = |𝓋𝟣 𝓋𝟤 … 𝓋𝖫⟩
    Mat <int> _shadow_ket;  //The ket configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables |𝒉⟩ = |𝒽𝟣 𝒽𝟤 … 𝒽𝖬⟩
    Mat <int> _shadow_bra;  //The bra configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ⟨𝒉ˈ| = ⟨𝒽ˈ𝖬 … 𝒽ˈ𝟤 𝒽ˈ𝟣|
    Mat <int> _flipped_site;  //The new sampled 𝓇ℯ𝒶𝑙 configuration |𝒗ⁿᵉʷ⟩
    Mat <int> _flipped_ket_site;  //The new sampled ket configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables |𝒉ⁿᵉʷ⟩
    Mat <int> _flipped_bra_site;  //The new sampled bra configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ⟨𝒉ˈⁿᵉʷ|

    //Monte Carlo moves statistics variables
    int _N_accepted_real;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ accepted along the MCMC
    int _N_proposed_real;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ proposed along the MCMC
    int _N_accepted_ket;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ accepted along the MCMC
    int _N_proposed_ket;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ proposed along the MCMC
    int _N_accepted_bra;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ accepted along the MCMC
    int _N_proposed_bra;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ proposed along the MCMC
    int _N_accepted_equal_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with equal-site spin-flips accepted along the MCMC
    int _N_proposed_equal_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with equal-site spin-flips proposed along the MCMC
    int _N_accepted_real_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ with nearest-neighbors-site spin-flips accepted along the MCMC
    int _N_proposed_real_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ with nearest-neighbors-site spin-flips proposed along the MCMC
    int _N_accepted_shadows_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with nearest-neighbors-site spin-flips accepted along the MCMC
    int _N_proposed_shadows_nn_site;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with nearest-neighbors-site spin-flips proposed along the MCMC
    int _N_accepted_global_ket_flip;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ with global ket spin-flips accepted along the MCMC
    int _N_proposed_global_ket_flip;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ with global ket spin-flips proposed along the MCMC
    int _N_accepted_global_bra_flip;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ with global bra spin-flips accepted along the MCMC
    int _N_proposed_global_bra_flip;  //Number of new configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ with global bra spin-flips proposed along the MCMC
    int _global_acc_real;  //Collects the statistics for _N_accepted_real among all the nodes
    int _global_prop_real;  //Collects the statistics for _N_proposed_real among all the nodes
    int _global_acc_ket;  //Collects the statistics for _N_accepted_ket among all the nodes
    int _global_prop_ket;  //Collects the statistics for _N_proposed_ket among all the nodes
    int _global_acc_bra;  //Collects the statistics for _N_accepted_bra among all the nodes
    int _global_prop_bra;  //Collects the statistics for _N_proposed_bra among all the nodes
    int _global_acc_equal_site;  //Collects the statistics for _N_accepted_equal_site among all the nodes
    int _global_prop_equal_site;  //Collects the statistics for _N_proposed_equal_site among all the nodes
    int _global_acc_real_nn_site;  //Collects the statistics for _N_accepted_real_nn_site among all the nodes
    int _global_prop_real_nn_site;  //Collects the statistics for _N_proposed_real_nn_site among all the nodes
    int _global_acc_shadows_nn_site;  //Collects the statistics for _N_accepted_shadows_nn_site among all the nodes
    int _global_prop_shadows_nn_site;  //Collects the statistics for _N_proposed_shadows_nn_site among all the nodes
    int _global_acc_global_ket_flip;  //Collects the statistics for _N_accepted_global_ket_flip among all the nodes
    int _global_prop_global_ket_flip;  //Collects the statistics for _N_proposed_global_ket_flip among all the nodes
    int _global_acc_global_bra_flip;  //Collects the statistics for _N_accepted_global_bra_flip among all the nodes
    int _global_prop_global_bra_flip;  //Collects the statistics for _N_proposed_global_bra_flip among all the nodes

    //Monte Carlo storage variables along the Markov chains
    field <cx_rowvec> _Connections;  //Non-zero matrix elements (i.e. the connections) of the observable operators
    field <field <Mat <int>>> _StatePrime;  //List of configuration |𝒮'⟩ associated to each observables connections
    mat _instReweight;  //Measured values of the 𝐑𝐞𝐰𝐞𝐢𝐠𝐡𝐭𝐢𝐧𝐠 ratio ingredients along the MCMC
    cx_mat _instObs_ket;  //Measured values of quantum observables on the configuration |𝒗 𝒉⟩  along the MCMC
    cx_mat _instObs_bra;  //Measured values of quantum observables on the configuration |𝒗 𝒉ˈ⟩ along the MCMC
    rowvec _instSquareMag;  //Measured values of the square magnetization on the configuration |𝒗⟩ along the MCMC
    mat _instSzSzCorr;  //Measured values of spin-spin correlation along the quantization axis on the configuration |𝒗⟩ along the MCMC
    cx_mat _instO_ket;  //Measured values of the local operators 𝓞(𝒗,𝒉) along the MCMC
    cx_mat _instO_bra;  //Measured values of the local operators 𝓞(𝒗,𝒉ˈ) along the MCMC

    //Simulation options variables
    bool _if_shadow;  //Chooses the 𝓈ℎ𝒶𝒹ℴ𝓌 or the 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 version of the 𝐭𝐕𝐌𝐂 algorithm
    bool _if_phi;  //Chooses whether to consider the global multiplicative variational phase in the vqs
    bool _if_shadow_off;  //Chooses to shut down the auxiliary variables in a 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏
    bool _if_vmc;  //Chooses to make a single simple 𝐕𝐌𝐂 without parameters optimization
    bool _if_imaginary_time;  //Chooses imaginary-time dinamics, i.e. ground-state properties with 𝛕 = 𝑖𝐭
    bool _if_real_time;  //Chooses real-time dynamics
    bool _if_extra_shadow_sum;  //Increases the sampling of |𝒉⟩ and ⟨𝒉ˈ| during the single 𝐌𝐂 measure
    bool _if_restart_from_config;  //Chooses to initialize the initial point of the MCMC from a previously optimized 𝓇ℯ𝒶𝑙 configuration |𝒗⟩

    //Options on the measurement of quantum properties
    bool _if_measure_ENERGY;  //Chooses whether to calculate the system energy at each time
    bool _if_measure_BLOCK_ENERGY;  //Chooses whether to calculate all energy details throughout the entire MCMC
    bool _if_measure_NON_DIAGONAL_OBS;  //Chooses whether to calculate non-diagonal operators at each time
    bool _if_measure_BLOCK_NON_DIAGONAL_OBS;  //Chooses whether to calculate all non-diagonal operators details throughout the entire MCMC
    bool _if_measure_DIAGONAL_OBS;  //Chooses whether to calculate diagonal operators at each time
    bool _if_measure_BLOCK_DIAGONAL_OBS;  //Chooses whether to calculate all diagonal operators details throughout the entire MCMC

    //Simulation parameters of the single 𝐕𝐌𝐂 step
    int _N_sweeps;  //Number of 𝐌𝐂 sweeps (i.e. #𝐌𝐂-steps in the single 𝐭𝐕𝐌𝐂 step)
    int _N_blks;  //Number of blocks to properly estimate uncertainties
    int _N_eq;  //Number of 𝐌𝐂 equilibration sweeps (i.e. 𝐌𝐂-steps) to do at the beginning of the single 𝐭𝐕𝐌𝐂 step
    int _M;  //Number of spin-flip moves to perform in the single sweep
    int _N_flips;  //Number of random spin-flips in each spin-flip move
    int _N_extra;  //Number of extra 𝐌𝐂-steps involving only the 𝓈ℎ𝒶𝒹ℴ𝓌 sampling
    int _N_blks_extra;  //Number of blocks in the extra 𝓈ℎ𝒶𝒹ℴ𝓌 sampling
    double _p_equal_site;  //Probability for the equal site 𝐌𝐂 move
    double _p_real_nn;  //Probability for the 𝓇ℯ𝒶𝑙 nearest-neighbors 𝐌𝐂 move
    double _p_shadow_nn;  //Probability for the 𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors 𝐌𝐂 move
    double _p_global_ket_flip;  //Probability for the global flip 𝐌𝐂 move on the 𝓈ℎ𝒶𝒹ℴ𝓌 ket
    double _p_global_bra_flip;  //Probability for the global flip 𝐌𝐂 move on the 𝓈ℎ𝒶𝒹ℴ𝓌 bra

    //𝐭𝐕𝐌𝐂 equations of motion regularization method options
    bool _if_QGT_REG;  //Chooses to regularize the Quantum Geometric Tensor
    int _regularization_method;  //Chooses how to regularize the Quantum Geometric Tensor
    double _eps;  //The value of the Quantum Geometric Tensor bias ε
    double _eps1;  //The value of the external cut-off in the SVD regularization
    double _eps2;  //The value of the internal cut-off in the SVD regularization
    double _lambda;  //The value of the Moore-Penrose pseudo-inverse threshold 𝝀
    double _lambda0;  //The value of 𝝀_𝟢 in the decaying diagonal regularization method
    double _lambda_min;  //The value of 𝝀ₘᵢₙ in the decaying diagonal regularization method
    double _b;  //The value of 𝑏 in the decaying diagonal regularization method
    vec _s;  //Set of the Quantum Geometric Tensor singular values
    vec _s_reg; //Set of the Quantum Geometric Tensor regularized singular values
    vec _s_inv; //Set of the Quantum Geometric Tensor regularized inverse singular values

    //𝐭𝐕𝐌𝐂 variables
    double _delta;  //The value of the integration step δₜ
    vec _cosII;  //The block averages of the non-zero reweighting ratio part ⟨cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]⟩ⱼᵇˡᵏ
    vec _sinII;  //The block averages of the (theoretically)-zero reweighting ratio part ⟨sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]⟩ⱼᵇˡᵏ
    field <cx_vec> _Observables;  //The block averages of the quantum non-diagonal observables computed along the MCMC ⟨𝒪⟩ⱼᵇˡᵏ
    vec _squareMag;  //The block averages of the square magnetization computed along the MCMC ⟨(𝗠ᶻ)^2⟩ⱼᵇˡᵏ
    mat _SzSzCorr;  //The block averages of the spin-spin correlation along the quantization axis as a function of the distance computed along the MCMC ⟨𝗖ⱼₖ(𝙧)⟩ⱼᵇˡᵏ
    field <cx_vec> _O;  //The block averages of the local operators computed along the MCMC ⟨𝓞ₖ⟩ⱼᵇˡᵏ, for k = 𝟣,…,nᵃˡᵖʰᵃ
    vec _global_cosII;  //Collects the statistics for _cosII among all the nodes
    vec _global_sinII;  //Collects the statistics for _sinII among all the nodes
    field <cx_vec> _global_Observables;  //Collects the statistics for _Observables among all the nodes
    vec _global_Mz2;  //Collects the statistics for _squareMag among all the nodes
    mat _global_Cz_of_r;  //Collects the statistics for _SzSzCorr among all the nodes
    cx_vec _mean_O;  // ⟨⟨𝓞ₖ⟩ᵇˡᵏ⟩
    cx_vec _mean_O_star;  // ⟨⟨𝓞⋆ₖ⟩ᵇˡᵏ⟩
    vec _mean_O_angled;  // ⟨≪𝓞≫ᵇˡᵏ⟩
    vec _mean_O_square;  // ⟨⌈𝓞⌋ᵇˡᵏ⟩
    cx_double _E;  // The standard stochastic average of ⟨Ĥ⟩ (without block averaging)
    cx_mat _Q;  //The Quantum Geometric Tensor ℚ
    cx_vec _F;  //The energy Gradient 𝔽 acting on 𝛂

    //Print options and related files
    bool _if_write_MOVE_STATISTICS;  //Chooses whether to write the acceptance statistics at the end of each 𝐭𝐕𝐌𝐂 step
    bool _if_write_MCMC_CONFIG;  //Chooses whether to write the sampled |𝒮⟩ along the single MCMC
    bool _if_write_FINAL_CONFIG;  //Chooses whether to write the optimized configuration |𝒮⟩ at the end of each 𝐭𝐕𝐌𝐂 step
    bool _if_write_ENERGY_ALL;  //Chooses whether to write all the the quantities that contribute to the calculation of the energy or only its 𝐌𝐂 estimate
    bool _if_write_OPT_VQS;  //Chooses whether to write the optimized set 𝓥ᵒᵖᵗ for the 𝒜𝓃𝓈𝒶𝓉𝓏 at the end of the 𝐭𝐕𝐌𝐂 dynamics
    bool _if_write_VQS_EVOLUTION;  //Chooses whether to write the set of optimized 𝓥 for the 𝒜𝓃𝓈𝒶𝓉𝓏 after each 𝐭𝐕𝐌𝐂 step
    bool _if_write_QGT;  //Chooses whether to write the Quantum Geometric Tensor at each 𝐭𝐕𝐌𝐂 step
    bool _if_write_QGT_CONDITION_NUMBER;  //Chooses whether to write the condition number of the Quantum Geometric Tensor at each 𝐭𝐕𝐌𝐂 step
    bool _if_write_QGT_EIGENVALUES;  //Chooses whether to write the Quantum Geometric Tensor eigenvalues at each 𝐭𝐕𝐌𝐂 step
    std::ofstream _file_MOVE_STATISTICS;
    std::ofstream _file_MCMC_CONFIG;
    std::ofstream _file_FINAL_CONFIG;
    std::ofstream _file_ENERGY;
    std::ofstream _file_SIGMAX;
    std::ofstream _file_MZ2;
    std::ofstream _file_SZSZ_CORR;
    std::ofstream _file_BLOCK_ENERGY;
    std::ofstream _file_BLOCK_SIGMAX;
    std::ofstream _file_BLOCK_MZ2;
    std::ofstream _file_BLOCK_SZSZ_CORR;
    std::ofstream _file_OPT_VQS;
    std::ofstream _file_VQS_EVOLUTION;
    std::ofstream _file_QGT;
    std::ofstream _file_QGT_CONDITION_NUMBER;
    std::ofstream _file_QGT_EIGENVALUES;

  public:

    //Constructor and Destructor
    VMC_Sampler(WaveFunction&, SpinHamiltonian&, int);
    ~VMC_Sampler() {};

    //Access functions
    WaveFunction& vqs() const {return _vqs;}  //Returns the reference to the 𝒜𝓃𝓈𝒶𝓉𝓏
    SpinHamiltonian& H() const {return _H;}  //Returns the reference to the spin Hamiltonian
    int n_spin() const {return _L;}  //Returns the number of quantum degrees of freedom
    int n_shadows() const {return _n_shadows;}  //Returns the number of auxiliary degrees of freedom
    cx_double i() const {return _i;}  //Returns the imaginary unit 𝑖
    mat I() const {return _I;}  //Returns the identity matrix 𝟙
    Mat <int> real_configuration() const {return _configuration;}  //Returns the sampled 𝓇ℯ𝒶𝑙 configuration |𝒗⟩
    Mat <int> shadow_ket() const {return _shadow_ket;}  //Returns the sampled ket configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables |𝒉⟩
    Mat <int> shadow_bra() const {return _shadow_bra;}  //Returns the sampled bra configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ⟨𝒉ˈ|
    Mat <int> new_real_config() const {return _flipped_site;}  //Returns the new sampled 𝓇ℯ𝒶𝑙 configuration |𝒗ⁿᵉʷ⟩
    Mat <int> new_shadow_ket() const {return _flipped_ket_site;}  //Returns the new sampled ket configuration |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra() const {return _flipped_bra_site;}  //Returns the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|
    void print_configuration() const;  //Prints on standard output the current sampled system configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
    field <cx_rowvec> get_connections() const {return _Connections;}  //Returns the list of connections
    field <field <Mat <int>>> all_state_prime() const {return _StatePrime;}  //Returns all the configuration |𝒮'⟩ connected to the current sampled configuration |𝒮⟩
    cx_mat InstObs_ket() const {return _instObs_ket;}  //Returns all the measured values of 𝒪ˡᵒᶜ(𝒗,𝒉) after a single 𝐭𝐕𝐌𝐂 run
    cx_mat InstObs_bra() const {return _instObs_bra;}  //Returns all the measured values of 𝒪ˡᵒᶜ(𝒗,𝒉') after a single 𝐭𝐕𝐌𝐂 run
    cx_mat InstO_ket() const {return _instO_ket;}  //Returns all the measured local operators 𝓞(𝒗,𝒉) after a single 𝐭𝐕𝐌𝐂 run
    cx_mat InstO_bra() const {return _instO_bra;}  //Returns all the measured local operators 𝓞(𝒗,𝒉') after a single 𝐭𝐕𝐌𝐂 run
    mat InstNorm() const {return _instReweight;}  //Returns all the measured values of 𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] and 𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')] after a single 𝐭𝐕𝐌𝐂 run
    double time_step() const {return _delta;}  //Returns the integration step parameter δₜ used in the dynamics solver
    vec cos() const {return _global_cosII;}  //Returns the collected statistics among the nodes for 𝑐𝑜𝑠[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]
    vec sin() const {return _global_sinII;}  //Returns the collected statistics among the nodes for 𝑠𝑖𝑛[ℐ(𝑣,𝒽)-ℐ(𝑣,𝒽')]
    field <cx_vec> Observables() const {return _global_Observables;}  //Returns the collected statistics among the nodes for the non-diagonal observables
    cx_mat QGT() const {return _Q;}  //Returns the 𝐌𝐂 estimate of the QGT
    cx_vec F() const {return _F;}  //Returns the 𝐌𝐂 estimate of the energy gradient
    cx_vec O() const {return _mean_O;}  //Returns the 𝐌𝐂 estimate of the local operators for 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏
    cx_vec O_star() const {return _mean_O_star;}  //Returns the 𝐌𝐂 estimate of the conjugate of local operators for 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏
    vec _O_angled() const {return _mean_O_angled;}  //Returns the 𝐌𝐂 estimate of the vector of ≪𝓞ₖ≫
    vec _O_square() const {return _mean_O_square;}  //Returns the 𝐌𝐂 estimate of the vector of ⌈𝓞ₖ⌋
    cx_double E() const {return _E;}  //Returns the 𝐌𝐂 estimate of the energy ⟨Ĥ⟩

    //Initialization functions
    void Init_Config(const Mat <int>& initial_real=Mat <int>(),  //Initializes the quantum configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩
                     const Mat <int>& initial_ket=Mat <int>(),
                     const Mat <int>& initial_bra=Mat <int>(),
                     bool zeroMag=true);
    void ShutDownShadows() {_if_shadow_off = true;}  //Shuts down the 𝓈ℎ𝒶𝒹ℴ𝓌 variables
    void setImagTimeDyn(double delta=0.01);  //Chooses the imaginary-time 𝐭𝐕𝐌𝐂 algorithm
    void setRealTimeDyn(double delta=0.01);  //Chooses the real-time 𝐭𝐕𝐌𝐂 algorithm
    void choose_regularization_method(int, double, double);  //Chooses whether to regularize the Quantum Geometric Tensor
    void setExtraShadowSum(int, int);  //Chooses to make the MC observables less noisy
    void setRestartFromConfig() {_if_restart_from_config = true;}  //Chooses the restart option at the beginning of the MCMC
    void setStepParameters(int, int, int, int, int, double, double, double, double, double, int);  //Sets the 𝐌𝐂 parameters for the single 𝐭𝐕𝐌𝐂 step

    //Print options functions
    void setFile_Move_Statistics(std::string, int);
    void setFile_MCMC_Config(std::string, int);
    void setFile_final_Config(std::string, MPI_Comm, int only_one_rank=1);
    void setFile_Energy(std::string, int, int);
    void setFile_non_Diagonal_Obs(std::string, int);
    void setFile_Diagonal_Obs(std::string, int);
    void setFile_block_Energy(std::string, int);
    void setFile_block_non_Diagonal_Obs(std::string, int);
    void setFile_block_Diagonal_Obs(std::string, int);
    void setFile_opt_VQS(std::string, int);
    void setFile_VQS_evolution(std::string, int);
    void setFile_QGT(std::string, int);
    void setFile_QGT_condition_number(std::string, int);
    void setFile_QGT_eigenvalues(std::string, int);
    void write_Move_Statistics(int, MPI_Comm);
    void write_MCMC_Config(int, int);
    void write_final_Config(int, int, int only_one_rank=1);
    void write_opt_VQS(int);
    void write_VQS_evolution(int, int);
    void write_QGT(int);
    void write_QGT_condition_number(int);
    void write_QGT_eigenvalues(int);
    void CloseFile(int, int only_one_rank=1);
    void Finalize(int, int only_one_rank=1);

    //Measurement functions
    void Reset();
    vec average_in_blocks(const rowvec&) const;
    cx_vec average_in_blocks(const cx_rowvec&) const;
    vec Shadow_average_in_blocks(const cx_rowvec&, const cx_rowvec&) const;
    vec Shadow_angled_average_in_blocks(const cx_rowvec&, const cx_rowvec&) const;
    vec Shadow_square_average_in_blocks(const cx_rowvec&, const cx_rowvec&) const;
    void compute_Reweighting_ratio(MPI_Comm);
    vec compute_errorbar(const vec&) const;
    cx_vec compute_errorbar(const cx_vec&) const;
    vec compute_progressive_averages(const vec&) const;
    cx_vec compute_progressive_averages(const cx_vec&) const;
    void compute_Quantum_observables(MPI_Comm);
    void compute_O();
    void compute_QGTandGrad(MPI_Comm, int);
    void is_asymmetric(const mat&) const;  //Check the anti-symmetric properties of an Armadillo matrix
    void QGT_Check(int);  //Checks symmetry properties of the Quantum Geometric Tensor
    cx_mat reg_SVD_inverse(const cx_mat&);  //Computes the inverse of a complex matrix by regularizing the set of singular (eigen)-values
    mat reg_SVD_inverse(const mat&);  //Computes the inverse of a real matrix by regularizing the set of singular (eigen)-values
    void Measure();  //Measurement of the istantaneous observables along a single 𝐭𝐕𝐌𝐂 run
    void Estimate(MPI_Comm, int);  //𝐌𝐂 estimates of the quantum observable averages
    void write_Quantum_properties(int, int);  //Write on appropriate files all the required system quantum properties

    //Monte Carlo moves
    bool RandFlips_real(Mat <int>&, int);  //Decides how to make a single bunch_of_spin-flip move for the 𝓇ℯ𝒶𝑙 variables only
    bool RandFlips_shadows(Mat <int>&, int);  //Decides how to make a single bunch_of_spin-flip move for the 𝓈ℎ𝒶𝒹ℴ𝓌 variables (ket or bra only)
    bool RandFlips_real_nn_site(Mat <int>&, int);  //Decides how to make a single bunch_of_spin-flip move on two 𝓇ℯ𝒶𝑙 nearest-neighbors lattice site
    bool RandFlips_shadows_nn_site(Mat <int>&, Mat <int>&, int);  //Decides how to make a single bunch_of_spin-flip move on two 𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors lattice site
    void Move_real(int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
    void Move_ket(int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
    void Move_bra(int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
    void Move_equal_site(int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with equal-site spin-flips
    void Move_real_nn_site(int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ with nearest-neighbors-site spin-flips
    void Move_shadows_nn_site(int Nflips=1);  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ with nearest-neighbors-site spin-flips
    void Move_global_ket_flip();  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ where the ket configuration has been totally flipped
    void Move_global_bra_flip();  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ where the bra configuration has been totally flipped
    void Move();  //Samples a new system configuration

    //Sampling functions
    void Make_Sweep();  //Adds a point in the Monte Carlo Markov Chain
    void Reset_Moves_Statistics();  //Resets the 𝐌𝐂 moves statistics variables
    void tVMC_Step(MPI_Comm, int);  //Performs a single 𝐭𝐕𝐌𝐂 step

    //ODE Integrators
    void Euler(MPI_Comm, int);  //Updates the variational parameters with the Euler integration method
    void Heun(MPI_Comm, int);  //Updates the variational parameters with the Heun integration method
    void RK4(MPI_Comm, int);  //Updates the variational parameters with the fourth order Runge Kutta method

};


/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
VMC_Sampler :: VMC_Sampler(WaveFunction& wave, SpinHamiltonian& hamiltonian, int rank)
             : _vqs(wave), _H(hamiltonian), _L(wave.n_real()), _i(_H.i()),
               _I(eye(_vqs.n_alpha(), _vqs.n_alpha())), _n_shadows(wave.n_real() * wave.shadow_density()) {

  //Information
  if(rank == 0){

    std::cout << "#Define the 𝐕𝐌𝐂 sampler of the variational quantum state |Ψ(𝜙, 𝛂)⟩." << std::endl;
    std::cout << " The sampler is defined on a " << _vqs.type_of_ansatz() << " quantum state designed for lattice quantum systems." << std::endl;

  }

  /*############################################################*/
  //  Creates and initializes the Random Number Generator
  //  Each process involved in the parallelization of
  //  the executable code reads a different pair of
  //  numbers from the Primes_*.in file, according to its rank.
  /*############################################################*/
  if(rank == 0) std::cout << " Create and initialize the random number generator." << std::endl;
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open())
    for(int p = 0; p <= rank; p++) Primes >> p1 >> p2;
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
  if(_vqs.if_phi_neq_zero()) _if_phi = true;
  else _if_phi = false;
  _if_shadow_off = false;
  _if_vmc = true;  //Default algorithm → simple 𝐕𝐌𝐂
  _if_imaginary_time = false;
  _if_real_time = false;
  _if_QGT_REG = false;
  _if_extra_shadow_sum = false;
  _if_restart_from_config = false;

  //Sets the output file options
  _if_write_MOVE_STATISTICS = false;
  _if_write_MCMC_CONFIG = false;
  _if_write_FINAL_CONFIG = false;
  _if_write_ENERGY_ALL = false;
  _if_write_OPT_VQS = false;
  _if_write_VQS_EVOLUTION = false;
  _if_write_QGT = false;
  _if_write_QGT_CONDITION_NUMBER = false;
  _if_write_QGT_EIGENVALUES = false;
  _if_measure_ENERGY = false;
  _if_measure_BLOCK_ENERGY = false;
  _if_measure_NON_DIAGONAL_OBS = false;
  _if_measure_DIAGONAL_OBS = false;
  _if_measure_BLOCK_NON_DIAGONAL_OBS = false;
  _if_measure_BLOCK_DIAGONAL_OBS = false;

  //Data-members initialization
  _N_accepted_real = 0;
  _N_proposed_real = 0;
  _N_accepted_ket = 0;
  _N_proposed_ket = 0;
  _N_accepted_bra = 0;
  _N_proposed_bra = 0;
  _N_accepted_equal_site = 0;
  _N_proposed_equal_site = 0;
  _N_accepted_real_nn_site = 0;
  _N_proposed_real_nn_site = 0;
  _N_accepted_shadows_nn_site = 0;
  _N_proposed_shadows_nn_site = 0;
  _N_accepted_global_ket_flip = 0;
  _N_proposed_global_ket_flip = 0;
  _N_accepted_global_bra_flip = 0;
  _N_proposed_global_bra_flip = 0;
  _N_extra = 0;
  _N_blks_extra = 0;

  if(rank == 0) std::cout << " 𝐕𝐌𝐂 sampler correctly initialized." << std::endl;

}


void VMC_Sampler :: print_configuration() const {  //Helpful in debugging

  std::cout << "\n=====================================" << std::endl;
  std::cout << "Current configuration |𝒮⟩ = |𝒗 𝒉 𝒉ˈ⟩" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "|𝒗⟩      = ";
  _configuration.print();
  std::cout << "|𝒉⟩      = ";
  _shadow_ket.print();
  std::cout << "⟨𝒉ˈ|     = ";
  _shadow_bra.print();

}


void VMC_Sampler :: Init_Config(const Mat <int>& initial_real, const Mat <int>& initial_ket, const Mat <int>& initial_bra, bool zeroMag) {

  /*##############################################################################################*/
  //  Initializes the starting point of the MCMC, using the computational basis of σ̂ᶻ eigenstates
  //
  //        σ̂ᶻ|↑⟩ = +|↑⟩
  //        σ̂ᶻ|↓⟩ = -|↓⟩.
  //
  //  We give the possibility to randomly choose spin up or down for each lattice site
  //  by using the conditional ternary operator
  //
  //        condition ? result1 : result2
  //
  //  or to initialize the configuration by providing an acceptable 𝐢𝐧𝐢𝐭𝐢𝐚𝐥_* for the variables.
  //  If the boolean data-member 𝐢𝐟_𝒔𝒉𝒂𝒅𝒐𝒘_𝐨𝐟𝐟 is true, the 𝓈ℎ𝒶𝒹ℴ𝓌 variables are all initialized
  //  and fixed to zero, i.e. they are turned off in order to make the 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 a simple
  //  𝒜𝓃𝓈𝒶𝓉𝓏 deprived of the auxiliary variables.
  //  Beware that this is not equivalent to knowing how to analytically integrate the 𝓈ℎ𝒶𝒹ℴ𝓌
  //  variables!
  //  If 𝐳𝐞𝐫𝐨𝐌𝐚𝐠 is true the initial physical configuration |𝒗⟩ is prepared with
  //  zero total magnetization.
  //
  //  So, this function initializes the generic configuration to sample along the Markov Chain
  //
  //        |𝒮⟩ = |𝒗, 𝐡, 𝐡ˈ⟩.
  //
  //  As concerns the configuration of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables, we do not make any request with
  //  respect to its magnetization, being non-physical variables.
  /*##############################################################################################*/

  //Initializes the configuration depending on |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    if(!_if_restart_from_config){  //Restarts from a random configuration |𝒮⟩

      _configuration.set_size(1, _L);
      if(_if_shadow){

        _shadow_ket.set_size(1, _n_shadows);
        _shadow_bra.set_size(1, _n_shadows);

      }

    }
    else{  //Restarts from a previously sampled configuration |𝒮⟩

      _configuration = initial_real;
      if(_if_shadow){

        if(initial_ket.is_empty()) _shadow_ket.set_size(1, _n_shadows);
        else _shadow_ket = initial_ket;
        if(initial_bra.is_empty()) _shadow_bra.set_size(1, _n_shadows);
        else _shadow_bra = initial_bra;

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

    for(int j_row = 0; j_row < _configuration.n_rows; j_row++){

      for(int j_col = 0; j_col < _configuration.n_cols; j_col++)
        _configuration.at(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

    }
    //Performs a check on the magnetization
    if(zeroMag){  //Default case

      if(!_L % 2){

        std::cerr << " ##SizeError: Cannot initialize a random spin configuration with zero magnetization for an odd number of spins." << std::endl;
        std::cerr << "   Failed to initialize the starting point of the Markov Chain." << std::endl;
        std::abort();

      }
      int tempMag = 1;
      while(tempMag != 0){

        tempMag = 0;
        for(int j_row = 0; j_row < _configuration.n_rows; j_row++){

          for(int j_col = 0; j_col < _configuration.n_cols; j_col++)
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

    if(_if_shadow_off){

      _shadow_ket.fill(0);
      _shadow_bra.fill(0);

    }  //Shuts down the auxiliary variables
    else{

      if(initial_ket.is_empty()){

        //Randomly chooses spin up or spin down
        for(int j_row = 0; j_row < _shadow_ket.n_rows; j_row++){

          for(int j_col = 0; j_col < _shadow_ket.n_cols; j_col++)
            _shadow_ket.at(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

        }

      }
      if(initial_bra.is_empty()){

        //Randomly chooses spin up or spin down
        for(int j_row = 0; j_row < _shadow_ket.n_rows; j_row++){

          for(int j_col = 0; j_col < _shadow_ket.n_cols; j_col++)
            _shadow_bra.at(j_row, j_col) = (_rnd.Rannyu() < 0.5) ? (-1) : (+1);

        }

      }

    }

  }

  //Initializes the variational quantum state
  _vqs.Init_on_Config(_configuration);

}


void VMC_Sampler :: setImagTimeDyn(double delta) {

  /*#############################################################*/
  //  Allows to update the variational parameters by integration
  //  (with an ODE integrator) of the equation of motion in
  //  imaginary time
  //
  //        𝒕 → 𝝉 = 𝑖𝒕
  //
  //  and using an integration step parameter δₜ = 𝒅𝒆𝒍𝒕𝒂.
  /*#############################################################*/

  _if_vmc = false;
  _if_imaginary_time = true;
  _if_real_time = false;
  _delta = delta;

}


void VMC_Sampler :: setRealTimeDyn(double delta) {

  /*################################################################*/
  //  Allows to update the variational parameters by integration
  //  (with an ODE integrator) of the equation of motion in
  //  real time t and using an integration step parameter δₜ = 𝒅𝒆𝒍𝒕𝒂.
  /*################################################################*/

  _if_vmc = false;
  _if_imaginary_time = false;
  _if_real_time = true;
  _delta = delta;

}


void VMC_Sampler :: choose_regularization_method(int method_flag, double control_value_1, double control_value_2) {

  /*#################################################################################*/
  //  Chooses whether or not to regularize the 𝐭𝐕𝐌𝐂 equations of motion
  //  (which are highly non-linear in the variational parameters), and decides
  //  which regularization method to be used. The regularization flags are
  //  as follows:
  //
  //        →  𝑵𝒐 𝒓𝒆𝒈𝒖𝒍𝒂𝒓𝒊𝒛𝒂𝒕𝒊𝒐𝒏 (𝟬)
  //        →  𝑫𝒊𝒂𝒈𝒐𝒏𝒂𝒍 𝒓𝒆𝒈𝒖𝒍𝒂𝒓𝒊𝒛𝒂𝒕𝒊𝒐𝒏 (𝟭)
  //             Adds a small bias to the Quantum Geometric Tensor diagonal elements
  //                          ℚ → ℚ + 𝜀•𝟙  (𝓈ℎ𝒶𝒹ℴ𝓌)
  //                          𝕊 → 𝕊 + 𝜀•𝟙  (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //             Here the control parameter is the value of 𝜀;
  //        →  𝑴𝒐𝒐𝒓𝒆-𝑷𝒆𝒏𝒓𝒐𝒔𝒆 𝒑𝒔𝒆𝒖𝒅𝒐-𝒊𝒏𝒗𝒆𝒓𝒔𝒆 (𝟮)
  //             Uses the Moore-Penrose decomposition to find the
  //             inverse of the Quantum Geometric Tensor. Here it can be
  //             imposed a control tolerance parameter, 𝝀: any singular values
  //             less than 𝝀 are treated as zero;
  //        →  𝑫𝒆𝒄𝒂𝒚𝒊𝒏𝒈 𝒅𝒊𝒂𝒈𝒐𝒏𝒂𝒍 𝒓𝒆𝒈𝒖𝒍𝒂𝒓𝒊𝒛𝒂𝒕𝒊𝒐𝒏 (𝟯)
  //             Explicitly regularizes the Quantum Geometric Tensor with a decaying
  //             parameter
  //                          ℚ → ℚ + 𝝀(𝑝)•𝟙  (𝓈ℎ𝒶𝒹ℴ𝓌)
  //                          𝕊 → 𝕊 + 𝝀(𝑝)•𝟙  (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //             where 𝑝 identifies the 𝐭𝐕𝐌𝐂 step and the control parameter is
  //             choosen as 𝝀(𝑝) = 𝑚𝑎𝑥(𝝀_𝟢 • 𝑏ᵖ, 𝝀ₘᵢₙ) with 𝝀_𝟢 = 𝟣𝟢𝟢, 𝑏 = 𝟢.𝟫 and
  //             𝝀ₘᵢₙ = 𝟣𝟢^{-𝟦};
  //        →  𝑪𝒖𝒕-𝒐𝒇𝒇 𝑺𝑽𝑫 𝒓𝒆𝒈𝒖𝒍𝒂𝒓𝒊𝒛𝒂𝒕𝒊𝒐𝒏 (𝟰)
  //             Sets a (signed) external cut-off on all the singular values of the Quantum Geometric
  //             Tensor such that
  //                          ε𝟤 < |s| < ε𝟣,
  //             while neglecting all the singular values in the range
  //                          -ε𝟤 < s < ε𝟤.
  /*#################################################################################*/

  if(method_flag == 0){  //No regularization

    _if_QGT_REG = false;
    _regularization_method = 0;

  }
  else if(method_flag == 1){  //Simple diagonal regularization

    _if_QGT_REG = true;
    _regularization_method = 1;
    _eps = control_value_1;

  }
  else if(method_flag == 2){  //Moore-Penrose pseudo-inverse

    _if_QGT_REG = true;
    _regularization_method = 2;
    _lambda = control_value_1;

  }
  else if(method_flag == 3){  //Decaying diagonal regularization

    _if_QGT_REG = true;
    _regularization_method = 3;
    _lambda0 = 100;
    _lambda_min = control_value_1;
    _b = 0.9;

  }
  else if(method_flag == 4){  //Cut-off SVD regularization

    _if_QGT_REG = true;
    _regularization_method = 4;
    _eps1 = control_value_1;
    _eps2 = control_value_2;

  }
  else{

    std::cerr << " ##ValueError: method of regularization not available." << std::endl;
    std::cerr << "   Failed to choose regularization method for the Quantum Geometric Tensor." << std::endl;
    std::abort();

  }

}


void VMC_Sampler :: setExtraShadowSum(int Nextra, int Nblks) {

  _if_extra_shadow_sum = true;
  _N_extra = Nextra;
  _N_blks_extra = Nblks;

}


void VMC_Sampler :: setStepParameters(int Nsweeps, int Nblks, int Neq, int M, int Nflips,
                                      double p_equal_site, double p_real_nn, double p_shadow_nn,
                                      double p_global_ket_flip, double p_global_bra_flip, int rank) {

  _N_sweeps = Nsweeps;
  _N_blks = Nblks;
  _N_eq = Neq;
  _M = M;
  _N_flips = Nflips;
  _p_equal_site = p_equal_site;
  _p_real_nn = p_real_nn;
  _p_shadow_nn = p_shadow_nn;
  _p_global_ket_flip = p_global_ket_flip;
  _p_global_bra_flip = p_global_bra_flip;

  if(rank == 0){

    std::cout << " Parameters of the simulation in each nodes of the communicator:" << std::endl;
    std::cout << " \tNumber of spin sweeps in the single 𝐭𝐕𝐌𝐂 step:  " << _N_sweeps  << ";" << std::endl;
    std::cout << " \tNumber of blocks in the single 𝐭𝐕𝐌𝐂 step:  " << _N_blks << ";" << std::endl;
    std::cout << " \tEquilibration steps in the single 𝐭𝐕𝐌𝐂 step:  " << _N_eq << ";" << std::endl;
    std::cout << " \tNumber of spin-flip moves in the single 𝐌𝐂 sweep:  " << _M  << ";" << std::endl;
    std::cout << " \tNumber of spin-flips in the single spin-flip move:  " << _N_flips << ";" << std::endl;
    std::cout << " \tProbability for the equal-site 𝐌𝐂-move:  " << _p_equal_site * 100.0 << " %;" << std::endl;
    std::cout << " \tProbability for the nearest-neighbors 𝓇ℯ𝒶𝑙 𝐌𝐂-move:  " << _p_real_nn * 100.0 << " %;" << std::endl;
    std::cout << " \tProbability for the nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌 𝐌𝐂-move:  " << _p_shadow_nn * 100.0 << " %;" << std::endl;
    std::cout << " \tProbability for the global spin-flip 𝐌𝐂-move on the 𝓈ℎ𝒶𝒹ℴ𝓌 ket:  " << _p_global_ket_flip * 100.0 << " %;" << std::endl;
    std::cout << " \tProbability for the global spin-flip 𝐌𝐂-move on the 𝓈ℎ𝒶𝒹ℴ𝓌 bra:  " << _p_global_bra_flip * 100.0 << " %;" << std::endl;
    if(_if_extra_shadow_sum){

      std::cout << " \tNumber of extra 𝓈ℎ𝒶𝒹ℴ𝓌 sampling performed within each instantaneous measurement:  "  << _N_extra << ";" << std::endl;
      std::cout << " \tNumber of block for the extra 𝓈ℎ𝒶𝒹ℴ𝓌 sampling statistics:  " << _N_blks_extra << ";" << std::endl;

    }
    if(_if_QGT_REG){

      if(_regularization_method == 0)  std::cout << " \tIt was decided not to regularize the 𝐭𝐕𝐌𝐂 equations of motion;" << std::endl;
      else if(_regularization_method == 1) std::cout << " \tDiagonal QGT regularization with a control bias ε = " << _eps << ";" << std::endl;
      else if(_regularization_method == 2) std::cout << " \tMoore-Penrose pseudo-inverse QGT regularization with control tolerance 𝝀 = " << _lambda << ";" << std::endl;
      else if(_regularization_method == 3) std::cout << " \tDecaying diagonal QGT regularization with a control bias 𝝀(𝑝) = 𝑚𝑎𝑥(𝝀_𝟢 • 𝑏ᵖ, 𝝀ₘᵢₙ) with 𝝀_𝟢 = " << _lambda0 << ", 𝑏 = " << _b << " and 𝝀ₘᵢₙ = " << _lambda_min << ";" << std::endl;
      else if(_regularization_method == 4) std::cout << " \tCut-off SVD regularization with an external control bias ε𝟣 = " << _eps1 << " and an internal one ε𝟤 = " << _eps2 << ";" << std::endl;
      else{

        std::cerr << " ##ValueError: choosen regularization method not available!" << std::endl;
        std::cerr << "   Failed to set the parameters for the single simulation step." << std::endl;
        std::abort();

      }

    }
    std::cout << " \tIntegration step parameter:  " << _delta << "." << std::endl << std::endl;

  }

}


void VMC_Sampler :: setFile_Move_Statistics(std::string info, int rank) {

  _if_write_MOVE_STATISTICS = true;
  if(rank == 0){

    _file_MOVE_STATISTICS.open("Move_Statistics_" + info + ".dat");
    if(!_file_MOVE_STATISTICS.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ Move_Statistics_" << info << ".dat ›› for writing the acceptance statistics at the end of the single 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the acceptance statistics of the 𝐌𝐂 moves at the end of the single 𝐭𝐕𝐌𝐂 step on file ‹‹ Move_Statistics_" << info << ".dat ››." << std::endl;

    _file_MOVE_STATISTICS << "########################################################################################################\n";
    _file_MOVE_STATISTICS << "# Column Legend\n";
    _file_MOVE_STATISTICS << "#\n";
    _file_MOVE_STATISTICS << "#   1st: the 𝐭𝐕𝐌𝐂 step identifier\n";
    _file_MOVE_STATISTICS << "#   2nd: the sampling acceptance probability (%) of |𝒗⟩\n";
    _file_MOVE_STATISTICS << "#   3rd: the sampling acceptance probability (%) of |𝒉⟩\n";
    _file_MOVE_STATISTICS << "#   4th: the sampling acceptance probability (%) of ⟨𝒉ˈ|\n";
    _file_MOVE_STATISTICS << "#   5th: the sampling acceptance probability (%) of |𝒗 𝒉 𝒉ˈ⟩ moved on equal sites\n";
    _file_MOVE_STATISTICS << "#   6th: the sampling acceptance probability (%) of |𝒗 𝒉 𝒉ˈ⟩ moved after a global flip on the ket\n";
    _file_MOVE_STATISTICS << "#   7th: the sampling acceptance probability (%) of |𝒗 𝒉 𝒉ˈ⟩ moved after a global flip on the bra\n";
    _file_MOVE_STATISTICS << "#   8th: the sampling acceptance probability (%) of |𝒗⟩ moved on nearest-neighbors sites\n";
    _file_MOVE_STATISTICS << "#   9th: the sampling acceptance probability (%) of |𝒉⟩ and ⟨𝒉ˈ| moved on nearest-neighbors sites\n";
    _file_MOVE_STATISTICS << "########################################################################################################\n";

  }

}


void VMC_Sampler :: setFile_MCMC_Config(std::string info, int rank) {  //Helpful in debugging

  _if_write_MCMC_CONFIG = true;
  if(rank == 0){

    //Creates the output directory by checking if ./CONFIG/ folder already exists
    if(!is_directory("./CONFIG") || !exists("./CONFIG")) create_directory("./CONFIG");

    _file_MCMC_CONFIG.open("./CONFIG/MCMC_config_" + info + ".dat");
    if(!_file_MCMC_CONFIG.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ CONFIG/MCMC_config_" << info << ".dat ›› for writing the sampled configurations along a single MCMC." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the sampled configurations along a single MCMC on file ‹‹ CONFIG/MCMC_config_" << info << ".dat ››." << std::endl;
    _file_MCMC_CONFIG << "####################################################\n";
    _file_MCMC_CONFIG << "# Column Legend\n";
    _file_MCMC_CONFIG << "#\n";
    _file_MCMC_CONFIG << "#   1st: the 𝐌𝐂-step identifier\n";
    _file_MCMC_CONFIG << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
    _file_MCMC_CONFIG << "####################################################\n";

  }

}


void VMC_Sampler :: setFile_final_Config(std::string info, MPI_Comm common, int only_one_rank) {

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  if(only_one_rank == 1){  //Default case

    if(rank == 0){

      _if_write_FINAL_CONFIG = true;

      //Creates the output directory by checking if CONFIG folder already exists
      if(!is_directory("./CONFIG") || !exists("./CONFIG")) create_directory("./CONFIG");

      _file_FINAL_CONFIG.open("./CONFIG/final_config_" + info + ".dat");
      if(!_file_FINAL_CONFIG.good()){

        std::cerr << " ##FileError: Cannot open the file ‹‹ CONFIG/final_config_" << info << ".dat ›› for writing the final configurations at the end of each 𝐭𝐕𝐌𝐂 step." << std::endl;
        std::abort();

      }
      else std::cout << " Saving the final configurations sampled at the end of each 𝐭𝐕𝐌𝐂 step on file ‹‹ CONFIG/final_config_" << info << ".dat ››." << std::endl;

      _file_FINAL_CONFIG << "####################################################\n";
      _file_FINAL_CONFIG << "# Column Legend\n";
      _file_FINAL_CONFIG << "#\n";
      _file_FINAL_CONFIG << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_FINAL_CONFIG << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
      _file_FINAL_CONFIG << "####################################################\n";

    }

  }
  else{  //Writes the configurations for all the nodes of the communicator; Helpful in debugging

    _if_write_FINAL_CONFIG = true;

    //Creates the output directory by checking if CONFIG folder already exists
    if(!is_directory("./CONFIG") || !exists("./CONFIG")) create_directory("./CONFIG");

    _file_FINAL_CONFIG.open("./CONFIG/final_config_" + info + "_node_" + std::to_string(rank) + ".dat");
    if(!_file_FINAL_CONFIG.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ CONFIG/final_config_" << info << "_node_" << rank << ".dat ›› for writing the final configurations at the end of each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else
      if(rank == 0) std::cout << " Saving the final configurations sampled at the end of each 𝐭𝐕𝐌𝐂 step on file ‹‹ CONFIG/final_config_" << info << "_node_*.dat ››." << std::endl;

    _file_FINAL_CONFIG << "####################################################\n";
    _file_FINAL_CONFIG << "# Column Legend\n";
    _file_FINAL_CONFIG << "#\n";
    _file_FINAL_CONFIG << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_FINAL_CONFIG << "#   2nd: the sampled quantum configuration |𝒗 𝒉 𝒉ˈ⟩\n";
    _file_FINAL_CONFIG << "####################################################\n";

  }

}


void VMC_Sampler :: setFile_Energy(std::string info, int all_option, int rank){

  _if_measure_ENERGY = true;
  _if_write_ENERGY_ALL = all_option;
  if(rank == 0){

    _file_ENERGY.open("energy_" + info + ".dat");

    if(!_file_ENERGY.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ energy_" << info << ".dat ›› for writing 𝐌𝐂 calculations of E(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving 𝐌𝐂 calculations of E(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ energy_" << info << ".dat ››." << std::endl;

    if(_if_write_ENERGY_ALL){

      _file_ENERGY << "####################################################################\n";
      _file_ENERGY << "# Column Legend\n";
      _file_ENERGY << "#\n";
      _file_ENERGY << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_ENERGY << "#   2nd:  estimate of ⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆\n";
      _file_ENERGY << "#   3rd:  error on ⟨𝒄𝒐𝒔𝑰𝑰⟩𝓆\n";
      _file_ENERGY << "#   4th:  estimate of ⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆\n";
      _file_ENERGY << "#   5rd:  error on ⟨𝒔𝒊𝒏𝑰𝑰⟩𝓆\n";
      _file_ENERGY << "#   6th:  estimate of 𝑬ᴿ(𝜙,𝛂)\n";
      _file_ENERGY << "#   7th:  error on 𝑬ᴿ(𝜙,𝛂)\n";
      _file_ENERGY << "#   8th:  estimate of 𝑬ᴵ(𝜙,𝛂)\n";
      _file_ENERGY << "#   9th:  error on 𝑬ᴵ(𝜙,𝛂)\n";
      _file_ENERGY << "#   10th: standard 𝐌𝐂 average (without reweighting) of 𝑬ᴿ(𝜙,𝛂)\n";
      _file_ENERGY << "#   11th: standard 𝐌𝐂 average (without reweighting) of 𝑬ᴵ(𝜙,𝛂)\n";
      _file_ENERGY << "####################################################################\n";

    }
    else{

      _file_ENERGY << "#######################################\n";
      _file_ENERGY << "# Column Legend\n";
      _file_ENERGY << "#\n";
      _file_ENERGY << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_ENERGY << "#   2nd:  estimate of 𝑬ᴿ(𝜙,𝛂)\n";
      _file_ENERGY << "#   3rd:  error on 𝑬ᴿ(𝜙,𝛂)\n";
      _file_ENERGY << "#   4th:  estimate of 𝑬ᴵ(𝜙,𝛂)\n";
      _file_ENERGY << "#   5th:  error on 𝑬ᴵ(𝜙,𝛂)\n";
      _file_ENERGY << "#######################################\n";

    }

  }

}


void VMC_Sampler :: setFile_non_Diagonal_Obs(std::string info, int rank) {

  _if_measure_NON_DIAGONAL_OBS = true;
  if(rank == 0){

    _file_SIGMAX.open("sigmaX_" + info + ".dat");
    if(!_file_SIGMAX.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ sigmaX_" << info << ".dat ›› for writing 𝐌𝐂 calculations of σˣ(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving 𝐌𝐂 calculations of σˣ(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ sigmaX_" << info << ".dat ››." << std::endl;

    _file_SIGMAX << "####################################\n";
    _file_SIGMAX << "# Column Legend\n";
    _file_SIGMAX << "#\n";
    _file_SIGMAX << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_SIGMAX << "#   2nd:  estimate of ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
    _file_SIGMAX << "#   3rd:  error on ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
    _file_SIGMAX << "#   4th:  estimate of ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
    _file_SIGMAX << "#   5th:  error on ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
    _file_SIGMAX << "####################################\n";

  }

}


void VMC_Sampler :: setFile_Diagonal_Obs(std::string info, int rank) {

  _if_measure_DIAGONAL_OBS = true;
  if(rank == 0){

    _file_MZ2.open("square_mag_" + info + ".dat");
    _file_SZSZ_CORR.open("Cz_of_r_" + info + ".dat");
    if(!_file_MZ2.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ square_mag_" << info << ".dat ›› for writing 𝐌𝐂 calculations of (𝗠ᶻ)^2(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving 𝐌𝐂 calculations of (𝗠ᶻ)^2(𝜙,𝛂) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ square_mag_" << info << ".dat ››." << std::endl;
    if(!_file_SZSZ_CORR.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ Cz_of_r_" << info << ".dat ›› for writing 𝐌𝐂 calculations of the 𝗖ᶻ(𝙧) correlation function after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving 𝐌𝐂 calculations of the 𝗖ᶻ(𝙧) correlation function after each 𝐭𝐕𝐌𝐂 step on file ‹‹ Cz_of_r_" << info << ".dat ››." << std::endl;

    _file_MZ2 << "###################################\n";
    _file_MZ2 << "# Column Legend\n";
    _file_MZ2 << "#\n";
    _file_MZ2 << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_MZ2 << "#   2nd:  estimate of (𝗠 ᶻ)^2(𝜙,𝛂)\n";
    _file_MZ2 << "#   3rd:  error on (𝗠 ᶻ)^2(𝜙,𝛂)\n";
    _file_MZ2 << "###################################\n";

    _file_SZSZ_CORR << "####################################\n";
    _file_SZSZ_CORR << "# Column Legend\n";
    _file_SZSZ_CORR << "#\n";
    _file_SZSZ_CORR << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_SZSZ_CORR << "#   2nd:  spin distance 𝙧 = |𝙭 - 𝙮|\n";
    _file_SZSZ_CORR << "#   3rd:  estimate of 𝗖ᶻ(𝙧)\n";
    _file_SZSZ_CORR << "#   4th:  error on 𝗖ᶻ(𝙧)\n";
    _file_SZSZ_CORR << "####################################\n";

  }

}


void VMC_Sampler :: setFile_block_Energy(std::string info, int rank){

  _if_measure_BLOCK_ENERGY = true;
  if(rank == 0){

    _file_BLOCK_ENERGY.open("block_energy_" + info + ".dat");
    if(!_file_BLOCK_ENERGY.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_energy_" << info << ".dat ›› for writing all the 𝐌𝐂 block calculations of E(𝜙,𝛂) during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving all the 𝐌𝐂 block calculations of E(𝜙,𝛂) during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_energy_" << info << ".dat ››." << std::endl;

    if(!_if_shadow){

      _file_BLOCK_ENERGY << "##########################################\n";
      _file_BLOCK_ENERGY << "# Column Legend\n";
      _file_BLOCK_ENERGY << "#\n";
      _file_BLOCK_ENERGY << "#   1st:   the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_BLOCK_ENERGY << "#   2nd:   the 𝐌𝐂-block identifier\n";
      _file_BLOCK_ENERGY << "#   3rd:   ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   4th:   error on ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   5th:   ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   6th:   error on ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   7th:   ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   8th:   error on ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   9th:   ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   10th:  error on ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "##########################################\n";

    }
    else{

      _file_BLOCK_ENERGY << "##################################################################\n";
      _file_BLOCK_ENERGY << "# Column Legend\n";
      _file_BLOCK_ENERGY << "#\n";
      _file_BLOCK_ENERGY << "#   1st:   the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_BLOCK_ENERGY << "#   2nd:   the 𝐌𝐂-block identifier\n";
      _file_BLOCK_ENERGY << "#   3rd:   ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   4th:   error on ⟨𝒄𝒐𝒔𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   5th:   ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   6th:   error on ⟨𝒔𝒊𝒏𝑰𝑰⟩ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   7th:   shadow (without reweighting) ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   8th:   shadow (without reweighting) ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   9th:   ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   10th:  error on ℜ𝓮{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   11th:  ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "#   12th:  error on ℑ𝓶{⟨Ĥ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_ENERGY << "##################################################################\n";

    }

  }

}


void VMC_Sampler :: setFile_block_non_Diagonal_Obs(std::string info, int rank) {

  _if_measure_BLOCK_NON_DIAGONAL_OBS = true;
  if(rank == 0){

    _file_BLOCK_SIGMAX.open("block_sigmaX_" + info + ".dat");
    if(!_file_BLOCK_SIGMAX.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_sigmaX_" << info << ".dat ›› for writing all the 𝐌𝐂 block calculations of σˣ(𝜙,𝛂) during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving all the 𝐌𝐂 block calculations of σˣ(𝜙,𝛂) during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_sigmaX_" << info << ".dat ››." << std::endl;

    if(!_if_shadow){

      _file_BLOCK_SIGMAX << "############################################\n";
      _file_BLOCK_SIGMAX << "# Column Legend\n";
      _file_BLOCK_SIGMAX << "#\n";
      _file_BLOCK_SIGMAX << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_BLOCK_SIGMAX << "#   2nd:  the 𝐌𝐂-block identifier\n";
      _file_BLOCK_SIGMAX << "#   3rd:  ℜ𝓮{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_SIGMAX << "#   4th:  progressive error ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
      _file_BLOCK_SIGMAX << "#   5th:  ℑ𝓶{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_SIGMAX << "#   6th:  progressive error on ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
      _file_BLOCK_SIGMAX << "############################################\n";

    }
    else{

      _file_BLOCK_SIGMAX << "####################################################################\n";
      _file_BLOCK_SIGMAX << "# Column Legend\n";
      _file_BLOCK_SIGMAX << "#\n";
      _file_BLOCK_SIGMAX << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_BLOCK_SIGMAX << "#   2nd:  the 𝐌𝐂-block identifier\n";
      _file_BLOCK_SIGMAX << "#   3rd:  shadow (without reweighting) ℜ𝓮{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_SIGMAX << "#   4th:  shadow (without reweighting) ℑ𝓶{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_SIGMAX << "#   5th:  ℜ𝓮{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_SIGMAX << "#   6th:  progressive error ℜ𝓮{𝜎ˣ}(𝜙,𝛂)\n";
      _file_BLOCK_SIGMAX << "#   7th:  ℑ𝓶{⟨𝜎̂ˣ⟩}ʲ𝓆 in block j\n";
      _file_BLOCK_SIGMAX << "#   8th:  progressive error on ℑ𝓶{𝜎ˣ}(𝜙,𝛂)\n";
      _file_BLOCK_SIGMAX << "####################################################################\n";

    }

  }

}


void VMC_Sampler :: setFile_block_Diagonal_Obs(std::string info, int rank) {

  _if_measure_BLOCK_DIAGONAL_OBS = true;
  if(rank == 0){

    _file_BLOCK_MZ2.open("block_square_mag_" + info + ".dat");
    _file_BLOCK_SZSZ_CORR.open("block_Cz_of_r_" + info + ".dat");
    if(!_file_BLOCK_MZ2.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_square_mag_" << info << ".dat ›› for writing all the 𝐌𝐂 block calculations of (𝗠ᶻ)^2 during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving all the 𝐌𝐂 block calculations of (𝗠ᶻ)^2 during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_square_mag_" << info << ".dat ››." << std::endl;
    if(!_file_BLOCK_SZSZ_CORR.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ block_Cz_of_r_" << info << ".dat ›› for writing all the 𝐌𝐂 block calculations of the 𝗖ᶻ(𝙧) correlation function during each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving all the 𝐌𝐂 block calculations of the 𝗖ᶻ(𝙧) correlation function during each 𝐭𝐕𝐌𝐂 step on file ‹‹ block_Cz_of_r_" << info << ".dat ››." << std::endl;

    _file_BLOCK_MZ2 << "###########################################\n";
    _file_BLOCK_MZ2 << "# Column Legend\n";
    _file_BLOCK_MZ2 << "#\n";
    _file_BLOCK_MZ2 << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_BLOCK_MZ2 << "#   2nd:  the 𝐌𝐂-block identifier\n";
    _file_BLOCK_MZ2 << "#   3rd:  (𝗠 ᶻ)^2ʲ𝓆 in block j\n";
    _file_BLOCK_MZ2 << "#   4th:  progressive error (𝗠 ᶻ)^2(𝜙,𝛂)\n";
    _file_BLOCK_MZ2 << "###########################################\n";

    _file_BLOCK_SZSZ_CORR << "#########################################################\n";
    _file_BLOCK_SZSZ_CORR << "# Column Legend\n";
    _file_BLOCK_SZSZ_CORR << "#\n";
    _file_BLOCK_SZSZ_CORR << "#   1st:  the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_BLOCK_SZSZ_CORR << "#   2nd:  the 𝐌𝐂-block identifier\n";
    _file_BLOCK_SZSZ_CORR << "#   3rd:  spin distance 𝙧 = |𝙭 - 𝙮|\n";
    _file_BLOCK_SZSZ_CORR << "#   4th:  ⟨𝗖ᶻ(𝙧)⟩ʲ𝓆 in block j at distance 𝙧\n";
    _file_BLOCK_SZSZ_CORR << "#   5th:  progressive error on 𝗖ᶻ(𝙧)(𝜙,𝛂) at distance 𝙧\n";
    _file_BLOCK_SZSZ_CORR << "########################################################\n";

  }

}


void VMC_Sampler :: setFile_opt_VQS(std::string info, int rank) {

  _if_write_OPT_VQS = true;
  if(rank == 0){

    _file_OPT_VQS.open("optimized_parameters_" + info + ".wf");
    if(!_file_OPT_VQS.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ optimized_" << info << ".wf ›› for writing the optimized set of variational parameters 𝓥." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the optimized set of variational parameters 𝓥 on file ‹‹ optimized_" << info << ".wf ››." << std::endl;

  }

}


void VMC_Sampler :: setFile_VQS_evolution(std::string info, int rank) {

  _if_write_VQS_EVOLUTION = true;
  if(rank == 0){

    _file_VQS_EVOLUTION.open("vqs_evolution_" + info + ".wf");
    if(!_file_VQS_EVOLUTION.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ vqs_evolution_" << info << ".wf ›› for writing the set of variational parameters 𝓥 at the end of each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the set of variational parameters 𝓥 at the end of each 𝐭𝐕𝐌𝐂 step on file ‹‹ vqs_evolution_" << info << ".wf ››." << std::endl;

    if(_vqs.type_of_ansatz() == "Shadow"){

      _file_VQS_EVOLUTION << "######################################################\n";
      _file_VQS_EVOLUTION << "# Column Legend\n";
      _file_VQS_EVOLUTION << "#\n";
      _file_VQS_EVOLUTION << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_VQS_EVOLUTION << "#   2nd: 𝒱ₖᴿ\n";
      _file_VQS_EVOLUTION << "#   3rd: 𝒱ₖᴵ\n";
      _file_VQS_EVOLUTION << "#   and so on for each variational parameters...\n";
      _file_VQS_EVOLUTION << "######################################################\n";

      _file_VQS_EVOLUTION << 0;
      _file_VQS_EVOLUTION << std::setprecision(8) << std::fixed;
      _file_VQS_EVOLUTION << "\t" << _vqs.phi().real() << "\t" << _vqs.phi().imag();
      for(int p = 0; p < _vqs.n_alpha(); p++)
        _file_VQS_EVOLUTION << "\t" << _vqs.alpha_at(p).real() << "\t" << _vqs.alpha_at(p).imag();
      _file_VQS_EVOLUTION << std::endl;

    }
    else if(_vqs.type_of_ansatz() == "Jastrow"){

      _file_VQS_EVOLUTION << "######################################################\n";
      _file_VQS_EVOLUTION << "# Column Legend\n";
      _file_VQS_EVOLUTION << "#\n";
      _file_VQS_EVOLUTION << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_VQS_EVOLUTION << "#   2nd: 𝒱ₖᴿ\n";
      _file_VQS_EVOLUTION << "#   3rd: 𝒱ₖᴵ\n";
      _file_VQS_EVOLUTION << "#   and so on for each variational parameters...\n";
      _file_VQS_EVOLUTION << "######################################################\n";

      _file_VQS_EVOLUTION << 0;
      _file_VQS_EVOLUTION << std::setprecision(8) << std::fixed;
      _file_VQS_EVOLUTION << "\t" << _vqs.phi().real() << " " << _vqs.phi().imag();
      for(int p = 0; p < _vqs.n_alpha(); p++)
        _file_VQS_EVOLUTION << " " << _vqs.alpha_at(p).real() << " " << _vqs.alpha_at(p).imag();
      _file_VQS_EVOLUTION << std::endl;

    }
    else{

      _file_VQS_EVOLUTION << "######################################################\n";
      _file_VQS_EVOLUTION << "# Column Legend\n";
      _file_VQS_EVOLUTION << "#\n";
      _file_VQS_EVOLUTION << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_VQS_EVOLUTION << "#   2nd: 𝒱ᴿ\n";
      _file_VQS_EVOLUTION << "#   3rd: 𝒱ᴵ\n";
      _file_VQS_EVOLUTION << "######################################################\n";

      _file_VQS_EVOLUTION << 0;
      _file_VQS_EVOLUTION << std::setprecision(8) << std::fixed;
      _file_VQS_EVOLUTION << "\t" << _vqs.phi().real() << "\t" << _vqs.phi().imag() << std::endl;
      for(int p = 0; p < _vqs.n_alpha(); p++)
        _file_VQS_EVOLUTION << 0 << "\t" << _vqs.alpha_at(p).real() << "\t" << _vqs.alpha_at(p).imag() << std::endl;

    }

  }

}


void VMC_Sampler :: setFile_QGT(std::string info, int rank) {  //Helpful in debugging

  _if_write_QGT = true;
  if(rank == 0){

    _file_QGT.open("qgt_" + info + ".dat");
    if(!_file_QGT.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_" << info << ".dat ›› for writing the Quantum Geometric Tensor after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the Quantum Geometric Tensor after each 𝐭𝐕𝐌𝐂 step on file ‹‹ qgt_" << info << ".dat ››." << std::endl;

    _file_QGT << "#######################################\n";
    _file_QGT << "# Column Legend\n";
    _file_QGT << "#\n";
    _file_QGT << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
    _file_QGT << "#   2nd: the Quantum Geometric Tensor\n";
    _file_QGT << "#######################################\n";

  }

}


void VMC_Sampler :: setFile_QGT_condition_number(std::string info, int rank) {  //Helpful in debugging

  _if_write_QGT_CONDITION_NUMBER = true;
  if(rank == 0){

    _file_QGT_CONDITION_NUMBER.open("qgt_condition_number_" + info + ".dat");
    if(!_file_QGT_CONDITION_NUMBER.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_condition_number_" << info << ".dat ›› for writing the Quantum Geometric Tensor condition number after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the Quantum Geometric Tensor condition number after each 𝐭𝐕𝐌𝐂 step on file ‹‹ qgt_condition_number_" << info << ".dat ››." << std::endl;

    if(_regularization_method == 0 || _regularization_method == 2 || _regularization_method == 4){

      _file_QGT_CONDITION_NUMBER << "###################################################################\n";
      _file_QGT_CONDITION_NUMBER << "# Column Legend\n";
      _file_QGT_CONDITION_NUMBER << "#\n";
      _file_QGT_CONDITION_NUMBER << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_CONDITION_NUMBER << "#   2nd: the QGT reciprocal condition number (no regularization)\n";
      _file_QGT_CONDITION_NUMBER << "#   OSS: values close to 1 suggest that the QGT well-conditioned\n";
      _file_QGT_CONDITION_NUMBER << "#        values close to 0 suggest that the QGT badly-conditioned\n";
      _file_QGT_CONDITION_NUMBER << "###################################################################\n";

    }
    else{

      _file_QGT_CONDITION_NUMBER << "####################################################################\n";
      _file_QGT_CONDITION_NUMBER << "# Column Legend\n";
      _file_QGT_CONDITION_NUMBER << "#\n";
      _file_QGT_CONDITION_NUMBER << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_CONDITION_NUMBER << "#   2nd: the QGT reciprocal condition number (no regularization)\n";
      _file_QGT_CONDITION_NUMBER << "#   3th: the QGT condition number (with regularization)\n";
      _file_QGT_CONDITION_NUMBER << "#   OSS: values close to 1 suggest that the QGT well-conditioned\n";
      _file_QGT_CONDITION_NUMBER << "#        values close to 0 suggest that the QGT badly-conditioned\n";
      _file_QGT_CONDITION_NUMBER << "####################################################################\n";

    }

  }

}


void VMC_Sampler :: setFile_QGT_eigenvalues(std::string info, int rank) {  //Helpful in debugging

  _if_write_QGT_EIGENVALUES = true;
  if(rank == 0){

    _file_QGT_EIGENVALUES.open("qgt_eigenvalues_" + info + ".dat");
    if(!_file_QGT_EIGENVALUES.good()){

      std::cerr << " ##FileError: Cannot open the file ‹‹ qgt_eigenvalues_" << info << ".dat ›› for writing the eigenvalues of the Quantum Geometric Tensor after each 𝐭𝐕𝐌𝐂 step." << std::endl;
      std::abort();

    }
    else std::cout << " Saving the Quantum Geometric Tensor eigenvalues (or singular values) after each 𝐭𝐕𝐌𝐂 step on file ‹‹ qgt_eigenvalues_" << info << ".dat ››." << std::endl;

    if(_regularization_method == 0){  //No regularization

      _file_QGT_EIGENVALUES << "######################################################################\n";
      _file_QGT_EIGENVALUES << "# Column Legend\n";
      _file_QGT_EIGENVALUES << "#\n";
      _file_QGT_EIGENVALUES << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_EIGENVALUES << "#   2nd: the first QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   3rd: the second QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   4th: the third QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   j-th: the j-th QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "######################################################################\n";

    }
    else if(_regularization_method == 1){  //Simple diagonal regularization

      _file_QGT_EIGENVALUES << "######################################################################\n";
      _file_QGT_EIGENVALUES << "# Column Legend\n";
      _file_QGT_EIGENVALUES << "#\n";
      _file_QGT_EIGENVALUES << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_EIGENVALUES << "#   2nd: the first non-regulatized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   3rd: the first regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   4th: the second non-regulatized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   5th: the second regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   j-th: the j-th non-regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   (j + 1)-th: the j-th regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "######################################################################\n";

    }
    else if(_regularization_method == 2){  //Moore-Penrose pseudo-inverse

      _file_QGT_EIGENVALUES << "######################################################################\n";
      _file_QGT_EIGENVALUES << "# Column Legend\n";
      _file_QGT_EIGENVALUES << "#\n";
      _file_QGT_EIGENVALUES << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_EIGENVALUES << "#   2nd: the first QGT (potentially singular) eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   2nd: the second QGT (potentially singular) eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   4th: the third QGT (potentially singular) eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   j-th: the j-th QGT (potentially singular) eigenvalues\n";
      _file_QGT_EIGENVALUES << "######################################################################\n";

    }
    else if(_regularization_method == 3){  //Decaying diagonal regularization

      _file_QGT_EIGENVALUES << "######################################################################\n";
      _file_QGT_EIGENVALUES << "# Column Legend\n";
      _file_QGT_EIGENVALUES << "#\n";
      _file_QGT_EIGENVALUES << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_EIGENVALUES << "#   2nd: the first non-regulatized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   3rd: the first regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   4th: the second non-regulatized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   5th: the second regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   j-th: the j-th non-regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "#   (j + 1)-th: the j-th regularized QGT eigenvalues\n";
      _file_QGT_EIGENVALUES << "######################################################################\n";

    }
    else if(_regularization_method == 4){  //Cut-off SVD regularization

      _file_QGT_EIGENVALUES << "######################################################################\n";
      _file_QGT_EIGENVALUES << "# Column Legend\n";
      _file_QGT_EIGENVALUES << "#\n";
      _file_QGT_EIGENVALUES << "#   1st: the 𝐭𝐕𝐌𝐂-step identifier\n";
      _file_QGT_EIGENVALUES << "#   2nd: the first non-regulatized QGT singular value\n";
      _file_QGT_EIGENVALUES << "#   3rd: the first regularized QGT singular value\n";
      _file_QGT_EIGENVALUES << "#   4th: the first regularized QGT inverse singular value\n";
      _file_QGT_EIGENVALUES << "#   5th: the second non-regulatized QGT singular value\n";
      _file_QGT_EIGENVALUES << "#   6th: the second regularized QGT singular value\n";
      _file_QGT_EIGENVALUES << "#   7th: the second regularized QGT inverse singular value\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   ...\n";
      _file_QGT_EIGENVALUES << "#   j-th: the j-th non-regularized QGT singular value\n";
      _file_QGT_EIGENVALUES << "#   (j + 1)-th: the j-th regularized QGT singular value\n";
      _file_QGT_EIGENVALUES << "#   (j + 2)-th: the j-th regularized QGT inverse singular value\n";
      _file_QGT_EIGENVALUES << "######################################################################\n";

    }
    else{

      std::cerr << " ##ValueError: method of regularization not available." << std::endl;
      std::cerr << "   Failed to write on file the Quantum Geometric Tensor eigenvalues (or singular values)." << std::endl;
      std::abort();

    }

  }

}


void VMC_Sampler :: write_Move_Statistics(int tvmc_step, MPI_Comm common) {

  if(_if_write_MOVE_STATISTICS){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

    //Resets global variables
    _global_acc_real = 0;
    _global_prop_real = 0;
    _global_acc_ket = 0;
    _global_prop_ket = 0;
    _global_acc_bra = 0;
    _global_prop_bra = 0;
    _global_acc_equal_site = 0;
    _global_prop_equal_site = 0;
    _global_acc_real_nn_site = 0;
    _global_prop_real_nn_site = 0;
    _global_acc_shadows_nn_site = 0;
    _global_prop_shadows_nn_site = 0;
    _global_acc_global_ket_flip = 0;
    _global_prop_global_ket_flip = 0;
    _global_acc_global_bra_flip = 0;
    _global_prop_global_bra_flip = 0;

    //Shares move statistics among all the nodes
    MPI_Reduce(&_N_accepted_real, &_global_acc_real, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_real, &_global_prop_real, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_ket, &_global_acc_ket, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_ket, &_global_prop_ket, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_bra, &_global_acc_bra, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_bra, &_global_prop_bra, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_equal_site, &_global_acc_equal_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_equal_site, &_global_prop_equal_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_real_nn_site, &_global_acc_real_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_real_nn_site, &_global_prop_real_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_shadows_nn_site, &_global_acc_shadows_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_shadows_nn_site, &_global_prop_shadows_nn_site, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_global_ket_flip, &_global_acc_global_ket_flip, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_global_ket_flip, &_global_prop_global_ket_flip, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_accepted_global_bra_flip, &_global_acc_global_bra_flip, 1, MPI_INTEGER, MPI_SUM, 0, common);
    MPI_Reduce(&_N_proposed_global_bra_flip, &_global_prop_global_bra_flip, 1, MPI_INTEGER, MPI_SUM, 0, common);

    if(rank == 0){

      _file_MOVE_STATISTICS << tvmc_step + 1;
      _file_MOVE_STATISTICS << std::setprecision(2) << std::fixed;
      _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_real / _global_prop_real;
      if(_N_proposed_ket == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_ket / _global_prop_ket;
      if(_N_proposed_bra == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_bra / _global_prop_bra;
      if(_N_proposed_equal_site == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_equal_site / _global_prop_equal_site;
      if(_N_proposed_global_ket_flip == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_global_ket_flip / _global_prop_global_ket_flip;
      if(_N_proposed_global_bra_flip == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_global_bra_flip / _global_prop_global_bra_flip;
      if(_N_proposed_real_nn_site == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_real_nn_site / _global_prop_real_nn_site;
      if(_N_proposed_shadows_nn_site == 0) _file_MOVE_STATISTICS << "\t" << 0.0;
      else _file_MOVE_STATISTICS << "\t" << 100.0 * _global_acc_shadows_nn_site / _global_prop_shadows_nn_site << std::endl;
      _file_MOVE_STATISTICS << std::endl;

    }

  }
  else return;

}


void VMC_Sampler :: write_MCMC_Config(int mcmc_step, int rank) {

  if(_if_write_MCMC_CONFIG){

    if(rank == 0){

      _file_MCMC_CONFIG << mcmc_step + 1;

      //Prints the 𝓇ℯ𝒶𝑙 configuration |𝒗⟩
      _file_MCMC_CONFIG << "\t\t|𝒗 ⟩" << std::setw(4);
      for(int j_row = 0; j_row < _configuration.n_rows; j_row++){

        for(int j_col = 0; j_col < _configuration.n_cols; j_col++)
          _file_MCMC_CONFIG << _configuration.at(j_row, j_col) << std::setw(4);
        _file_MCMC_CONFIG << std::endl << "   " << std::setw(4);

      }

      //Prints the ket configuration |𝒉⟩
      if(_shadow_ket.is_empty()) _file_MCMC_CONFIG << "\t\t|𝒉 ⟩" << std::endl;
      else{

        _file_MCMC_CONFIG << "\t\t|𝒉 ⟩" << std::setw(4);
        for(int j_row = 0; j_row < _shadow_ket.n_rows; j_row++){

          for(int j_col = 0; j_col < _shadow_ket.n_cols; j_col++)
            _file_MCMC_CONFIG << _shadow_ket.at(j_row, j_col) << std::setw(4);
          _file_MCMC_CONFIG << std::endl << "   " << std::setw(4);

        }

      }

      //Prints the bra configuration ⟨𝒉ˈ|
      if(_shadow_bra.is_empty()) _file_MCMC_CONFIG << "\t\t⟨𝒉ˈ|" << std::endl;
      else{

        _file_MCMC_CONFIG << "\t\t⟨𝒉ˈ|" << std::setw(4);
        for(int j_row = 0; j_row < _shadow_bra.n_rows; j_row++){

          for(int j_col = 0; j_col < _shadow_bra.n_cols; j_col++)
            _file_MCMC_CONFIG << _shadow_bra.at(j_row, j_col) << std::setw(4);
          _file_MCMC_CONFIG << std::endl;

        }

      }

    }

  }
  else return;

}


void VMC_Sampler :: write_final_Config(int tvmc_step, int rank, int only_one_rank) {

  if(_if_write_FINAL_CONFIG){

    if(only_one_rank == 1){  //Default case

      if(rank == 0){

        _file_FINAL_CONFIG << tvmc_step + 1 << "\t\t|𝒗 ⟩" << std::setw(4);
        //Prints the 𝓇ℯ𝒶𝑙 configuration |𝒗 ⟩
        for(int j_row = 0; j_row < _configuration.n_rows; j_row++){

          for(int j_col = 0; j_col < _configuration.n_cols; j_col++)
            _file_FINAL_CONFIG << _configuration.at(j_row, j_col) << std::setw(4);
          _file_FINAL_CONFIG << std::endl << "   " << std::setw(4);

        }

        //Prints the ket configuration |𝒉 ⟩
        if(_shadow_ket.is_empty()) _file_FINAL_CONFIG << "\t\t|𝒉 ⟩" << std::endl;
        else{

          _file_FINAL_CONFIG << "\t\t|𝒉 ⟩" << std::setw(4);
          for(int j_row = 0; j_row < _shadow_ket.n_rows; j_row++){

            for(int j_col = 0; j_col < _shadow_ket.n_cols; j_col++)
              _file_FINAL_CONFIG << _shadow_ket.at(j_row, j_col) << std::setw(4);
            _file_FINAL_CONFIG << std::endl;;

          }

        }

        //Prints the bra configuration ⟨𝒉ˈ|
        if(_shadow_bra.is_empty()) _file_FINAL_CONFIG << "\t\t⟨𝒉ˈ|" << std::endl;
        else{

          _file_FINAL_CONFIG << "\t\t⟨𝒉ˈ|" << std::setw(4);
          for(int j_row = 0; j_row < _shadow_bra.n_rows; j_row++){

            for(int j_col = 0; j_col < _shadow_bra.n_cols; j_col++)
              _file_FINAL_CONFIG << _shadow_bra.at(j_row, j_col) << std::setw(4);
            _file_FINAL_CONFIG << std::endl;

          }

        }

      }

    }
    else{

      _file_FINAL_CONFIG << tvmc_step + 1 << "\t\t|𝒗 ⟩" << std::setw(4);
      //Prints the 𝓇ℯ𝒶𝑙 configuration |𝒗 ⟩
      for(int j_row = 0; j_row < _configuration.n_rows; j_row++){

        for(int j_col = 0; j_col < _configuration.n_cols; j_col++)
          _file_FINAL_CONFIG << _configuration.at(j_row, j_col) << std::setw(4);
        _file_FINAL_CONFIG << std::endl << "   " << std::setw(4);

      }

      //Prints the ket configuration |𝒉 ⟩
      if(_shadow_ket.is_empty()) _file_FINAL_CONFIG << "\t\t|𝒉 ⟩" << std::endl;
      else{

        _file_FINAL_CONFIG << "\t\t|𝒉 ⟩" << std::setw(4);
        for(int j_row = 0; j_row < _shadow_ket.n_rows; j_row++){

          for(int j_col = 0; j_col < _shadow_ket.n_cols; j_col++)
            _file_FINAL_CONFIG << _shadow_ket.at(j_row, j_col) << std::setw(4);
          _file_FINAL_CONFIG << std::endl;;

        }

      }

      //Prints the bra configuration ⟨𝒉ˈ|
      if(_shadow_bra.is_empty()) _file_FINAL_CONFIG << "\t\t⟨𝒉ˈ|" << std::endl;
      else{

        _file_FINAL_CONFIG << "\t\t⟨𝒉ˈ|" << std::setw(4);
        for(int j_row = 0; j_row < _shadow_bra.n_rows; j_row++){

          for(int j_col = 0; j_col < _shadow_bra.n_cols; j_col++)
            _file_FINAL_CONFIG << _shadow_bra.at(j_row, j_col) << std::setw(4);
          _file_FINAL_CONFIG << std::endl;

        }

      }

    }

  }
  else return;

}


void VMC_Sampler :: write_opt_VQS(int rank) {

  if(_if_write_OPT_VQS){

    if(rank == 0){

      if(_vqs.type_of_ansatz()  == "Neural Network") _file_OPT_VQS << _vqs.n_real() << "\n" << _vqs.shadow_density() << std::endl;
      else _file_OPT_VQS << _vqs.n_real() << std::endl;
      if(_if_phi) _file_OPT_VQS << _vqs .phi() << std::endl;
      for(int p = 0; p < _vqs.n_alpha(); p++) _file_OPT_VQS << _vqs.alpha_at(p) << std::endl;

    }

  }
  else return;

}


void VMC_Sampler :: write_VQS_evolution(int tvmc_step, int rank) {

  if(_if_write_VQS_EVOLUTION){

    if(rank == 0){

      if(_vqs.type_of_ansatz() == "Shadow"){

        _file_VQS_EVOLUTION << tvmc_step + 1;
        _file_VQS_EVOLUTION << std::setprecision(8) << std::fixed;
        _file_VQS_EVOLUTION << "\t" << _vqs.phi().real() << "\t" << _vqs.phi().imag();
        for(int p = 0; p < _vqs.n_alpha(); p++)
          _file_VQS_EVOLUTION << "\t" << _vqs.alpha_at(p).real() << "\t" << _vqs.alpha_at(p).imag();
        _file_VQS_EVOLUTION << std::endl;

      }
      else if(_vqs.type_of_ansatz() == "Jastrow"){

        _file_VQS_EVOLUTION << tvmc_step + 1;
        _file_VQS_EVOLUTION << std::setprecision(8) << std::fixed;
        _file_VQS_EVOLUTION << "\t" << _vqs.phi().real() << " " << _vqs.phi().imag();
        for(int p = 0; p < _vqs.n_alpha(); p++)
          _file_VQS_EVOLUTION << " " << _vqs.alpha_at(p).real() << " " << _vqs.alpha_at(p).imag();
        _file_VQS_EVOLUTION << std::endl;

      }
      else{

        _file_VQS_EVOLUTION << tvmc_step + 1;
        _file_VQS_EVOLUTION << std::setprecision(8) << std::fixed;
        _file_VQS_EVOLUTION << "\t" << _vqs.phi().real() << "\t" << _vqs.phi().imag() << std::endl;
        for(int p = 0; p < _vqs.n_alpha(); p++)
          _file_VQS_EVOLUTION << tvmc_step + 1 << "\t" << _vqs.alpha_at(p).real() << "\t" << _vqs.alpha_at(p).imag() << std::endl;

      }

    }

  }
  else return;

}


void VMC_Sampler :: write_QGT(int tvmc_step) {  //Helpful in debugging

  if(_if_write_QGT){

    _file_QGT << tvmc_step + 1 << "\t\t";
    _file_QGT << std::setprecision(10) << std::fixed;
    if(_vqs.type_of_ansatz() != "Neural Network"){

      for(int j = 0; j < _Q.row(0).n_elem; j++) _file_QGT << _Q.row(0)[j].real() << "  ";
      _file_QGT << std::endl;
      for(int d = 1; d < _Q.n_rows; d++){

        _file_QGT << "\t\t";
        for(int j = 0; j < _Q.row(d).n_elem; j++) _file_QGT << _Q.row(d)[j].real() << "  ";
        _file_QGT << std::endl;

      }

    }
    else{

      for(int j = 0; j < _Q.row(0).n_elem; j++) _file_QGT << _Q.row(0)[j] << "  ";
      _file_QGT << std::endl;
      for(int d = 1; d < _Q.n_rows; d++){

        _file_QGT << "\t\t";
        for(int j = 0; j < _Q.row(d).n_elem; j++) _file_QGT << _Q.row(d)[j] << "  ";
        _file_QGT << std::endl;

      }

    }

  }
  else return;

}


void VMC_Sampler :: write_QGT_condition_number(int tvmc_step) {  //Helpful in debugging

  if(_if_write_QGT_CONDITION_NUMBER){

    _file_QGT_CONDITION_NUMBER << tvmc_step + 1 << "\t\t";
    _file_QGT_CONDITION_NUMBER << std::setprecision(6) << std::fixed;
    if(_regularization_method == 0 || _regularization_method == 2 || _regularization_method == 4){

      if(_if_shadow) _file_QGT_CONDITION_NUMBER << rcond(real(_Q)) << std::endl;
      else _file_QGT_CONDITION_NUMBER << rcond(_Q) << std::endl;

    }
    else if(_regularization_method == 1){

      if(_if_shadow) _file_QGT_CONDITION_NUMBER << rcond(real(_Q)) << "\t" << rcond(real(_Q) + _eps * _I) << std::endl;
      else _file_QGT_CONDITION_NUMBER << rcond(_Q) << "\t" << rcond(_Q + _eps * _I) << std::endl;

    }
    else if(_regularization_method == 3){

      if(_if_shadow) _file_QGT_CONDITION_NUMBER << rcond(real(_Q)) << "\t" << rcond(real(_Q) + std::max(_lambda0 * std::pow(_b, tvmc_step + 1), _lambda_min) * _I) << std::endl;
      else _file_QGT_CONDITION_NUMBER << rcond(_Q) << "\t" << rcond(_Q + std::max(_lambda0 * std::pow(_b, tvmc_step + 1), _lambda_min) * _I) << std::endl;

    }
    else return;

  }
  else return;

}


void VMC_Sampler :: write_QGT_eigenvalues(int tvmc_step) {  //Helpful in debugging

  if(_if_write_QGT_EIGENVALUES){

    //Function variables
    vec eigenval_no_reg;
    vec eigenval_reg;
    bool no_reg = false, reg = false;

    //Computes the eigenvalues (or singular values) according to the regularization method
    _file_QGT_EIGENVALUES << std::setprecision(4) << std::scientific;
    if(_regularization_method == 0 || _regularization_method == 2){  //No regularization or Moore-Penrose pseudo-inverse

      if(_if_shadow) no_reg = eig_sym(eigenval_no_reg, real(_Q));
      else no_reg = eig_sym(eigenval_no_reg, _Q);
      _file_QGT_EIGENVALUES << tvmc_step + 1 << "\t";
      if(no_reg == true)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << eigenval_no_reg[eig_ID] << "\t";
      else _file_QGT_EIGENVALUES << no_reg << "\t";
      _file_QGT_EIGENVALUES << std::endl;

    }
    else if(_regularization_method == 1){  //Simple diagonal regularization

      _file_QGT_EIGENVALUES << tvmc_step + 1 << "\t";
      if(_if_shadow){

        no_reg = eig_sym(eigenval_no_reg, real(_Q));
        reg = eig_sym(eigenval_reg, real(_Q) + _eps * _I);
      }
      else{

        no_reg = eig_sym(eigenval_no_reg, _Q);
        reg = eig_sym(eigenval_reg, _Q + _eps * _I);

      }
      if(no_reg == true && reg == true)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << eigenval_no_reg[eig_ID] << "\t" << eigenval_reg[eig_ID] << "\t";
      else if(no_reg == true && reg == false)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << eigenval_no_reg[eig_ID] << "\t" << reg << "\t";
      else if(no_reg == false && reg == true)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << no_reg << "\t" << eigenval_reg[eig_ID] << "\t";
      else
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << no_reg << "\t" << reg << "\t";
      _file_QGT_EIGENVALUES << std::endl;

    }
    else if(_regularization_method == 3){  //Decaying diagonal regularization

      _file_QGT_EIGENVALUES << tvmc_step + 1 << "\t";
      if(_if_shadow){

        no_reg = eig_sym(eigenval_no_reg, real(_Q));
        reg = eig_sym(eigenval_reg, real(_Q) + std::max(_lambda0 * std::pow(_b, tvmc_step + 1), _lambda_min) * _I);

      }
      else{

        no_reg = eig_sym(eigenval_no_reg, _Q);
        reg = eig_sym(eigenval_reg, _Q + std::max(_lambda0 * std::pow(_b, tvmc_step + 1), _lambda_min) * _I);

      }
      if(no_reg == true && reg == true)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << eigenval_no_reg[eig_ID] << "\t" << eigenval_reg[eig_ID] << "\t";
      else if(no_reg == true && reg == false)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << eigenval_no_reg[eig_ID] << "\t" << reg << "\t";
      else if(no_reg == false && reg == true)
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << no_reg << "\t" << eigenval_reg[eig_ID] << "\t";
      else
        for(int eig_ID = 0; eig_ID < eigenval_no_reg.n_elem; eig_ID++) _file_QGT_EIGENVALUES << no_reg << "\t" << reg << "\t";
      _file_QGT_EIGENVALUES << std::endl;

    }
    else if(_regularization_method == 4){  //Cut-off SVD regularization

      _file_QGT_EIGENVALUES << tvmc_step + 1 << "\t";
      for(int eig_ID = 0; eig_ID < _s.n_elem; eig_ID++) _file_QGT_EIGENVALUES << _s[eig_ID] << "\t" << _s_reg[eig_ID] << "\t" << _s_inv[eig_ID] << "\t";
      _file_QGT_EIGENVALUES << std::endl;

    }
    else{

      std::cerr << " ##ValueError: method of regularization not available." << std::endl;
      std::cerr << "   Failed to write on file the Quantum Geometric Tensor eigenvalues (or singular values)." << std::endl;
      std::abort();

    }

  }
  else return;

}


void VMC_Sampler :: CloseFile(int rank, int only_one_rank) {

  if(_if_write_FINAL_CONFIG && only_one_rank != 1) _file_FINAL_CONFIG.close();
  else if(_if_write_FINAL_CONFIG && only_one_rank == 1)
    if(rank == 0)  _file_FINAL_CONFIG.close();

  if(rank == 0){

    if(_if_write_MOVE_STATISTICS) _file_MOVE_STATISTICS.close();
    if(_if_write_MCMC_CONFIG) _file_MCMC_CONFIG.close();
    _file_ENERGY.close();
    if(_if_measure_BLOCK_ENERGY) _file_BLOCK_ENERGY.close();
    if(_if_measure_NON_DIAGONAL_OBS) _file_SIGMAX.close();
    if(_if_measure_DIAGONAL_OBS){

      _file_MZ2.close();
      _file_SZSZ_CORR.close();

    }
    if(_if_measure_BLOCK_NON_DIAGONAL_OBS) _file_BLOCK_SIGMAX.close();
    if(_if_measure_BLOCK_DIAGONAL_OBS){

      _file_BLOCK_MZ2.close();
      _file_BLOCK_SZSZ_CORR.close();

    }
    if(_if_write_VQS_EVOLUTION) _file_VQS_EVOLUTION.close();
    if(_if_write_OPT_VQS) _file_OPT_VQS.close();
    if(_if_write_QGT) _file_QGT.close();
    if(_if_write_QGT_CONDITION_NUMBER) _file_QGT_CONDITION_NUMBER.close();
    if(_if_write_QGT_EIGENVALUES) _file_QGT_EIGENVALUES.close();

  }

}


void VMC_Sampler :: Finalize(int rank, int only_one_rank) {

  _rnd.SaveSeed(rank, only_one_rank);

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
  _instSzSzCorr.reset();

}


cx_mat VMC_Sampler :: reg_SVD_inverse(const cx_mat& X) {

  /*#########################################################################################*/
  //  Computes the inverse of X using the singular value decomposition (𝐒𝐕𝐃) technique.
  //  Given the complex square matrix X, first it is decomposed as
  //
  //        X = U • 𝚫 • V^†
  //
  //  where 𝚫 is the diagonal matrix of singular values.
  //  After decomposition, a regularization on 𝚫 is carried out, controlled by the
  //  ε𝟣 and ε𝟤 parameter: we neglect all the singular value such that |𝚫ⱼⱼ| < ε𝟤;
  //  then if ε𝟤 < |𝚫ⱼⱼ| < ε𝟣, we regularize this small singular value as
  //
  //        𝚫ⱼⱼ → 𝚫ⱼⱼ = sign(𝚫ⱼⱼ) • ε𝟣
  //
  //  where sign(𝚫ⱼⱼ) is the sign of the eigenvalue. After checking all the diagonal of
  //  𝚫 following this criterion, we use the regularized matrix 𝚫(ε𝟣, ε𝟤) to obtain the
  //  inverse of X:
  //
  //        X^{-1} = [V^†]^{-1} • [𝚫(ε𝟣, ε𝟤)]^{-1} • U^{-1}
  //
  //  where clearly 𝚫 is the matrix whose diagonal presents the inverses of the regularized
  //  singular values, i.e.
  //
  //        [𝚫(ε𝟣, ε𝟤)]^{-1} = [𝚫ⱼⱼ(ε𝟣, ε𝟤)]^-1 • 𝟙.
  /*#########################################################################################*/

  //Function variables
  cx_mat U, V;

  //Performs the 𝐒𝐕𝐃 decomposition of the entering matrix
  svd(U, _s, V, X);

  //Performs the regularization of the singular values
  _s_reg.set_size(_s.n_elem);  // 𝚫ⱼⱼ(ε)
  _s_inv.set_size(_s.n_elem);  // [𝚫(ε𝟣, ε𝟤)]^{-1}
  for(int j = 0; j < _s.n_elem; j++){

    if(_s[j] > _eps2 && _s[j] < _eps1){

      _s_reg[j] = + _eps1;
      _s_inv[j] = 1.0 / _eps1;

    }
    else if(_s[j] > - _eps1 && _s[j] < - _eps2){

      _s_reg[j] = - _eps1;
      _s_inv[j] = - 1.0 / _eps1;

    }
    else if(_s[j] > - _eps2 && _s[j] < + _eps2){

      _s_reg[j] = 1E10;  //symbolic assignment, it will be zero in the inverse
      _s_inv[j] = 0.0;

    }
    else{

      _s_reg[j] = _s[j];
      _s_inv[j] = 1.0 / _s[j];

    }

  }

  //Computes and returns the inverse matrix X^{-1}
  return (V.t()).i() * diagmat(_s_inv) * U.i();

}


mat VMC_Sampler :: reg_SVD_inverse(const mat& X) {

  /*#########################################################################################*/
  //  Computes the inverse of X using the singular value decomposition (𝐒𝐕𝐃) technique.
  //  Given the complex square matrix X, first it is decomposed as
  //
  //        X = U • 𝚫 • Vᵀ
  //
  //  where 𝚫 is the diagonal matrix of singular values.
  //  After decomposition, a regularization on 𝚫 is carried out, controlled by the
  //  ε𝟣 and ε𝟤 parameter: we neglect all the singular value such that |𝚫ⱼⱼ| < ε𝟤;
  //  then if ε𝟤 < |𝚫ⱼⱼ| < ε𝟣, we regularize this small singular value as
  //
  //        𝚫ⱼⱼ → 𝚫ⱼⱼ = sign(𝚫ⱼⱼ) • ε𝟣
  //
  //  where sign(𝚫ⱼⱼ) is the sign of the eigenvalue. After checking all the diagonal of
  //  𝚫 following this criterion, we use the regularized matrix 𝚫(ε𝟣, ε𝟤) to obtain the
  //  inverse of X:
  //
  //        X^{-1} = [Vᵀ]^{-1} • [𝚫(ε𝟣, ε𝟤)]^{-1} • U^{-1}
  //
  //  where clearly 𝚫 is the matrix whose diagonal presents the inverses of the regularized
  //  singular values, i.e.
  //
  //        [𝚫(ε𝟣, ε𝟤)]^{-1} = [𝚫ⱼⱼ(ε𝟣, ε𝟤)]^-1 • 𝟙.
  /*#########################################################################################*/

  //Function variables
  mat U, V;

  //Performs the 𝐒𝐕𝐃 decomposition of the entering matrix
  svd(U, _s, V, X);

  //Performs the regularization of the singular values
  _s_reg.set_size(_s.n_elem);  // 𝚫ⱼⱼ(ε)
  _s_inv.set_size(_s.n_elem);  // [𝚫(ε𝟣, ε𝟤)]^{-1}
  for(int j = 0; j < _s.n_elem; j++){

    if(_s[j] > _eps2 && _s[j] < _eps1){

      _s_reg[j] = + _eps1;
      _s_inv[j] = 1.0 / _eps1;

    }
    else if(_s[j] > - _eps1 && _s[j] < - _eps2){

      _s_reg[j] = - _eps1;
      _s_inv[j] = - 1.0 / _eps1;

    }
    else if(_s[j] > - _eps2 && _s[j] < + _eps2){

      _s_reg[j] = 1E10;  //symbolic assignment, it will be zero in the inverse
      _s_inv[j] = 0.0;

    }
    else{

      _s_reg[j] = _s[j];
      _s_inv[j] = 1.0 / _s[j];

    }

  }

  //Computes and returns the inverse matrix X^{-1}
  return (V.t()).i() * diagmat(_s_inv) * U.i();

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
  //        𝒜 ≡ 𝒜(𝒗) = Σ𝒗' ⟨𝒗|𝔸|𝒗'⟩ • Ψ(𝒗',𝛂)/Ψ(𝒗,𝛂)        (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //        𝒜 ≡ 𝒜(𝒗,𝒉) = Σ𝒗' ⟨𝒗|𝔸|𝒗'⟩ • Φ(𝒗',𝒉,𝛂)/Φ(𝒗,𝒉,𝛂)  (𝓈ℎ𝒶𝒹ℴ𝓌)
  //
  //  where the matrix elements ⟨𝒗|𝔸|𝒗'⟩ are the connections of the
  //  quantum observable operator 𝔸 related to the 𝓇ℯ𝒶𝑙 configuration |𝒗⟩ and
  //  the |𝒗'⟩ configurations are all the system configurations connected to |𝒗⟩.
  //  Whereupon, we can compute the Monte Carlo average value of 𝐀𝐍𝐘 quantum
  //  observable 𝔸 on the variational state as
  //
  //        ⟨𝔸⟩ = ⟨𝒜⟩             (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //        ⟨𝔸⟩ = ≪𝒜ᴿ≫ + ⌈𝒜ᴵ⌋   (𝓈ℎ𝒶𝒹ℴ𝓌)
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
  //  N̲O̲T̲E̲: in the case of the 𝓈ℎ𝒶𝒹ℴ𝓌 wave function it may be necessary to make many more
  //        integrations of the auxiliary variables, compared to those already made in each
  //        simulation together with the 𝓇ℯ𝒶𝑙 ones. This is due to the fact that the
  //        correlations induced by the auxiliary variables, which are not physical,
  //        could make the instantaneous measurement of the observables very noisy,
  //        making the algorithm unstable, especially in the inversion of the QGT.
  //        Therefore we add below the possibility to take further samples of the
  //        𝓈ℎ𝒶𝒹ℴ𝓌 variables within the single Monte Carlo measurement, to increase the
  //        statistics and make the block observables less noisy along each simulation.
  /*########################################################################################################*/

  //Find the connections of each non-diagonal observables (including the energy)
  _H.FindConn(_configuration, _StatePrime, _Connections);  // ⟨𝒗|𝔸|𝒗'⟩ for all |𝒗'⟩

  //Function variables
  int n_props;  //Number of quantum non-diagonal observables to be computed via 𝐌𝐂
  if(_if_measure_NON_DIAGONAL_OBS || _if_measure_BLOCK_NON_DIAGONAL_OBS) n_props = _Connections.n_rows;
  else n_props = 1;  //Only energy computation
  rowvec magnetization;  //Storage variable for (𝗠ᶻ)^2
  vec Cz_of_r;  //Storage variable for 𝗖ᶻ(𝙧)
  double r = 0.0;  //Storage variable for the correlation distance 𝙧
  double d = 0.0;  //Spin distance
  int l_max;  //Max correlation length
  _Observables.set_size(n_props, 1);  //Only sizing, this should be computed in 𝐄𝐬𝐭𝐢𝐦𝐚𝐭𝐞()
  _global_Observables.set_size(n_props, 1);  //Only sizing, this should be computed later
  vec cosin(2, fill::zeros);  //Storage variable for cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')] and sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
  cx_vec A_ket(n_props, fill::zeros);  //Storage variable for 𝒜(𝒗,𝒉)
  cx_vec A_bra(n_props, fill::zeros);  //Storage variable for 𝒜(𝒗,𝒉ˈ)
  cx_vec O_ket(_vqs.n_alpha(), fill::zeros);  //Storage variable for 𝓞(𝒗,𝒉)
  cx_vec O_bra(_vqs.n_alpha(), fill::zeros);  //Storage variable for 𝓞(𝒗,𝒉ˈ)

  //Makes the 𝓈ℎ𝒶𝒹ℴ𝓌 measurement less noisy
  if(_if_shadow == true && _if_extra_shadow_sum == true){

    //Extra sampling of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables
    if(_N_blks_extra == 0){

      std::cerr << " ##ValueError: not to use “block averaging” during the extra 𝓈ℎ𝒶𝒹ℴ𝓌 sampling set _N_blks_extra = 𝟣." << std::endl;
      std::cerr << "   Failed to measure instantaneous quantum properties of the system." << std::endl;
      std::abort();

    }
    else if(_N_blks_extra == 1){  //No “block averaging”

      for(int extra_step = 0; extra_step < _N_extra; extra_step++){

        for(int n_bunch = 0; n_bunch < _M; n_bunch++){

          this -> Move_ket(_N_flips);
          this -> Move_bra(_N_flips);

        }
        if(_rnd.Rannyu() < _p_global_ket_flip) this -> Move_global_ket_flip();
        if(_rnd.Rannyu() < _p_global_bra_flip) this -> Move_global_bra_flip();

        cosin[0] += _vqs.cosII(_configuration, _shadow_ket, _shadow_bra);
        cosin[1] += _vqs.sinII(_configuration, _shadow_ket, _shadow_bra);
        _vqs.LocalOperators(_configuration, _shadow_ket, _shadow_bra);
        O_ket += _vqs.O().col(0);
        O_bra += _vqs.O().col(1);
        for(int Nobs = 0; Nobs < n_props; Nobs++){

          for(int mel = 0; mel < _Connections[Nobs].n_elem; mel++){

            A_ket[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _shadow_ket);  // 𝒜(𝒗,𝒉)
            A_bra[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _shadow_bra);  // 𝒜(𝒗,𝒉')

          }

        }

      }
      cosin /= double(_N_extra);  //  ⟨⟨𝑐𝑜𝑠⟩ᵇˡᵏ⟩ & ⟨⟨𝑠𝑖𝑛⟩ᵇˡᵏ⟩
      A_ket /= double(_N_extra);  //  ⟨⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ⟩
      A_bra /= double(_N_extra);  //  ⟨⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ⟩
      O_ket /= double(_N_extra);  //  ⟨⟨𝓞(𝒗,𝒉)⟩ᵇˡᵏ⟩
      O_bra /= double(_N_extra);  //  ⟨⟨𝓞(𝒗,𝒉')⟩ᵇˡᵏ⟩

    }
    else{  //“block averaging”

      int blk_size = std::floor(double(_N_extra / _N_blks_extra));
      double cos_blk, sin_blk;
      cx_vec A_ket_blk(n_props);
      cx_vec A_bra_blk(n_props);
      cx_vec O_ket_blk(_vqs.n_alpha());
      cx_vec O_bra_blk(_vqs.n_alpha());
      for(int block_ID = 0; block_ID < _N_blks_extra; block_ID++){

        cos_blk = 0.0;
        sin_blk = 0.0;
        A_ket_blk.zeros();
        A_bra_blk.zeros();
        O_ket_blk.zeros();
        O_bra_blk.zeros();
        for(int l =  0; l < blk_size; l++){  //Computes single block estimates of the instantaneous measurement

          for(int n_bunch = 0; n_bunch < _M; n_bunch++){  //Moves only the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration

            this -> Move_ket(_N_flips);
            this -> Move_bra(_N_flips);

          }
	  if(_rnd.Rannyu() < _p_global_ket_flip) this -> Move_global_ket_flip();
          if(_rnd.Rannyu() < _p_global_bra_flip) this -> Move_global_bra_flip();
         
	  cos_blk += _vqs.cosII(_configuration, _shadow_ket, _shadow_bra);
          sin_blk += _vqs.sinII(_configuration, _shadow_ket, _shadow_bra);
          _vqs.LocalOperators(_configuration, _shadow_ket, _shadow_bra);
          O_ket_blk += _vqs.O().col(0);
          O_bra_blk += _vqs.O().col(1);
          for(int Nobs = 0; Nobs < n_props; Nobs++){

            for(int mel = 0; mel < _Connections[Nobs].n_elem; mel++){

              A_ket_blk[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _shadow_ket);  // 𝒜(𝒗,𝒉)
              A_bra_blk[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _shadow_bra);  // 𝒜(𝒗,𝒉')

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
      cosin /= double(_N_blks_extra);  //  ⟨⟨𝑐𝑜𝑠⟩ᵇˡᵏ⟩ & ⟨⟨𝑠𝑖𝑛⟩ᵇˡᵏ⟩
      A_ket /= double(_N_blks_extra);  //  ⟨⟨𝒜(𝒗,𝒉)⟩ᵇˡᵏ⟩
      A_bra /= double(_N_blks_extra);  //  ⟨⟨𝒜(𝒗,𝒉')⟩ᵇˡᵏ⟩
      O_ket /= double(_N_blks_extra);  //  ⟨⟨𝓞(𝒗,𝒉)⟩ᵇˡᵏ⟩
      O_bra /= double(_N_blks_extra);  //  ⟨⟨𝓞(𝒗,𝒉')⟩ᵇˡᵏ⟩

    }

  }
  else{

    //Computes cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')] and sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cosin[0] = _vqs.cosII(_configuration, _shadow_ket, _shadow_bra);
    cosin[1] = _vqs.sinII(_configuration, _shadow_ket, _shadow_bra);

    //Instantaneous evaluation of the quantum observables
    for(int Nobs = 0; Nobs < n_props; Nobs++){

      for(int mel = 0; mel < _Connections[Nobs].n_elem; mel++){

        A_ket[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _shadow_ket);  // 𝒜(𝒗,𝒉)
        A_bra[Nobs] += _Connections.at(Nobs, 0)[mel] * _vqs.PhiNew_over_PhiOld(_configuration, _StatePrime.at(Nobs, 0).at(0, mel), _shadow_bra);  // 𝒜(𝒗,𝒉')

      }

    }

    //Instantaneous evaluation of the local operators
    _vqs.LocalOperators(_configuration, _shadow_ket, _shadow_bra);  //Computes 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉')
    O_ket = _vqs.O().col(0);
    O_bra = _vqs.O().col(1);

  }

  //Computes diagonal observables (𝗠ᶻ)^2 and 𝗖ᶻ(𝙧)
  if(_if_measure_DIAGONAL_OBS || _if_measure_BLOCK_DIAGONAL_OBS){

    magnetization.zeros(1);
    l_max = std::floor(_L / 2.0);
    Cz_of_r.zeros(l_max + 1);  //The +1 refers to the auto-correlation

    //Instantaneous squared magnetization (𝗠ᶻ)^2 = (Σⱼ 𝜎ⱼᶻ)(Σₖ 𝜎ₖᶻ)
    for(int j_row = 0; j_row < _configuration.n_rows; j_row++)
      for(int j_col = 0; j_col < _configuration.n_cols; j_col++) magnetization[0] += double(_configuration.at(j_row, j_col));

    //Instantaneous 𝗖ᶻ(𝙧)
    //Computes all the interactions in PBCs, see LRHJas in ansatz.cpp
    for(int j = 0; j < _L; j++){

      for(int k = j; k < _L; k++){

        //Compute 𝙧
        d = std::abs(double(j - k));
        if(_L % 2 == 0) r = l_max - std::abs(d - 1.0*l_max);  // 𝖫 𝒆𝒗𝒆𝒏
        else{  // 𝖫 𝒐𝒅𝒅

          if(d < l_max + 1) r = d;
          else if(d == l_max + 1) r = d - 1;
          else if(d > l_max + 1) r = (l_max + 1) - (d - l_max);

        }

        //Add the correlation
        Cz_of_r[r] += double(_configuration.at(0, j) * _configuration.at(0, k));

      }

    }

    _instSquareMag.insert_cols(_instSquareMag.n_cols, magnetization % magnetization);  // ≡ instantaneous measure of (𝗠ᶻ)^2
    _instSzSzCorr.insert_cols(_instSzSzCorr.n_cols, Cz_of_r);  // ≡ instantaneous measure of 𝗖ᶻ(𝙧)

  }

  //Adds Monte Carlo statistics
  _instReweight.insert_cols(_instReweight.n_cols, cosin);  // ≡ instantaneous measure of the 𝑐𝑜𝑠 and of the 𝑠𝑖𝑛
  _instObs_ket.insert_cols(_instObs_ket.n_cols, A_ket);  // ≡ instantaneous measure of 𝒜(𝒗,𝒉)
  _instObs_bra.insert_cols(_instObs_bra.n_cols, A_bra);  // ≡ instantaneous measure of 𝒜(𝒗,𝒉')
  _instO_ket.insert_cols(_instO_ket.n_cols, O_ket);  // ≡ instantaneous measure of 𝓞(𝒗,𝒉)
  _instO_bra.insert_cols(_instO_bra.n_cols, O_bra);  // ≡ instantaneous measure of 𝓞(𝒗,𝒉')

}


void VMC_Sampler :: Estimate(MPI_Comm common, int p) {

  /*#############################################################################################*/
  //  This function is called at the end of the single 𝐭𝐕𝐌𝐂 step and
  //  estimates the averages of the quantum observables
  //  as a Monte Carlo stochastic mean value on the choosen variational quantum state, i.e.:
  //
  //        ⟨𝔸⟩ = ⟨𝒜⟩             (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //        ⟨𝔸⟩ = ≪𝒜ᴿ≫ + ⌈𝒜ᴵ⌋   (𝓈ℎ𝒶𝒹ℴ𝓌)
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
  //  In the 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 case we have:
  //
  //        ≪◦≫ → ‹›, i.e. the standard Monte Carlo expectation value
  //        ⌈◦⌋ → 0
  //
  //  The instantaneous values along the single Markov chain necessary to make the Monte Carlo
  //  estimates just defined are computed by the 𝐌𝐞𝐚𝐬𝐮𝐫𝐞() function and are stored in the
  //  following data-members:
  //
  //        _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭  ‹--›  quantum non-diagonal observable 𝒜(𝒗,𝒉)
  //        _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚  ‹--›  quantum non-diagonal observable 𝒜(𝒗,𝒉')
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
    this -> compute_QGTandGrad(common, p);
    this -> QGT_Check(rank);

  }

}


void VMC_Sampler :: write_Quantum_properties(int tvmc_step, int rank) {

  /*############################################################*/
  //  We save on the output file the real and imaginary part
  //  with the relative uncertainties of the
  //  quantum observables via "block averaging": if everything
  //  has gone well, the imaginary part of the estimates of
  //  quantum operators MUST be statistically zero.
  /*############################################################*/

  if(rank == 0){

    //Computes progressive averages of the reweighting ratio with "block averaging" uncertainties
    vec prog_cos = this -> compute_progressive_averages(_global_cosII);
    vec err_cos = this -> compute_errorbar(_global_cosII);
    vec prog_sin = this -> compute_progressive_averages(_global_sinII);
    vec err_sin = this -> compute_errorbar(_global_sinII);

    //Calculates energy and non-diagonal quantum properties.
    //The algorithm distinguishes 𝓈ℎ𝒶𝒹ℴ𝓌 or 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏.
    if(!_if_shadow){

      //Computes progressive averages of non-diagonal quantum observables with "block averaging" uncertainties
      cx_vec prog_energy = this -> compute_progressive_averages(_global_Observables.at(0, 0));
      cx_vec err_energy = this -> compute_errorbar(_global_Observables.at(0, 0));
      cx_vec prog_Sx;
      cx_vec err_Sx;
      if(_if_measure_NON_DIAGONAL_OBS || _if_measure_BLOCK_NON_DIAGONAL_OBS){

        prog_Sx = this -> compute_progressive_averages(_global_Observables.at(1, 0));
        err_Sx = this -> compute_errorbar(_global_Observables.at(1, 0));

      }

      //Writes variational energy along the 𝐭𝐕𝐌𝐂: 𝐸(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝐸(𝜙,𝛂)]
      if(_if_measure_ENERGY){

        if(_if_write_ENERGY_ALL){

          _file_ENERGY << std::setprecision(5) << std::fixed;
          _file_ENERGY << tvmc_step + 1 << "\t";
          _file_ENERGY << prog_cos[_N_blks - 1] << "\t" << err_cos[_N_blks - 1] << "\t";
          _file_ENERGY << prog_sin[_N_blks - 1] << "\t" << err_sin[_N_blks - 1] << "\t";
          _file_ENERGY << std::setprecision(18) << std::fixed;
          _file_ENERGY << prog_energy[_N_blks - 1].real() << "\t" << err_energy[_N_blks - 1].real() << "\t";
          _file_ENERGY << prog_energy[_N_blks - 1].imag() << "\t" << err_energy[_N_blks - 1].imag() << "\t";
          _file_ENERGY << _E.real() << "\t" << _E.imag();
          _file_ENERGY << std::endl;

        }
        else{

          _file_ENERGY << std::setprecision(5) << std::fixed;
          _file_ENERGY << tvmc_step + 1 << "\t";
          _file_ENERGY << std::setprecision(18) << std::fixed;
          _file_ENERGY << prog_energy[_N_blks - 1].real() << "\t" << err_energy[_N_blks - 1].real() << "\t";
          _file_ENERGY << prog_energy[_N_blks - 1].imag() << "\t" << err_energy[_N_blks - 1].imag() << "\t";
          _file_ENERGY << std::endl;

        }

      }

      //Writes all the statistics calculations for the energy on file
      if(_if_measure_BLOCK_ENERGY){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          _file_BLOCK_ENERGY << std::setprecision(10) << std::fixed;
          _file_BLOCK_ENERGY << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_BLOCK_ENERGY << prog_cos[block_ID] << "\t" << err_cos[block_ID] << "\t";
          _file_BLOCK_ENERGY << prog_sin[block_ID] << "\t" << err_sin[block_ID] << "\t";
          _file_BLOCK_ENERGY << prog_energy[block_ID].real() << "\t" << err_energy[block_ID].real() << "\t";
          _file_BLOCK_ENERGY << prog_energy[block_ID].imag() << "\t" << err_energy[block_ID].imag() << "\t";
          _file_BLOCK_ENERGY << std::endl;

        }

      }

      //Writes non-diagonal system properties along the 𝐭𝐕𝐌𝐂 on file
      if(_if_measure_NON_DIAGONAL_OBS){

        // 𝝈(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝝈(𝜙, 𝛂)]
        _file_SIGMAX << std::setprecision(20) << std::fixed;
        _file_SIGMAX << tvmc_step + 1 << "\t";
        _file_SIGMAX << prog_Sx[_N_blks - 1].real() << "\t" << err_Sx[_N_blks - 1].real() << "\t";
        _file_SIGMAX << prog_Sx[_N_blks - 1].imag() << "\t" << err_Sx[_N_blks - 1].imag() << "\t";
        _file_SIGMAX << std::endl;

      }

      //Writes all the statistics calculations for the non-diagonal observables on file
      if(_if_measure_BLOCK_NON_DIAGONAL_OBS){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          _file_BLOCK_SIGMAX << std::setprecision(10) << std::fixed;
          _file_BLOCK_SIGMAX << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_BLOCK_SIGMAX << prog_Sx[block_ID].real() << "\t" << err_Sx[block_ID].real() << "\t";
          _file_BLOCK_SIGMAX << prog_Sx[block_ID].imag() << "\t" << err_Sx[block_ID].imag() << "\t";
          _file_BLOCK_SIGMAX << std::endl;

        }

      }

    }
    else{

      //Computes the true Shadow observable via reweighting ratio in each block
      vec shadow_energy = real(_global_Observables.at(0, 0)) / _global_cosII;  //Computes ⟨Ĥ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
      vec shadow_Sx;
      vec prog_Sx;
      vec err_Sx;

      //Computes progressive averages of quantum observables with "block averaging" uncertainties
      vec prog_energy = this -> compute_progressive_averages(shadow_energy);
      vec err_energy = this -> compute_errorbar(shadow_energy);
      if(_if_measure_NON_DIAGONAL_OBS || _if_measure_BLOCK_NON_DIAGONAL_OBS){

        shadow_Sx = real(_global_Observables.at(1, 0)) / _global_cosII;  //Computes ⟨σ̂ˣ⟩ⱼᵇˡᵏ/⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block
        prog_Sx = this -> compute_progressive_averages(shadow_Sx);
        err_Sx = this -> compute_errorbar(shadow_Sx);

      }

      //Writes variational energy along the 𝐭𝐕𝐌𝐂: 𝐸(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝐸(𝜙,𝛂)]
      if(_if_measure_ENERGY){

        if(_if_write_ENERGY_ALL){

          _file_ENERGY << std::setprecision(8) << std::fixed;
          _file_ENERGY << tvmc_step + 1 << "\t";
          _file_ENERGY << prog_cos[_N_blks - 1] << "\t" << err_cos[_N_blks - 1] << "\t";
          _file_ENERGY << prog_sin[_N_blks - 1] << "\t" << err_sin[_N_blks - 1] << "\t";
          _file_ENERGY << std::setprecision(12) << std::fixed;
          _file_ENERGY << prog_energy[_N_blks - 1] << "\t" << err_energy[_N_blks - 1] << "\t";
          _file_ENERGY << 0.0 << "\t" << 0.0 << "\t" << real(_global_Observables.at(0, 0)[_N_blks - 1]) << "\t" << imag(_global_Observables.at(0, 0)[_N_blks - 1]);
          _file_ENERGY << std::endl;

        }
        else{

          _file_ENERGY << std::setprecision(8) << std::fixed;
          _file_ENERGY << tvmc_step + 1 << "\t";
          _file_ENERGY << std::setprecision(12) << std::fixed;
          _file_ENERGY << prog_energy[_N_blks - 1] << "\t" << err_energy[_N_blks - 1] << "\t";
          _file_ENERGY << 0.0 << "\t" << 0.0;
          _file_ENERGY << std::endl;

        }

      }

      //Writes all the statistics calculations for the energy on file
      if(_if_measure_BLOCK_ENERGY){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          _file_BLOCK_ENERGY << std::setprecision(8) << std::fixed;
          _file_BLOCK_ENERGY << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_BLOCK_ENERGY << prog_cos[block_ID] << "\t" << err_cos[block_ID] << "\t";
          _file_BLOCK_ENERGY << prog_sin[block_ID] << "\t" << err_sin[block_ID] << "\t";
          _file_BLOCK_ENERGY << std::setprecision(12) << std::fixed;
          _file_BLOCK_ENERGY << real(_global_Observables.at(0, 0)[block_ID]) << "\t" << imag(_global_Observables.at(0, 0)[block_ID]) << "\t";
          _file_BLOCK_ENERGY << prog_energy[block_ID] << "\t" << err_energy[block_ID] << "\t";
          _file_BLOCK_ENERGY << 0.0 << "\t" << 0.0 << "\t";
          _file_BLOCK_ENERGY << std::endl;

        }

      }

      //Writes non-diagonal system properties along the 𝐭𝐕𝐌𝐂 on file
      if(_if_measure_NON_DIAGONAL_OBS){

        // 𝝈(𝜙,𝛂) +/- 𝓈𝓉𝒹[𝝈(𝜙, 𝛂)]
        _file_SIGMAX << std::setprecision(20) << std::fixed;
        _file_SIGMAX << tvmc_step + 1 << "\t";
        _file_SIGMAX << prog_Sx[_N_blks - 1] << "\t" << err_Sx[_N_blks - 1] << "\t";
        _file_SIGMAX << 0.0 << "\t" << 0.0 << "\t";
        _file_SIGMAX << std::endl;

      }

      //Writes all the statistics calculations for the non-diagonal observables on file
      if(_if_measure_BLOCK_NON_DIAGONAL_OBS){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          _file_BLOCK_SIGMAX << std::setprecision(15) << std::fixed;
          _file_BLOCK_SIGMAX << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_BLOCK_SIGMAX << real(_global_Observables.at(1, 0)[block_ID]) << "\t" << imag(_global_Observables.at(1, 0)[block_ID]) << "\t";
          _file_BLOCK_SIGMAX << prog_Sx(block_ID) << "\t" << err_Sx(block_ID) << "\t";
          _file_BLOCK_SIGMAX << 0.0 << "\t" << 0.0 << "\t";
          _file_BLOCK_SIGMAX << std::endl;

        }

      }

    }

    //Calculates diagonal quantum properties.
    //In this case the algorithm does not distinguish 𝓈ℎ𝒶𝒹ℴ𝓌 or 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏
    if(_if_measure_DIAGONAL_OBS || _if_measure_BLOCK_DIAGONAL_OBS){

      vec prog_Mz2 = this -> compute_progressive_averages(_global_Mz2);
      vec err_Mz2 = this -> compute_errorbar(_global_Mz2);
      mat prog_Cz_of_r(_N_blks, _SzSzCorr.n_cols);
      mat err_Cz_of_r(_N_blks, _SzSzCorr.n_cols);
      for(int d = 0; d < prog_Cz_of_r.n_cols; d++){

        prog_Cz_of_r.col(d) = this -> compute_progressive_averages(_global_Cz_of_r.col(d));
        err_Cz_of_r.col(d) = this -> compute_errorbar(_global_Cz_of_r.col(d));

      }

      //Writes diagonal system properties along the 𝐭𝐕𝐌𝐂 on file
      if(_if_measure_DIAGONAL_OBS){

        //Writes (𝗠ᶻ)^2
        _file_MZ2 << std::setprecision(20) << std::fixed;
        _file_MZ2 << tvmc_step + 1 << "\t";
        _file_MZ2 << prog_Mz2[_N_blks - 1] << "\t" << err_Mz2[_N_blks - 1];
        _file_MZ2 << std::endl;

        //Writes 𝗖ᶻ(𝙧)
        for(int r = 0; r < prog_Cz_of_r.n_cols; r++){

          _file_SZSZ_CORR << std::setprecision(20) << std::fixed;
          _file_SZSZ_CORR << tvmc_step + 1 << "\t" << r << "\t";
          _file_SZSZ_CORR << prog_Cz_of_r.at(_N_blks - 1, r) << "\t" << err_Cz_of_r.at(_N_blks - 1, r);
          _file_SZSZ_CORR << std::endl;

        }

      }

      //Writes all the statistics calculations for the diagonal observables on file
      if(_if_measure_BLOCK_DIAGONAL_OBS){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          //Writes (𝗠ᶻ)^2
          _file_BLOCK_MZ2 << std::setprecision(10) << std::fixed;
          _file_BLOCK_MZ2 << tvmc_step + 1 << "\t" << block_ID + 1 << "\t";
          _file_BLOCK_MZ2 << prog_Mz2[block_ID] << "\t" << err_Mz2[block_ID];
          _file_BLOCK_MZ2 << std::endl;

          //Writes 𝗖ᶻ(𝙧)
          for(int r = 0; r < prog_Cz_of_r.n_cols; r++){

            _file_BLOCK_SZSZ_CORR << std::setprecision(10) << std::fixed;
            _file_BLOCK_SZSZ_CORR << tvmc_step + 1 << "\t" << block_ID + 1 << "\t" << r << "\t";
            _file_BLOCK_SZSZ_CORR << prog_Cz_of_r.at(block_ID, r) << "\t" << err_Cz_of_r.at(block_ID, r);
            _file_BLOCK_SZSZ_CORR << std::endl;

          }

        }

      }

    }

    //Quantum Geometric Tensor properties
    if(!_if_vmc){  //Helpful in debugging

      this -> write_QGT(tvmc_step);
      this -> write_QGT_condition_number(tvmc_step);
      this -> write_QGT_eigenvalues(tvmc_step);

    }

  }

}


vec VMC_Sampler :: average_in_blocks(const rowvec& instantaneous_quantity) const {

  /*############################################################*/
  //  This function takes a row from one of the matrix
  //  data-members which contains the instantaneous values
  //  of a certain system properties, calculated along a single
  //  Monte Carlo Markov Chain, and calculates all the averages
  //  in each block of this system properties.
  //  This calculation involves a real-valued quantity.
  /*############################################################*/

  //Function variables
  int blk_size = std::floor(double(instantaneous_quantity.n_elem / _N_blks));  //Sets the block length
  vec blocks_quantity(_N_blks);
  double sum_in_each_block;

  //Computes Monte Carlo averages in each block
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    sum_in_each_block = 0.0;  //Resets the storage variable in each block
    for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++) sum_in_each_block += instantaneous_quantity[l];
    blocks_quantity[block_ID] = sum_in_each_block / double(blk_size);

  }

  return blocks_quantity;

}


cx_vec VMC_Sampler :: average_in_blocks(const cx_rowvec& instantaneous_quantity) const {

  /*############################################################*/
  //  This function takes a row from one of the matrix
  //  data-members which contains the instantaneous values
  //  of a certain system properties, calculated at each
  //  Monte Carlo Markov Chain, and calculates all the averages
  //  in each block of this system properties.
  //  This calculation involves a complex-valued quantity.
  /*############################################################*/

  //Function variables
  int blk_size = std::floor(double(instantaneous_quantity.n_elem / _N_blks));  //Sets the block length
  cx_vec blocks_quantity(_N_blks);
  cx_double sum_in_each_block;

  //Computes Monte Carlo averages in each block
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    sum_in_each_block = 0.0;  //Resets the storage variable in each block
    for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++) sum_in_each_block += instantaneous_quantity[l];
    blocks_quantity[block_ID] = sum_in_each_block / double(blk_size);

  }

  return blocks_quantity;

}


vec VMC_Sampler :: Shadow_average_in_blocks(const cx_rowvec& instantaneous_quantity_ket, const cx_rowvec& instantaneous_quantity_bra) const {

  /*################################################################*/
  //  Computes
  //
  //        ⟨𝔸⟩ᵇˡᵏ = ≪𝒜ᴿ≫ᵇˡᵏ + ⌈𝒜ᴵ⌋ᵇˡᵏ
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_N_blks));  //Sets the block length
  vec blocks_quantity(_N_blks);
  double sum_in_each_block;

  //Computes Monte Carlo Shadow averages in each block ( ! without the reweighting ratio ! )
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    sum_in_each_block = 0.0;
    for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

      sum_in_each_block += _instReweight.row(0)[l] * (instantaneous_quantity_ket[l].real() + instantaneous_quantity_bra[l].real());
      sum_in_each_block += _instReweight.row(1)[l] * (instantaneous_quantity_bra[l].imag() - instantaneous_quantity_ket[l].imag());

    }
    sum_in_each_block *= 0.5;
    blocks_quantity[block_ID] = sum_in_each_block / double(blk_size);

  }

  return blocks_quantity;

}


vec VMC_Sampler :: Shadow_angled_average_in_blocks(const cx_rowvec& instantaneous_quantity_ket, const cx_rowvec& instantaneous_quantity_bra) const {

  /*################################################################*/
  //  Computes
  //
  //        ≪𝒜ᴿ≫ᵇˡᵏ
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_N_blks));  //Sets the block length
  vec blocks_angled_quantity(_N_blks);
  double angled_sum_in_each_block;

  //Computes Monte Carlo Shadow “angled” averages in each block ( ! without the reweighting ratio ! )
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    angled_sum_in_each_block = 0.0;
    for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
      angled_sum_in_each_block += _instReweight.row(0)[l] * (instantaneous_quantity_ket[l].real() + instantaneous_quantity_bra[l].real());
    angled_sum_in_each_block *= 0.5;
    blocks_angled_quantity[block_ID] = angled_sum_in_each_block / double(blk_size);

  }

  return blocks_angled_quantity;

}


vec VMC_Sampler :: Shadow_square_average_in_blocks(const cx_rowvec& instantaneous_quantity_ket, const cx_rowvec& instantaneous_quantity_bra) const {

  /*################################################################*/
  //  Computes
  //
  //        ⌈𝒜ᴵ⌋ᵇˡᵏ
  //
  //  in each block for a choosen system property.
  /*################################################################*/

  //Function variables
  int blk_size = std::floor(double(instantaneous_quantity_ket.n_elem/_N_blks));  //Sets the block length
  vec blocks_square_quantity(_N_blks);
  double square_sum_in_each_block;

  //Computes Monte Carlo Shadow “square” averages in each block ( ! without the reweighting ratio ! )
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    square_sum_in_each_block = 0.0;
    for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
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

  _global_cosII.zeros(_N_blks);
  _global_sinII.zeros(_N_blks);
  _cosII = this -> average_in_blocks(_instReweight.row(0));  //Computes ⟨𝑐𝑜𝑠⟩ⱼᵇˡᵏ in each block, for j = 𝟣,…,𝖭ᵇˡᵏ
  _sinII = this -> average_in_blocks(_instReweight.row(1));  //Computes ⟨𝑠𝑖𝑛⟩ⱼᵇˡᵏ in each block, for j = 𝟣,…,𝖭ᵇˡᵏ

  MPI_Barrier(common);

  //Shares block averages among all the nodes
  MPI_Reduce(_cosII.begin(), _global_cosII.begin(), _N_blks, MPI_DOUBLE, MPI_SUM, 0, common);
  MPI_Reduce(_sinII.begin(), _global_sinII.begin(), _N_blks, MPI_DOUBLE, MPI_SUM, 0, common);
  if(rank == 0){

    _global_cosII /= double(size);
    _global_sinII /= double(size);

  }

}


void VMC_Sampler :: compute_Quantum_observables(MPI_Comm common) {

  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐞𝐧𝐞𝐫𝐠𝐲.
  //  We compute the stochastic average via the blocking technique of
  //
  //        𝐸(𝜙,𝛂) = ⟨Ĥ⟩ ≈ ⟨ℰ⟩            (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //        𝐸(𝜙,𝛂) = ⟨Ĥ⟩ ≈ ≪ℰᴿ≫ + ⌈ℰᴵ⌋   (𝓈ℎ𝒶𝒹ℴ𝓌)
  //
  //  We remember that the matrix rows _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐤𝐞𝐭(0) and _𝐢𝐧𝐬𝐭𝐎𝐛𝐬_𝐛𝐫𝐚(0) contains
  //  the instantaneous values of the Hamiltonian operator along the MCMC, i.e.
  //  ℰ(𝒗,𝒉) and ℰ(𝒗,𝒉ˈ).
  /*#################################################################################*/
  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝒏𝐨𝐧-𝐝𝐢𝐚𝐠𝐨𝐧𝐚𝐥 𝐚𝐧𝐝 𝐝𝐢𝐚𝐠𝐨𝐧𝐚𝐥 𝐨𝐛𝐬𝐞𝐫𝐯𝐚𝐛𝐥𝐞𝐬.
  //  We compute the stochastic average via the blocking technique of
  //
  //        𝝈ˣ(𝜙,𝛂) = ⟨𝞼ˣ⟩ ≈ ⟨𝜎ˣ⟩             (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //        𝝈ˣ(𝜙,𝛂) = ⟨𝞼ˣ⟩ ≈ ≪𝜎ˣᴿ≫ + ⌈𝜎ˣᴵ⌋   (𝓈ℎ𝒶𝒹ℴ𝓌)
  //
  //  and so on for the others quantum properties.
  //  As regards the properties represented by diagonal operators in the
  //  computational basis, the calculations are easier and no distinction should
  //  be made between the two cases 𝓈ℎ𝒶𝒹ℴ𝓌 or 𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏.
  /*#################################################################################*/

  //MPI variables for parallelization
  int rank,size;
  MPI_Comm_size(common, &size);
  MPI_Comm_rank(common, &rank);

  //Computes non-diagonal quantum properties in each block
  for(int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++) _global_Observables.at(n_obs, 0).zeros(_N_blks);
  if(!_if_shadow){

    for(int n_obs = 0; n_obs < _Observables.n_rows; n_obs++)
      _Observables.at(n_obs, 0) = this -> average_in_blocks(_instObs_ket.row(n_obs));

  }
  else{

    for(int n_obs = 0; n_obs < _Observables.n_rows; n_obs++){

      _Observables.at(n_obs, 0).set_size(_N_blks);
      _Observables.at(n_obs, 0).set_real(this -> Shadow_average_in_blocks(_instObs_ket.row(n_obs), _instObs_bra.row(n_obs)));
      _Observables.at(n_obs, 0).set_imag(zeros(_N_blks));

    }

  }

  //Computes diagonal quantum properties in each block
  if(_if_measure_DIAGONAL_OBS || _if_measure_BLOCK_DIAGONAL_OBS){

    _SzSzCorr.set_size(_N_blks, _instSzSzCorr.n_rows);
    _global_Mz2.zeros(_N_blks);
    _global_Cz_of_r.zeros(_N_blks, _instSzSzCorr.n_rows);

    _squareMag = this -> average_in_blocks(_instSquareMag);
    for(int r = 0; r < _SzSzCorr.n_cols; r++) _SzSzCorr.col(r) = this -> average_in_blocks(_instSzSzCorr.row(r));

  }

  MPI_Barrier(common);

  //Shares block averages among all the nodes
  for(int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++)
    MPI_Reduce(_Observables.at(n_obs, 0).begin(), _global_Observables.at(n_obs, 0).begin(), _N_blks, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
  if(_if_measure_DIAGONAL_OBS || _if_measure_BLOCK_DIAGONAL_OBS){

    MPI_Reduce(_squareMag.begin(), _global_Mz2.begin(), _N_blks, MPI_DOUBLE, MPI_SUM, 0, common);
    MPI_Reduce(_SzSzCorr.begin(), _global_Cz_of_r.begin(), _N_blks * _instSzSzCorr.n_rows, MPI_DOUBLE, MPI_SUM, 0, common);

  }
  if(rank == 0){

    for(int n_obs = 0; n_obs < _global_Observables.n_rows; n_obs++) _global_Observables.at(n_obs, 0) /= double(size);
    if(_if_measure_DIAGONAL_OBS || _if_measure_BLOCK_DIAGONAL_OBS){

      _global_Mz2 /= double(size);
      _global_Cz_of_r /= double(size);

    }

  }

}


vec VMC_Sampler :: compute_errorbar(const vec& block_averages) const {

  /*################################################################*/
  //  Computes the statistical uncertainties of a certain quantity
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
  //  This calculation involves a real-valued quantity.
  /*################################################################*/

  //Function variables
  vec errors(block_averages.n_elem);
  vec squared_block_averages;  // ⟨◦⟩ⱼᵇˡᵏ • ⟨◦⟩ⱼᵇˡᵏ
  double sum_ave, sum_ave_squared;  //Storage variables

  //Block averaging method
  squared_block_averages = block_averages % block_averages;  //Armadillo Schur product
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    sum_ave  = 0.0;
    sum_ave_squared = 0.0;
    for(int j = 0; j < (block_ID + 1); j++){

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


cx_vec VMC_Sampler :: compute_errorbar(const cx_vec& block_averages) const {

  /*################################################################*/
  //  Computes the statistical uncertainties of a certain quantity
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
  //  This calculation involves a complex-valued quantity.
  /*################################################################*/

  //Function variables
  cx_vec errors(block_averages.n_elem);
  vec block_averages_re = real(block_averages);
  vec block_averages_im = imag(block_averages);

  //Block averaging method, keeping real and imaginary part separated
  errors.set_real(compute_errorbar(block_averages_re));
  errors.set_imag(compute_errorbar(block_averages_im));

  return errors;

}


vec VMC_Sampler :: compute_progressive_averages(const vec& block_averages) const {

  /*################################################################*/
  //  Computes the progressive averages of a certain quantity
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
  //  This calculation involves a real-valued quantity.
  /*################################################################*/

  //Function variables
  vec prog_ave(_N_blks);
  double sum_ave;

  //Block averaging
  for(int block_ID = 0; block_ID < _N_blks; block_ID++){

    sum_ave = 0.0;
    for(int j = 0; j < (block_ID + 1); j++) sum_ave += block_averages[j];
    sum_ave /= double(block_ID + 1);
    prog_ave[block_ID] = sum_ave;

  }

  return prog_ave;

}


cx_vec VMC_Sampler :: compute_progressive_averages(const cx_vec& block_averages) const {

  /*################################################################*/
  //  Computes the progressive averages of a certain quantity
  //  by using the “block averaging”, where the argument represents
  //  the set of the single-block Monte Carlo averages ⟨◦⟩ⱼᵇˡᵏ of
  //  that quantity ◦, with j = 𝟣,…,𝖭ᵇˡᵏ.
  //  This calculation involves a complex-valued quantity.
  /*################################################################*/

  //Function variables
  cx_vec prog_ave(block_averages.n_elem);
  vec block_averages_re = real(block_averages);
  vec block_averages_im = imag(block_averages);

  //Block averaging method, keeping real and imaginary part separated
  prog_ave.set_real(compute_progressive_averages(block_averages_re));
  prog_ave.set_imag(compute_progressive_averages(block_averages_im));

  return prog_ave;

}


void VMC_Sampler :: compute_O() {

  //Gives size
  _O.set_size(_vqs.n_alpha(), 2);

  if(!_if_shadow){

    for(int lo_ID = 0; lo_ID < _O.n_rows; lo_ID++){

      _O.at(lo_ID, 0) = this -> average_in_blocks(_instO_ket.row(lo_ID));  // ⟨𝓞ₖ⟩ⱼᵇˡᵏ
      _O.at(lo_ID, 1) = this -> average_in_blocks(conj(_instO_ket.row(lo_ID)));  // ⟨𝓞⋆ₖ⟩ⱼᵇˡᵏ

    }

  }
  else{

    for(int lo_ID = 0; lo_ID < _O.n_rows; lo_ID++){

      //Computes ≪𝓞ₖ≫ⱼᵇˡᵏ
      _O.at(lo_ID, 0).set_size(_N_blks);
      _O.at(lo_ID, 0).set_real(this -> Shadow_angled_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O.at(lo_ID, 0).set_imag(zeros(_N_blks));

      //Computes ⌈𝓞ₖ⌋ⱼᵇˡᵏ
      _O.at(lo_ID, 1).set_size(_N_blks);
      _O.at(lo_ID, 1).set_real(this -> Shadow_square_average_in_blocks(_instO_ket.row(lo_ID), _instO_bra.row(lo_ID)));
      _O.at(lo_ID, 1).set_imag(zeros(_N_blks));

    }

  }

}


void VMC_Sampler :: compute_QGTandGrad(MPI_Comm common, int p) {

  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐐𝐮𝐚𝐧𝐭𝐮𝐦 𝐆𝐞𝐨𝐦𝐞𝐭𝐫𝐢𝐜 𝐓𝐞𝐧𝐬𝐨𝐫.
  //  We compute stochastically the 𝐐𝐆𝐓 defined as
  //
  //        ℚ = 𝙎ₘₙ                                  (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //        𝙎ₘₙ ≈ ⟨𝓞⋆ₘ𝓞ₙ⟩ - ⟨𝓞⋆ₘ⟩•⟨𝓞ₙ⟩.
  //
  //        ℚ = 𝙎 + 𝘼•𝘽•𝘼                            (𝓈ℎ𝒶𝒹ℴ𝓌)
  //        𝙎ₘₙ ≈ ≪𝓞ₘ𝓞ₙ≫ - ≪𝓞ₘ≫•≪𝓞ₙ≫ - ⌈𝓞ₘ⌋⌈𝓞ₙ⌋
  //        𝘼ₘₙ ≈ -⌈𝓞ₘ𝓞ₙ⌋ + ⌈𝓞ₘ⌋≪𝓞ₙ≫ - ≪𝓞ₘ≫⌈𝓞ₙ⌋
  //        where 𝘽 is the inverse matrix of 𝙎.
  /*#################################################################################*/
  /*#################################################################################*/
  //  𝐂𝐨𝐦𝐩𝐮𝐭𝐞𝐬 𝐕𝐌𝐂 𝐄𝐧𝐞𝐫𝐠𝐲 𝐆𝐫𝐚𝐝𝐢𝐞𝐧𝐭.
  //  We compute stochastically the Gradient which drive the optimization defined as
  //
  //        𝔽ₖ ≈ ⟨ℰ𝓞⋆ₖ⟩ - ⟨ℰ⟩•⟨𝓞⋆ₖ⟩                  (𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌)
  //
  //        𝔽ᴿ ≈ 𝞒 - 𝘼•𝘽•𝞨                           (𝓈ℎ𝒶𝒹ℴ𝓌)
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
  int n_alpha = _vqs.n_alpha();
  int blk_size = std::floor(double(_N_sweeps / _N_blks));  //Sets the block length
  cx_mat Q(n_alpha, n_alpha, fill::zeros);
  cx_vec F(n_alpha, fill::zeros);
  _Q.zeros(n_alpha, n_alpha);
  _F.zeros(n_alpha);

  if(!_if_shadow){

    _mean_O.zeros(n_alpha);
    _mean_O_star.zeros(n_alpha);
    cx_vec mean_O(n_alpha);  // ⟨⟨𝓞ₖ⟩ᵇˡᵏ⟩
    cx_vec mean_O_star(n_alpha);  // ⟨⟨𝓞⋆ₖ⟩ᵇˡᵏ⟩
    cx_double block_qgt, block_gradE;

    //Computes 𝐸(𝜙,𝛂) = ⟨Ĥ⟩ stochastically without progressive errorbars
    cx_double E = mean(_Observables.at(0, 0));

    //Computes 𝓞ₖ and 𝓞⋆ₖ stochastically without progressive errorbars
    for(int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O[lo_ID] = mean(_O.at(lo_ID, 0));
      mean_O_star[lo_ID] = mean(_O.at(lo_ID, 1));

    }

    //Computes ℚ = 𝙎ₘₙ stochastically without progressive errorbars
    for(int m = 0; m < n_alpha; m++){

      for(int n = m; n < n_alpha; n++){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          block_qgt = 0.0;
          for(int l = block_ID * blk_size; l < (block_ID +  1) * blk_size; l++)
            block_qgt += std::conj(_instO_ket.at(m, l)) * _instO_ket.at(n, l);  //Accumulate 𝓞⋆ₘ𝓞ₙ in each block
          Q.at(m, n) += block_qgt / double(blk_size);  // ⟨𝙎ₘₙ⟩ᵇˡᵏ
          if(m != n) Q.at(n, m) = std::conj(Q.at(m, n));  //The Quantum Geometric Tensor is in general hermitean

        }

      }

    }
    Q /= double(_N_blks);  // ⟨ℚ⟩ ≈ ⟨⟨𝙎ₘₙ⟩ᵇˡᵏ⟩
    Q = Q - kron(mean_O_star, mean_O.st());

    //Computes 𝔽 = 𝔽ₖ stochastically without progressive errorbars
    for(int k = 0; k < n_alpha; k++){

      for(int block_ID = 0; block_ID < _N_blks; block_ID++){

        block_gradE = 0.0;
        for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++)
          block_gradE += _instObs_ket.at(0, l) * std::conj(_instO_ket.at(k, l));  //Accumulate ℰ𝓞⋆ₖ in each block
        F[k] += block_gradE / double(blk_size);  // ⟨𝔽ₖ⟩ᵇˡᵏ

      }

    }
    F /= double(_N_blks);  // ⟨𝔽⟩ ≈ ⟨⟨𝔽ₖ⟩ᵇˡᵏ⟩
    F = F - E * mean_O_star;

    MPI_Barrier(common);

    //Shares block averages among all the nodes
    MPI_Reduce(mean_O.begin(), _mean_O.begin(), n_alpha, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(mean_O_star.begin(), _mean_O_star.begin(), n_alpha, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(&E, &_E, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(Q.begin(), _Q.begin(), _Q.n_rows * _Q.n_cols, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(F.begin(), _F.begin(), _F.n_elem, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);

    if(rank == 0){

      _mean_O /= double(size);
      _mean_O_star /= double(size);
      _E /= double(size);
      _Q /= double(size);
      _F /= double(size);

    }

  }
  else{

    _mean_O_angled.zeros(n_alpha);
    _mean_O_square.zeros(n_alpha);
    vec mean_O_angled(n_alpha);  // ⟨≪𝓞ₖ≫ᵇˡᵏ⟩ with reweighting correction
    vec mean_O_square(n_alpha);  // ⟨⌈𝓞ₖ⌋ᵇˡᵏ⟩ with reweighting correction
    mat S(n_alpha, n_alpha, fill::zeros);  // 𝙎ₘₙ ≈ ≪𝓞ₘ𝓞ₙ≫ - ≪𝓞ₘ≫•≪𝓞ₙ≫ - ⌈𝓞ₘ⌋⌈𝓞ₙ⌋
    mat A(n_alpha, n_alpha, fill::zeros);  // 𝘼ₘₙ ≈ -⌈𝓞ₘ𝓞ₙ⌋ + ⌈𝓞ₘ⌋≪𝓞ₙ≫ - ≪𝓞ₘ≫⌈𝓞ₙ⌋
    mat AB;
    vec Gamma(n_alpha, fill::zeros);  // 𝞒ₖ ≈ -⟨Ĥ⟩•⌈𝓞ₖ⌋ + ≪𝓞ₖ•ℰᴵ≫ + ⌈𝓞ₖ•ℰᴿ⌋
    vec Omega(n_alpha, fill::zeros);  // 𝞨ₖ ≈ ⟨Ĥ⟩•≪𝓞ₖ≫ + ⌈𝓞ₖ•ℰᴵ⌋ - ≪𝓞ₖ•ℰᴿ≫
    double block_corr_angled, block_corr_square;
    double mean_cos = mean(_cosII);

    for(int lo_ID = 0; lo_ID < n_alpha; lo_ID++){

      mean_O_angled[lo_ID] = mean(real(_O.at(lo_ID, 0))) / mean_cos;
      mean_O_square[lo_ID] = mean(real(_O.at(lo_ID, 1))) / mean_cos;

    }

    //Computes 𝐸(𝜙,𝛂) = ⟨Ĥ⟩ stochastically without progressive errorbars
    cx_double E;
    E.real(mean(real(_Observables.at(0, 0))) / mean_cos);  // ⟨⟨Ĥ⟩ᵇˡᵏ⟩ with reweighting correction
    E.imag(0.0);

    //Computes ℚ = 𝙎 + 𝘼•𝘽•𝘼 stochastically without progressive errorbars
    for(int m = 0; m < n_alpha; m++){

      for(int n = m; n < n_alpha; n++){

        for(int block_ID = 0; block_ID < _N_blks; block_ID++){

          block_corr_angled = 0.0;
          block_corr_square = 0.0;
          for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

            //Accumulate 𝓞ₘ𝓞ₙ in each block (angled part)
            block_corr_angled += _instReweight.at(0, l) * (_instO_ket.at(m, l).real() * _instO_bra.at(n, l).real() + _instO_bra.at(m, l).real() * _instO_ket.at(n, l).real());
            //Accumulate 𝓞ₘ𝓞ₙ in each block (square part)
            if(m != n) block_corr_square += _instReweight.at(1, l) * (_instO_bra.at(m, l).real() * _instO_ket.at(n, l).real() - _instO_ket.at(m, l).real() * _instO_bra.at(n, l).real());

          }
          if(m == n) S.at(m, n) += 0.5 * block_corr_angled / double(blk_size);  //Computes the diagonal elements of S
          else{

            S.at(m, n) += 0.5 * block_corr_angled / double(blk_size);  //This is a symmetric matrix, so we calculate only the upper triangular matrix
            S.at(n, m) = S.at(m, n);
            A.at(m, n) -= 0.5 * block_corr_square / double(blk_size);  //This is an anti-symmetric matrix, so we calculate only the upper triangular matrix
            A.at(n, m) = - A.at(m, n);

          }

        }

      }

    }
    S /= double(_N_blks);  // ⟨⟨≪𝓞ₘ𝓞ₙ≫ᵇˡᵏ⟩⟩ without reweighting correction
    A /= double(_N_blks);  // ⟨⟨⌈𝓞ₘ𝓞ₙ⌋ᵇˡᵏ⟩⟩ without reweighting correction
    S /= mean_cos;
    A /= mean_cos;
    S = S - kron(mean_O_angled, mean_O_angled.t()) + kron(mean_O_square, mean_O_square.t());
    A = A + kron(mean_O_square, mean_O_angled.t()) - kron(mean_O_angled, mean_O_square.t());
    if(_if_QGT_REG){

      if(_regularization_method == 1) AB = A * (S + _eps * _I).i();
      else if(_regularization_method == 2) AB =  A * pinv(S, _lambda);
      else if(_regularization_method == 3) AB = A * (S + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i();
      else if(_regularization_method == 4) AB = A * reg_SVD_inverse(S);

    }
    else AB = A * S.i();
    Q.set_real(symmatu(S + AB * A));  // ⟨ℚ⟩ ≈ ⟨⟨𝙎 + 𝘼•𝘽•𝘼⟩ᵇˡᵏ⟩

    //Computes 𝔽 = {𝔽ᴿ, 𝔽ᴵ} stochastically without progressive errorbars
    for(int k = 0; k < n_alpha; k++){  //Computes ⟨𝞒ₖ⟩ᵇˡᵏ

      for(int block_ID = 0; block_ID < _N_blks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate 𝓞ₖ•ℰᴵ in each block (angled part)
          block_corr_angled += _instReweight.at(0, l) * (_instO_ket.at(k, l).real() * _instObs_bra.at(0, l).imag() + _instO_bra.at(k, l).real() * _instObs_ket.at(0, l).imag());
          //Accumulate 𝓞ₖ•ℰᴿ in each block (square part)
          block_corr_square += _instReweight.at(1, l) * (_instO_bra.at(k, l).real() * _instObs_ket.at(0, l).real() - _instO_ket.at(k, l).real() * _instObs_bra.at(0, l).real());

        }
        Gamma[k] += 0.5 * (block_corr_angled + block_corr_square) / double(blk_size);

      }

    }
    for(int k = 0; k < n_alpha; k++){  //Computes ⟨𝞨ₖ⟩ᵇˡᵏ

      for(int block_ID = 0; block_ID < _N_blks; block_ID++){

        block_corr_angled = 0.0;
        block_corr_square = 0.0;
        for(int l = block_ID * blk_size; l < (block_ID + 1) * blk_size; l++){

          //Accumulate 𝓞ₖ•ℰᴿ in each block (angled part)
          block_corr_angled += _instReweight.at(0, l) * (_instO_ket.at(k, l).real() * _instObs_bra.at(0, l).real() + _instO_bra.at(k, l).real() * _instObs_ket.at(0, l).real());
          //Accumulate 𝓞ₖ•ℰᴵ in each block (square part)
          block_corr_square += _instReweight.at(1, l) * (_instO_bra.at(k, l).real() * _instObs_ket.at(0, l).imag() - _instO_ket.at(k, l).real() * _instObs_bra.at(0, l).imag());

        }
        Omega[k] += 0.5 * (block_corr_square - block_corr_angled) / double(blk_size);

      }

    }
    Gamma /= double(_N_blks);  // ⟨⟨𝞒ₖ⟩ᵇˡᵏ⟩ without reweighting correction
    Omega /= double(_N_blks);  // ⟨⟨𝞨ₖ⟩ᵇˡᵏ⟩ without reweighting correction
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
    MPI_Reduce(&E, &_E, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(Q.begin(), _Q.begin(), _Q.n_rows * _Q.n_cols, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);
    MPI_Reduce(F.begin(), _F.begin(), _F.n_elem, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, common);

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


void VMC_Sampler :: QGT_Check(int rank) {  //Helpful in debugging

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


void VMC_Sampler :: is_asymmetric(const mat& A) const {  //Helpful in debugging

  int failed = 0;

  for(int m = 0; m < A.n_rows; m++){

    for(int n = m; n < A.n_cols; n++)
      if(A.at(m, n) != (-1.0) * A.at(n, m)) failed++;

  }

  if(failed != 0) std::cout << "The matrix is not anti-Symmetric." << std::endl;
  else return;

}


void VMC_Sampler :: Reset_Moves_Statistics() {

  _N_accepted_real = 0;
  _N_proposed_real = 0;
  _N_accepted_ket = 0;
  _N_proposed_ket = 0;
  _N_accepted_bra = 0;
  _N_proposed_bra = 0;
  _N_accepted_equal_site = 0;
  _N_proposed_equal_site = 0;
  _N_accepted_real_nn_site = 0;
  _N_proposed_real_nn_site = 0;
  _N_accepted_shadows_nn_site = 0;
  _N_proposed_shadows_nn_site = 0;
  _N_accepted_global_ket_flip = 0;
  _N_proposed_global_ket_flip = 0;
  _N_accepted_global_bra_flip = 0;
  _N_proposed_global_bra_flip = 0;

}


bool VMC_Sampler :: RandFlips_real(Mat <int>& flipped_site, int Nflips) {

  /*#############################################################################*/
  //  Random spin flips for the 𝓇ℯ𝒶𝑙 quantum degrees of freedom.
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the physical quantum degrees of freedom only.
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  representing it as the list of indeces of the 𝓇ℯ𝒶𝑙 flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    flipped_site.set_size(Nflips, 1);
    for(int j = 0; j < Nflips; j++)
      flipped_site.at(j, 0) = _rnd.Rannyu_INT(0, _L - 1);  //Choose a random spin to flip

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


void VMC_Sampler :: Move_real(int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the 𝓇ℯ𝒶𝑙 variables have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐫𝐞𝐚𝐥 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_real(_flipped_site, Nflips)){

    _N_proposed_real++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _shadow_ket, _flipped_ket_site,
                                             _shadow_bra, _flipped_bra_site,
                                             "real");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_real++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(int fs_row = 0; fs_row < _flipped_site.n_rows; fs_row++){  //Move the quantum spin configuration

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


bool VMC_Sampler :: RandFlips_shadows(Mat <int>& flipped_shadow_site, int Nflips) {

  /*##############################################################################*/
  //  Random spin flips for the 𝓈ℎ𝒶𝒹ℴ𝓌 quantum degrees of freedom (ket or bra).
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the auxiliary quantum degrees of freedom in the ket or bra
  //  configuration only.
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈ⟩
  //                or
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡 𝐡ˈⁿᵉʷ⟩
  //
  //  representing it as the list of indeces of the 𝓈ℎ𝒶𝒹ℴ𝓌 flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  The function prevents from flipping the same site more than once.
  /*##############################################################################*/

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    flipped_shadow_site.set_size(Nflips, 1);
    for(int j = 0; j < Nflips; j++)
      flipped_shadow_site.at(j, 0) = _rnd.Rannyu_INT(0, _n_shadows - 1);  //Choose a random spin to flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_shadow_site);
  if(test.n_elem == flipped_shadow_site.n_rows) return true;
  else return false;

}


void VMC_Sampler :: Move_ket(int Nflips) {

  /*##################################################################*/
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables (ket) have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of auxiliary spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 𝓈ℎ𝒶𝒹ℴ𝓌 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐬𝐡𝐚𝐝𝐨𝐰𝐬 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
  /*##################################################################*/

  if(this -> RandFlips_shadows(_flipped_ket_site, Nflips)){

    _N_proposed_ket++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _shadow_ket, _flipped_ket_site,
                                             _shadow_bra, _flipped_bra_site,
                                             "ket");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_ket++;
      for(int fs_row = 0; fs_row < _flipped_ket_site.n_rows; fs_row++){  //Move the quantum ket configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _shadow_ket.at(0, _flipped_ket_site.at(fs_row, 0)) *= -1;
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


void VMC_Sampler :: Move_bra(int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables (bra) have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of auxiliary spins.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 𝓈ℎ𝒶𝒹ℴ𝓌 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐬𝐡𝐚𝐝𝐨𝐰𝐬 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_shadows(_flipped_bra_site, Nflips)){

    _N_proposed_bra++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _shadow_ket, _flipped_ket_site,
                                             _shadow_bra, _flipped_bra_site,
                                             "bra");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_bra++;
      for(int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum bra configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _shadow_bra.at(0, _flipped_bra_site.at(fs_row, 0)) *= -1;
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


void VMC_Sampler :: Move_equal_site(int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which the 𝓇ℯ𝒶𝑙 and the 𝓈ℎ𝒶𝒹ℴ𝓌 variables have been
  //  tried to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins on
  //  𝐨𝐧 𝐭𝐡𝐞 𝐬𝐚𝐦𝐞 𝐥𝐚𝐭𝐭𝐢𝐜𝐞 𝐬i𝐭𝐞𝐬.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐫𝐞𝐚𝐥 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_real(_flipped_site, Nflips)){

    _N_proposed_equal_site++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _shadow_ket, _flipped_ket_site,
                                             _shadow_bra, _flipped_bra_site,
                                             "equal site");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_equal_site++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(int fs_row = 0; fs_row < _flipped_site.n_rows; fs_row++){  //Move the quantum configuration

        if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

          _configuration.at(0, _flipped_site.at(fs_row, 0)) *= -1;
          _shadow_ket.at(0, _flipped_site.at(fs_row, 0)) *= -1;
          _shadow_bra.at(0, _flipped_site.at(fs_row, 0)) *= -1;

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


bool VMC_Sampler :: RandFlips_real_nn_site(Mat <int>& flipped_real_nn_site, int Nflips) {

  /*#############################################################################*/
  //  Random spin flips for the 𝓇ℯ𝒶𝑙 quantum degrees of freedom.
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the physical quantum degrees of freedom only.
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  representing it as the list of indeces of the 𝓇ℯ𝒶𝑙 flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  If a certain lattice site is selected, 𝐢𝐭𝐬 𝐟𝐢𝐫𝐬𝐭 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫
  //  site it is automatically added to the list of flipped sites.
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Function variables
  int index_site;

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    flipped_real_nn_site.set_size(2 * Nflips, 1);
    for(int j = 0; j < Nflips; j++){

      if(_H.if_PBCs()) index_site = _rnd.Rannyu_INT(0, _L - 1);
      else index_site  = _rnd.Rannyu_INT(0, _L - 2);
      flipped_real_nn_site.at(j, 0) = index_site;  //Choose a random spin to flip

      //Adds the right nearest neighbor lattice site
      if(_H.if_PBCs()){

        if(index_site == _L - 1) flipped_real_nn_site.at(j + 1, 0) = 0;  //Pbc
        else flipped_real_nn_site.at(j + 1, 0) = index_site + 1;

      }
      else flipped_real_nn_site.at(j + 1) = index_site + 1;

    }

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  uvec test = find_unique(flipped_real_nn_site);
  if(test.n_elem == flipped_real_nn_site.n_rows) return true;
  else return false;

}


void VMC_Sampler :: Move_real_nn_site(int Nflips) {

   /*###############################################################*/
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the 𝓇ℯ𝒶𝑙 variables have been tried
  //  to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗ⁿᵉʷ 𝐡 𝐡ˈ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of spins
  //  with their respective 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫 lattice site.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐫𝐞𝐚𝐥_𝐧𝐧_𝐬𝐢𝐭𝐞 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_real_nn_site(_flipped_site, Nflips)){

    _N_proposed_real_nn_site++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _shadow_ket, _flipped_ket_site,
                                             _shadow_bra, _flipped_bra_site,
                                             "real");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_real_nn_site++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(int fs_row = 0; fs_row < _flipped_site.n_rows; fs_row++){  //Move the quantum spin configuration

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


bool VMC_Sampler :: RandFlips_shadows_nn_site(Mat <int>& flipped_ket_site, Mat <int>& flipped_bra_site, int Nflips) {

  /*#############################################################################*/
  //  Random spin flips for the 𝓈ℎ𝒶𝒹ℴ𝓌 quantum degrees of freedom.
  //
  //  This function decides whether or not to do a single spin-flip move
  //  for the 𝓈ℎ𝒶𝒹ℴ𝓌 quantum degrees of freedom only (both ket and bra).
  //  In particular if it returns true, the move is done, otherwise not.
  //  A spin-flip move consists in randomly selecting 𝐍𝐟𝐥𝐢𝐩𝐬 lattice sites
  //  and create a new quantum configuration
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  representing it as the list of indeces of the 𝓈ℎ𝒶𝒹ℴ𝓌 flipped
  //  lattice sites (see 𝐦𝐨𝐝𝐞𝐥.𝐜𝐩𝐩).
  //  If a certain lattice site is selected, 𝐢𝐭𝐬 𝐟𝐢𝐫𝐬𝐭 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫
  //  site it is automatically added to the list of flipped sites.
  //  The function prevents from flipping the same site more than once.
  /*#############################################################################*/

  //Function variables
  int index_site_ket;
  int index_site_bra;

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    flipped_ket_site.set_size(2 * Nflips, 1);
    flipped_bra_site.set_size(2 * Nflips, 1);
    for(int j = 0; j < Nflips; j++){

      if(_H.if_PBCs()){

          index_site_ket = _rnd.Rannyu_INT(0, _L - 1);
          index_site_bra = _rnd.Rannyu_INT(0, _L - 1);

      }
      else{

        index_site_ket  = _rnd.Rannyu_INT(0, _L - 2);
        index_site_bra = _rnd.Rannyu_INT(0, _L - 2);

      }
      flipped_ket_site.at(j, 0) = index_site_ket;  //Choose a random spin to flip
      flipped_bra_site.at(j, 0) = index_site_bra;  //Choose a random spin to flip

      //Adds the right nearest neighbor lattice site
      if(_H.if_PBCs()){

        if(index_site_ket == _L - 1) flipped_ket_site.at(j + 1, 0) = 0;  //Pbc
        if(index_site_bra == _L - 1) flipped_bra_site.at(j + 1, 0) = 0;  //Pbc
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


void VMC_Sampler :: Move_shadows_nn_site(int Nflips) {

  /*################################################################*/
  //  This function proposes a new configuration for the chosen 𝐕𝐐𝐒
  //  in which only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables (both ket and bra)
  //  have been tried to move, i.e.
  //
  //        |𝒮ⁿᵉʷ⟩ = |𝒗 𝐡ⁿᵉʷ 𝐡ˈⁿᵉʷ⟩
  //
  //  by flipping a certain (given) number 𝐍𝐟𝐥𝐢𝐩𝐬 of auxiliary spins
  //  with their respective 𝐫𝐢𝐠𝐡𝐭 𝐧𝐞𝐚𝐫𝐞𝐬𝐭 𝐧𝐞𝐢𝐠𝐡𝐛𝐨𝐫 lattice site.
  //  In particular, it first randomly selects 𝐍𝐟𝐥𝐢𝐩𝐬 𝓈ℎ𝒶𝒹ℴ𝓌 lattice
  //  sites to flip.
  //  If this selection is successful, i.e. if the result of
  //  𝐑𝐚𝐧𝐝𝐅𝐥𝐢𝐩𝐬_𝐬𝐡𝐚𝐝𝐨𝐰𝐬_𝐧𝐧_𝐬𝐢𝐭𝐞 is true, then it decides whether or not
  //  to accept |𝒮ⁿᵉʷ⟩ through the Metropolis-Hastings test.
  /*################################################################*/

  if(this -> RandFlips_shadows_nn_site(_flipped_ket_site, _flipped_bra_site, Nflips)){

    _N_proposed_shadows_nn_site++;
    double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                             _shadow_ket, _flipped_ket_site,
                                             _shadow_bra, _flipped_bra_site,
                                             "braket");
    if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

      _N_accepted_shadows_nn_site++;
      _vqs.Update_on_Config(_configuration, _flipped_site);
      for(int fs_row = 0; fs_row < _flipped_ket_site.n_rows; fs_row++){  //Move the quantum ket configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _shadow_ket.at(0, _flipped_ket_site.at(fs_row, 0)) *= -1;
        else{  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟐

          /*
            .........
            .........
            .........
          */

        }

      }

      for(int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum bra configuration

        if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
          _shadow_bra.at(0, _flipped_bra_site.at(fs_row, 0)) *= -1;
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


void VMC_Sampler :: Move_global_ket_flip() {

  _N_proposed_global_ket_flip++;

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    _flipped_ket_site.set_size(_L, 1);
    for(int j = 0; j < _L; j++) _flipped_ket_site.at(j, 0) = j;  //Global spin-flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                           _shadow_ket, _flipped_ket_site,
                                           _shadow_bra, _flipped_bra_site,
                                           "ket");
  if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

    _N_accepted_global_ket_flip++;
    _vqs.Update_on_Config(_configuration, _flipped_site);
    for(int fs_row = 0; fs_row < _flipped_ket_site.n_rows; fs_row++){  //Move the quantum ket configuration

      if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
        _shadow_ket.at(0, _flipped_ket_site.at(fs_row, 0)) *= -1;
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


void VMC_Sampler :: Move_global_bra_flip() {

  _N_proposed_global_bra_flip++;

  //Initializes the new configuration according to |𝚲|
  if(_H.dimensionality() == 1){  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏

    _flipped_bra_site.set_size(_L, 1);
    for(int j = 0; j < _L; j++) _flipped_bra_site.at(j, 0) = j;  //Global spin-flip

  }
  else{  //𝚲 ϵ ℤᵈ, 𝖽 = 2

    /*
      ..........
      ..........
      ..........
    */

  }

  double p = _vqs.PMetroNew_over_PMetroOld(_configuration, _flipped_site,
                                           _shadow_ket, _flipped_ket_site,
                                           _shadow_bra, _flipped_bra_site,
                                           "bra");
  if(_rnd.Rannyu() < p){  //Metropolis-Hastings test

    _N_accepted_global_bra_flip++;
    _vqs.Update_on_Config(_configuration, _flipped_site);
    for(int fs_row = 0; fs_row < _flipped_bra_site.n_rows; fs_row++){  //Move the quantum ket configuration

      if(_H.dimensionality() == 1)  //𝚲 ϵ ℤᵈ, 𝖽 = 𝟏
        _shadow_bra.at(0, _flipped_bra_site.at(fs_row, 0)) *= -1;
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
  if(
     _p_equal_site < 0 || _p_equal_site > 1 || _p_real_nn < 0 || _p_real_nn > 1 || _p_shadow_nn < 0 || _p_shadow_nn > 1 ||
     _p_global_ket_flip < 0 || _p_global_ket_flip > 1 || _p_global_bra_flip < 0 || _p_global_bra_flip > 1
    ){

    std::cerr << " ##ValueError: the function options MUST be a probability!" << std::endl;
    std::cerr << "   Failed to move the system configuration." << std::endl;
    std::abort();

  }

  //Moves with a certain probability
  this -> Move_real(_N_flips);
  if(_if_shadow == true && _if_shadow_off == false){

    this -> Move_ket(_N_flips);
    this -> Move_bra(_N_flips);
    if(_rnd.Rannyu() < _p_equal_site) this -> Move_equal_site(_N_flips);
    if(_rnd.Rannyu() < _p_shadow_nn) this -> Move_shadows_nn_site(_N_flips);
    if(_rnd.Rannyu() < _p_global_ket_flip) this -> Move_global_ket_flip();
    if(_rnd.Rannyu() < _p_global_bra_flip) this -> Move_global_bra_flip();

  }
  if(_rnd.Rannyu() < _p_real_nn) this -> Move_real_nn_site(_N_flips);

}


void VMC_Sampler :: Make_Sweep() {

  for(int n_bunch = 0; n_bunch < _M; n_bunch++) this -> Move();

}


void VMC_Sampler :: tVMC_Step(MPI_Comm common, int p) {

  /*###############################################################################################*/
  //  Runs the single evolution step of the quantum state.
  //  We perform the single Variational Monte Carlo run using
  //  the following parameters:
  //
  //        • N̲ˢ̲ʷ̲ᵉ̲ᵉ̲ᵖ̲: is the number of Monte Carlo sweeps.
  //                  In each single 𝐌𝐂 sweep a bunch of spins is considered,
  //                  randomly chosen and whose dimension is expressed by the variable N̲ᶠ̲ˡ̲ⁱ̲ᵖ̲ˢ̲,
  //                  and it is tried to flip this bunch of spins with the probability defined
  //                  by the Metropolis-Hastings algorithm; this operation is repeated a certain
  //                  number of times in the single sweep, where this certain number is defined
  //                  by the variables M̲; once the new proposed configuration is accepted or not,
  //                  instantaneous quantum properties are measured on that state, and the single
  //                  sweep ends; different Monte Carlo moves are applied in different situations,
  //                  involving all or only some of the 𝓇ℯ𝒶𝑙 and/or 𝓈ℎ𝒶𝒹ℴ𝓌 variables;
  //
  //        • e̲q̲ᵗ̲ⁱ̲ᵐ̲ᵉ̲: is the number of Monte Carlo steps, i.e. the number
  //                  of sweeps to be employed in the thermalization phase
  //                  of the system (i.e., the phase in which new quantum
  //                  configurations are sampled but nothing is measured;
  //
  //        • N̲ᵇ̲ˡ̲ᵏ̲ˢ̲: is the number of blocks to be used in the estimation of the
  //                 Monte Carlo quantum averages and uncertainties of the observables
  //                 via the blocking method;
  //
  //  The single 𝐕𝐌𝐂 run allows us to move a single step in the variational
  //  parameter evolution scheme.
  /*###############################################################################################*/

  //MPI variables for parallelization
  int rank;
  MPI_Comm_rank(common, &rank);

  //Initialization and Equilibration
  if(_if_restart_from_config) this -> Init_Config(_configuration, _shadow_ket, _shadow_bra);
  else this -> Init_Config();
  for(int eq_step = 0; eq_step < _N_eq; eq_step++) this -> Make_Sweep();

  //Monte Carlo measurement
  for(int mcmc_step = 0; mcmc_step < _N_sweeps; mcmc_step++){  //The Monte Carlo Markov Chain

    this -> Make_Sweep();  //Samples a new system configuration |𝒮ⁿᵉʷ⟩ (i.e. a new point of the mcmc)
    this -> Measure();  //Measure quantum properties on the new sampled system configuration |𝒮ⁿᵉʷ⟩
    this -> write_MCMC_Config(mcmc_step, rank);  //Records the sampled |𝒮ⁿᵉʷ⟩

  }

  //Computes the quantum averages
  this -> Estimate(common, p);

}


void VMC_Sampler :: Euler(MPI_Comm common, int p) {

  /*#########################################################################*/
  //  Updates the variational parameters (𝜙,𝛂) according to the choosen
  //  𝐭𝐕𝐌𝐂 equations of motion through the Euler integration method.
  //  The equations for the parameters dynamics are:
  //
  //        ==================
  //          𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌
  //        ==================
  //          • 𝐈𝐦𝐚𝐠𝐢𝐧𝐚𝐫𝐲-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝒊-𝐭𝐕𝐌𝐂)
  //              𝕊(τ)•𝛂̇(τ) = - 𝔽(τ)
  //              ϕ̇(τ) = - 𝛂̇(τ) • ⟨𝓞⟩ - ⟨ℰᴿ⟩
  //          • 𝐑𝐞𝐚𝐥-𝐭𝐢𝐦𝐞 𝐝𝐲𝐧𝐚𝐦𝐢𝐜𝐬 (𝐭𝐕𝐌𝐂)
  //              𝕊(𝑡)•𝛂̇(𝑡) =  - 𝑖 • 𝔽(𝑡)
  //              ϕ̇(𝑡) = - 𝛂̇(𝑡) • ⟨𝓞⟩ - 𝑖 • ⟨ℰᴿ⟩
  //
  //        ============
  //          𝓈ℎ𝒶𝒹ℴ𝓌
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
  //  In the Euler method we obtain the new parameters in the following way:
  //  𝒾𝒻
  //
  //        α̇(𝑡) = 𝒻{α(𝑡)}
  //
  //  𝓉𝒽ℯ𝓃
  //
  //        α(𝑡+𝑑𝑡) = α(𝑡) + 𝑑𝑡 • 𝒻{α(𝑡)}
  //
  //  where 𝒻{α(𝑡)} is numerically integrated by inversion of the Quantum
  //  Geometric Tensor.
  /*#########################################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

    //Function variables
    cx_vec new_alpha(_vqs.n_alpha());  // α(𝑡+𝑑𝑡)
    cx_double new_phi;  // 𝜙(𝑡+𝑑𝑡)

    //Solves the 𝐭𝐕𝐌𝐂 equations of motion
    if(rank == 0){

        /*################*/
       /*  𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌  */
      /*################*/
      if(!_if_shadow){

        //Function variables
        cx_vec alpha_dot;  // 𝜶̇
        cx_double phi_dot;  //ϕ̇

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) alpha_dot = - _i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) alpha_dot = - _i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) alpha_dot = - _i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) alpha_dot = -_i * reg_SVD_inverse(_Q) * _F;

          }
          else alpha_dot = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_dot = - _i * _E.real();
            for(int k = 0; k < alpha_dot.n_elem; k++) phi_dot -= alpha_dot[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) alpha_dot = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) alpha_dot = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) alpha_dot = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) alpha_dot = - reg_SVD_inverse(_Q) * _F;

          }
          else alpha_dot = - _Q.i() * _F;

          if(_if_phi){

            phi_dot = - _E.real();
            for(int k = 0; k < alpha_dot.n_elem; k++) phi_dot -= alpha_dot[k] * _mean_O[k];

          }

        }

        //Updates the variational parameters
        new_alpha = _vqs.alpha() + _delta * alpha_dot;  // α(𝑡+𝑑𝑡) = α(𝑡) + 𝑑𝑡 • α̇(𝑡)
        if(_if_phi) new_phi = _vqs.phi() + _delta * phi_dot;

      }

        /*############*/
       /*  𝓈ℎ𝒶𝒹ℴ𝓌  */
      /*############*/
      else{

        //Function variables
        vec alpha_dot_re;  // 𝜶̇ᴿ
        vec alpha_dot_im;  // 𝜶̇ᴵ
        double phi_dot_re = 0.0;  // ϕ̇ᴿ
        double phi_dot_im = 0.0;  // ϕ̇ᴵ

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              alpha_dot_re = (real(_Q) + _eps * _I).i() * real(_F);
              alpha_dot_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              alpha_dot_re = pinv(real(_Q), _lambda) * real(_F);
              alpha_dot_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              alpha_dot_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              alpha_dot_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              alpha_dot_re = reg_SVD_inverse(real(_Q)) * real(_F);
              alpha_dot_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            alpha_dot_re = (real(_Q)).i() * real(_F);
            alpha_dot_im = (real(_Q)).i() * imag(_F);

          }
          if(_if_phi){

            phi_dot_im = - _E.real();
            for(int k = 0; k < alpha_dot_re.n_elem; k++){

              phi_dot_re += - alpha_dot_re[k] * _mean_O_angled[k] - alpha_dot_im[k] * _mean_O_square[k];
              phi_dot_im += alpha_dot_re[k] * _mean_O_square[k] - alpha_dot_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              alpha_dot_re = (real(_Q) + _eps * _I).i() * imag(_F);
              alpha_dot_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              alpha_dot_re = pinv(real(_Q), _lambda) * imag(_F);
              alpha_dot_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              alpha_dot_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              alpha_dot_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              alpha_dot_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              alpha_dot_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            alpha_dot_re = (real(_Q)).i() * imag(_F);
            alpha_dot_im = - (real(_Q)).i() * real(_F);

          }
          if(_if_phi){

            phi_dot_re = - _E.real();
            for(int k = 0; k < alpha_dot_re.n_elem; k++){

              phi_dot_re += - alpha_dot_re[k] * _mean_O_angled[k] - alpha_dot_im[k] * _mean_O_square[k];
              phi_dot_im += alpha_dot_re[k] * _mean_O_square[k] - alpha_dot_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(real(_vqs.alpha()) + _delta * alpha_dot_re);
        new_alpha.set_imag(imag(_vqs.alpha()) + _delta * alpha_dot_im);
        if(_if_phi){

          new_phi.real(_vqs.phi().real() + _delta * phi_dot_re);
          new_phi.imag(_vqs.phi().imag() + _delta * phi_dot_im);

        }

      }

    }

    MPI_Barrier(common);

    //Updates parameters of all the nodes in the communicator
    MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
    _vqs.set_alpha(new_alpha);
    if(_if_phi){

      MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_phi(new_phi);

    }

  }
  else return;

}


void VMC_Sampler :: Heun(MPI_Comm common, int p) {

  /*###############################################################*/
  //  The Heun method is a so-called predictor-corrector method,
  //  which achieves a second order accuracy.
  //  In the Heun method we first obtain the auxiliary updates
  //  of the variational parameters
  //
  //        𝛂̃(𝑡 + δₜ) = 𝛂(𝑡) + δₜ•𝒻{α(𝑡)}
  //
  //  as in the Euler method. We remember that
  //
  //        α̇(𝑡) = 𝒻{α(𝑡)}.
  //
  //  These updates are used to performed a second evolution
  //  step via the 𝐭𝐕𝐌𝐂_𝐒𝐭𝐞𝐩() function, and then obtained a second
  //  order updates as
  //
  //        𝛂(𝑡 + δₜ) = 𝛂(𝑡) + 1/2•δₜ•[𝒻{α(𝑡)} + f{𝛂̃(𝑡 + δₜ)}].
  //
  //  The first 𝐭𝐕𝐌𝐂 step in this integration is performed in the
  //  main program.
  /*###############################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      cx_vec alpha_t = _vqs.alpha();  // 𝛂(𝑡)
      cx_vec alpha_dot_t;  // α̇(𝑡) = 𝒻{α(𝑡)}
      cx_vec alpha_dot_tilde_t;  // f{𝛂̃(𝑡 + δₜ)}
      cx_vec new_alpha(_vqs.n_alpha());
      cx_double phi_t = _vqs.phi();  // ϕ(𝑡)
      cx_double phi_dot_t;
      cx_double phi_dot_tilde_t;
      cx_double new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) alpha_dot_t = - _i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) alpha_dot_t = - _i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) alpha_dot_t = _i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) alpha_dot_t = _i * reg_SVD_inverse(_Q) * _F;

          }
          else alpha_dot_t = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_dot_t = - _i * _E.real();
            for(int k = 0; k < alpha_dot_t.n_elem; k++) phi_dot_t -= alpha_dot_t[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) alpha_dot_t = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) alpha_dot_t = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) alpha_dot_t = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) alpha_dot_t = - reg_SVD_inverse(_Q) * _F;

          }
          else alpha_dot_t = - _Q.i() * _F;

          if(_if_phi){

            phi_dot_t = - _E.real();
            for(int k = 0; k < alpha_dot_t.n_elem; k++) phi_dot_t -= alpha_dot_t[k] * _mean_O[k];

          }

        }

        //Updates the variational parameters
        new_alpha = alpha_t + _delta * alpha_dot_t;
        if(_if_phi) new_phi = phi_t + _delta * phi_dot_t;

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐭𝐕𝐌𝐂 step at time 𝑡 + δₜ
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) alpha_dot_tilde_t = - _i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) alpha_dot_tilde_t = - _i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) alpha_dot_tilde_t = - _i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) alpha_dot_tilde_t = - _i * reg_SVD_inverse(_Q) * _F;

          }
          else alpha_dot_tilde_t = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_dot_tilde_t = - _i * _E.real();
            for(int k = 0; k < alpha_dot_tilde_t.n_elem; k++) phi_dot_tilde_t -= alpha_dot_tilde_t[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) alpha_dot_tilde_t = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) alpha_dot_tilde_t = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) alpha_dot_tilde_t = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) alpha_dot_tilde_t = - reg_SVD_inverse(_Q) * _F;

          }
          else alpha_dot_tilde_t = - _Q.i() * _F;

          if(_if_phi){

            phi_dot_tilde_t = - _E.real();
            for(int k = 0; k < alpha_dot_tilde_t.n_elem; k++) phi_dot_tilde_t -= alpha_dot_tilde_t[k] * _mean_O[k];

          }

        }

        //Final update of the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * (alpha_dot_t + alpha_dot_tilde_t);  // 𝛂(𝑡 + δₜ)
        if(_if_phi) new_phi = phi_t + 0.5 * _delta * (phi_dot_t + phi_dot_tilde_t);

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

      /*############*/
     /*  𝓈ℎ𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // 𝜙ᴿ(𝑡)
      double phi_t_im = _vqs.phi().imag();  // 𝜙ᴵ(𝑡)
      vec alpha_t_re = real(_vqs.alpha());  // 𝛂ᴿ(𝑡)
      vec alpha_t_im = imag(_vqs.alpha());  // 𝛂ᴵ(𝑡)
      vec alpha_dot_t_re;  // α̇ᴿ(𝑡) = 𝒻{αᴿ(𝑡)}
      vec alpha_dot_t_im;  // α̇ᴵ(𝑡) = 𝒻{αᴵ(𝑡)}
      cx_vec new_alpha(_vqs.n_alpha());
      double phi_dot_t_re = 0.0;  // 𝜙̇ᴿ(𝑡)
      double phi_dot_t_im = 0.0;  // 𝜙̇ᴵ(𝑡)
      vec alpha_dot_tilde_t_re;
      vec alpha_dot_tilde_t_im;
      double phi_dot_tilde_re = 0.0;
      double phi_dot_tilde_im = 0.0;
      cx_double new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              alpha_dot_t_re = (real(_Q) + _eps * _I).i() * real(_F);
              alpha_dot_t_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              alpha_dot_t_re = pinv(real(_Q), _lambda) * real(_F);
              alpha_dot_t_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              alpha_dot_t_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              alpha_dot_t_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              alpha_dot_t_re = reg_SVD_inverse(real(_Q)) * real(_F);
              alpha_dot_t_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            alpha_dot_t_re = real(_Q).i() * real(_F);
            alpha_dot_t_im = real(_Q).i() * imag(_F);

          }

          if(_if_phi){

            phi_dot_t_im = - _E.real();
            for(int k = 0; k < alpha_dot_t_re.n_elem; k++){

              phi_dot_t_re += - alpha_dot_t_re[k] * _mean_O_angled[k] - alpha_dot_t_im[k] * _mean_O_square[k];
              phi_dot_t_im += alpha_dot_t_re[k] * _mean_O_square[k] - alpha_dot_t_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              alpha_dot_t_re = (real(_Q) + _eps * _I).i() * imag(_F);
              alpha_dot_t_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              alpha_dot_t_re = pinv(real(_Q), _lambda) * imag(_F);
              alpha_dot_t_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              alpha_dot_t_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              alpha_dot_t_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              alpha_dot_t_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              alpha_dot_t_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            alpha_dot_t_re = real(_Q).i() * imag(_F);
            alpha_dot_t_im = - real(_Q).i() * real(_F);

          }

          if(_if_phi){

            phi_dot_t_re = - _E.real();
            for(int k = 0; k < alpha_dot_t_re.n_elem; k++){

              phi_dot_t_re += - alpha_dot_t_re[k] * _mean_O_angled[k] - alpha_dot_t_im[k] * _mean_O_square[k];
              phi_dot_t_im += alpha_dot_t_re[k] * _mean_O_square[k] - alpha_dot_t_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + _delta * alpha_dot_t_re);
        new_alpha.set_imag(alpha_t_im + _delta * alpha_dot_t_im);
        if(_if_phi){

          new_phi.real(phi_t_re + _delta * phi_dot_t_re);
          new_phi.imag(phi_t_im + _delta * phi_dot_t_im);

        }

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐭𝐕𝐌𝐂 step at time 𝑡 + δₜ
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              alpha_dot_tilde_t_re = (real(_Q) + _eps * _I).i() * real(_F);
              alpha_dot_tilde_t_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              alpha_dot_tilde_t_re = pinv(real(_Q), _lambda) * real(_F);
              alpha_dot_tilde_t_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              alpha_dot_tilde_t_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              alpha_dot_tilde_t_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              alpha_dot_tilde_t_re = reg_SVD_inverse(real(_Q)) * real(_F);
              alpha_dot_tilde_t_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            alpha_dot_tilde_t_re = real(_Q).i() * real(_F);
            alpha_dot_tilde_t_im = real(_Q).i() * imag(_F);

          }

          if(_if_phi){

            phi_dot_tilde_im = - _E.real();
            for(int k = 0; k < alpha_dot_tilde_t_re.n_elem; k++){

              phi_dot_tilde_re += - alpha_dot_tilde_t_re[k] * _mean_O_angled[k] - alpha_dot_tilde_t_im[k] * _mean_O_square[k];
              phi_dot_tilde_im += alpha_dot_tilde_t_re[k] * _mean_O_square[k] - alpha_dot_tilde_t_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method ==  1){

              alpha_dot_tilde_t_re = (real(_Q) + _eps * _I).i() * imag(_F);
              alpha_dot_tilde_t_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              alpha_dot_tilde_t_re = pinv(real(_Q), _lambda) * imag(_F);
              alpha_dot_tilde_t_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              alpha_dot_tilde_t_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              alpha_dot_tilde_t_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              alpha_dot_tilde_t_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              alpha_dot_tilde_t_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            alpha_dot_tilde_t_re = real(_Q).i() * imag(_F);
            alpha_dot_tilde_t_im = - real(_Q).i() * real(_F);

          }

          if(_if_phi){

            phi_dot_tilde_re = - _E.real();
            for(int k = 0; k < alpha_dot_tilde_t_re.n_elem; k++){

              phi_dot_tilde_re += - alpha_dot_tilde_t_re[k] * _mean_O_angled[k] - alpha_dot_tilde_t_im[k] * _mean_O_square[k];
              phi_dot_tilde_im += alpha_dot_tilde_t_re[k] * _mean_O_square[k] - alpha_dot_tilde_t_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Final update of the variational parameters
        new_alpha.set_real(alpha_t_re + 0.5 * _delta * (alpha_dot_t_re + alpha_dot_tilde_t_re));
        new_alpha.set_imag(alpha_t_im + 0.5 * _delta * (alpha_dot_t_im + alpha_dot_tilde_t_im));
        if(_if_phi){

          new_phi.real(phi_t_re + 0.5 * _delta * (phi_dot_t_re + phi_dot_tilde_re));
          new_phi.imag(phi_t_im + 0.5 * _delta * (phi_dot_t_im + phi_dot_tilde_im));

        }

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

  }
  else return;

}


void VMC_Sampler :: RK4(MPI_Comm common, int p) {

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
  //  The first 𝐭𝐕𝐌𝐂 step in this integration is performed in the main program.
  /*############################################################################*/

  if(!_if_vmc){

    //MPI variables for parallelization
    int rank;
    MPI_Comm_rank(common, &rank);

      /*################*/
     /*  𝓃ℴ𝓃-𝓈ℎ𝒶𝒹ℴ𝓌  */
    /*################*/
    if(!_if_shadow){

      //Function variables
      cx_vec alpha_t = _vqs.alpha();  // 𝛂(𝑡)
      cx_double phi_t = _vqs.phi();  // 𝜙(𝑡)
      cx_vec k1;  // κ𝟣 = 𝒻{α(𝑡)}
      cx_vec k2;  // κ𝟤 = 𝒻{α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣}
      cx_vec k3;  // κ𝟥 = 𝒻{α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤}
      cx_vec k4;  // κ𝟦 = 𝒻{α(𝑡) + 𝛿ₜ•κ𝟥}
      cx_vec new_alpha(_vqs.n_alpha());  //Storage variable for the set of 𝛂 at one of the 4th Runge-Kutta step
      cx_double phi_k1, phi_k2, phi_k3, phi_k4;
      cx_double new_phi;  //Storage variable for the global phase 𝜙 at one of the 4th Runge-Kutta step

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k1 = - _i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k1 = - _i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k1 = - _i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k1 = - _i * reg_SVD_inverse(_Q) * _F;

          }
          else k1 = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_k1 = - _i * _E.real();
            for(int k = 0; k < k1.n_elem; k++) phi_k1 -= k1[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k1 = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k1 = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k1 = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k1 = - reg_SVD_inverse(_Q) * _F;

          }
          else k1 = - _Q.i() * _F;

          if(_if_phi){

            phi_k1 = - _E.real();
            for(int k = 0; k < k1.n_elem; k++) phi_k1 -= k1[k] * _mean_O[k];

          }

        }

        //Updates the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * k1;  // α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣
        if(_if_phi) new_phi = phi_t + 0.5 * _delta * phi_k1;

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐭𝐕𝐌𝐂 step with parameters α(𝑡) → α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k2 = -_i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k2 = -_i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k2 = -_i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k2 = -_i * reg_SVD_inverse(_Q) * _F;

          }
          else k2 = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_k2 = - _i * _E.real();
            for(int k = 0; k < k2.n_elem; k++) phi_k2 -= k2[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k2 = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k2 = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k2 = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k2 = - reg_SVD_inverse(_Q) * _F;

          }
          else k2 = - _Q.i() * _F;

          if(_if_phi){

            phi_k2 = - _E.real();
            for(int k = 0; k < k2.n_elem; k++) phi_k2 -= k2[k] * _mean_O[k];

          }

        }

        //Updates the variational parameters
        new_alpha = alpha_t + 0.5 * _delta * k2;  // α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤
        if(_if_phi) new_phi = phi_t + 0.5 * _delta * phi_k2;

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /**************/
      /* THIRD STEP */
      /**************/
      //Makes a second 𝐭𝐕𝐌𝐂 step with parameters α(𝑡) → α(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k3 = - _i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k3 = - _i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k3 = - _i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k3 = - _i * reg_SVD_inverse(_Q) * _F;

          }
          else k3 = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_k3 = - _i * _E.real();
            for(int k = 0; k < k3.n_elem; k++) phi_k3 -= k3[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k3 = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k3 = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k3 = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k3 = - reg_SVD_inverse(_Q) * _F;

          }
          else k3 = - _Q.i() * _F;

          if(_if_phi){

            phi_k3 = - _E.real();
            for(int k = 0; k < k3.n_elem; k++) phi_k3 -= k3[k] * _mean_O[k];

          }

        }

        //Updates the variational parameters
        new_alpha = alpha_t + _delta * k3;  // α(𝑡) + 𝛿ₜ•κ𝟥
        if(_if_phi) new_phi = phi_t + _delta * phi_k3;

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* FOURTH STEP */
      /***************/
      //Makes a second 𝐭𝐕𝐌𝐂 step with parameters α(𝑡) → α(𝑡) + 𝛿ₜ•κ𝟥
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k4 = - _i * (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k4 = - _i * pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k4 = - _i * (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k4 = - _i * reg_SVD_inverse(_Q) * _F;

          }
          else k4 = - _i * _Q.i() * _F;

          if(_if_phi){

            phi_k4 = - _i * _E.real();
            for(int k = 0; k < k4.n_elem; k++) phi_k4 -= k4[k] * _mean_O[k];

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1) k4 = - (_Q + _eps * _I).i() * _F;
            else if(_regularization_method == 2) k4 = - pinv(_Q, _lambda) * _F;
            else if(_regularization_method == 3) k4 = - (_Q + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * _F;
            else if(_regularization_method == 4) k4 = - reg_SVD_inverse(_Q) * _F;

          }
          else k4 = - _Q.i() * _F;

          if(_if_phi){

            phi_k4 = - _E.real();
            for(int k = 0; k < k4.n_elem; k++) phi_k4 -= k4[k] * _mean_O[k];

          }

        }

        //Final update of the variational parameters
        new_alpha = alpha_t + _delta * ((1.0 / 6.0) * k1 + (1.0 / 3.0) * k2 + (1.0 / 3.0) * k3 + (1.0 / 6.0) * k4);  // αₖ(𝑡 + 𝛿ₜ)
        if(_if_phi) new_phi = phi_t + _delta * ((1.0 / 6.0) * phi_k1 + (1.0 / 3.0) * phi_k2 + (1.0 / 3.0) * phi_k3 + (1.0 / 6.0) * phi_k4);

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

      /*############*/
     /*  𝓈ℎ𝒶𝒹ℴ𝓌  */
    /*############*/
    else{

      //Function variables
      double phi_t_re = _vqs.phi().real();  // 𝜙ᴿ(𝑡)
      double phi_t_im = _vqs.phi().imag();  // 𝜙ᴵ(𝑡)
      vec alpha_t_re = real(_vqs.alpha());  // 𝛂ᴿ(𝑡)
      vec alpha_t_im = imag(_vqs.alpha());  // 𝛂ᴵ(𝑡)
      vec k1_re;  // κ𝟣ᴿ = 𝒻{αᴿ(𝑡)}
      vec k1_im;  // κ𝟣ᴵ = 𝒻{αᴵ(𝑡)}
      vec k2_re;  // κ𝟤ᴿ = 𝒻{αᴿ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣ᴿ}
      vec k2_im;  // κ𝟤ᴵ = 𝒻{αᴵ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟣ᴵ}
      vec k3_re;  // κ𝟥ᴿ = 𝒻{αᴿ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤ᴿ}
      vec k3_im;  // κ𝟥ᴵ = 𝒻{αᴵ(𝑡) + 𝟣/𝟤•𝛿ₜ•κ𝟤ᴵ}
      vec k4_re;  // κ𝟦ᴿ = 𝒻{αᴿ(𝑡) + 𝛿ₜ•κ𝟥ᴿ}
      vec k4_im;  // κ𝟦ᴵ = 𝒻{αᴵ(𝑡) + 𝛿ₜ•κ𝟥ᴵ}
      cx_vec new_alpha(_vqs.n_alpha());
      double phi_k1_re = 0.0, phi_k2_re = 0.0, phi_k3_re = 0.0, phi_k4_re = 0.0;
      double phi_k1_im = 0.0, phi_k2_im = 0.0, phi_k3_im = 0.0, phi_k4_im = 0.0;
      cx_double new_phi;

      /**************/
      /* FIRST STEP */
      /**************/
      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k1_re = (real(_Q) + _eps * _I).i() * real(_F);
              k1_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              k1_re = pinv(real(_Q), _lambda) * real(_F);
              k1_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              k1_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              k1_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              k1_re = reg_SVD_inverse(real(_Q)) * real(_F);
              k1_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            k1_re = (real(_Q)).i() * real(_F);
            k1_im = (real(_Q)).i() * imag(_F);

          }

          if(_if_phi){

            phi_k1_im = - _E.real();
            for(int k = 0; k < k1_re.n_elem; k++){

              phi_k1_re += - k1_re[k] * _mean_O_angled[k] - k1_im[k] * _mean_O_square[k];
              phi_k1_im += k1_re[k] * _mean_O_square[k] - k1_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k1_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k1_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              k1_re = pinv(real(_Q), _lambda) * imag(_F);
              k1_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              k1_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              k1_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              k1_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              k1_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            k1_re = (real(_Q)).i() * imag(_F);
            k1_im = - (real(_Q)).i() * real(_F);

          }

          if(_if_phi){

            phi_k1_re = - _E.real();
            for(int k = 0; k < k1_re.n_elem; k++){

              phi_k1_re += - k1_re[k] * _mean_O_angled[k] - k1_im[k] * _mean_O_square[k];
              phi_k1_im += k1_re[k] * _mean_O_square[k] - k1_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + 0.5 * _delta * k1_re);
        new_alpha.set_imag(alpha_t_im + 0.5 * _delta * k1_im);
        if(_if_phi){

          new_phi.real(phi_t_re + 0.5 * _delta * phi_k1_re);
          new_phi.imag(phi_t_im + 0.5 * _delta * phi_k1_im);

        }

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* SECOND STEP */
      /***************/
      //Makes a second 𝐭𝐕𝐌𝐂 step at time 𝑡 + δₜ
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k2_re = (real(_Q) + _eps * _I).i() * real(_F);
              k2_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              k2_re = pinv(real(_Q), _lambda) * real(_F);
              k2_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              k2_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              k2_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              k2_re = reg_SVD_inverse(real(_Q)) * real(_F);
              k2_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            k2_re = (real(_Q)).i() * real(_F);
            k2_im = (real(_Q)).i() * imag(_F);

          }
          if(_if_phi){

            phi_k2_im = - _E.real();
            for(int k = 0; k < k2_re.n_elem; k++){

              phi_k2_re += - k2_re[k] * _mean_O_angled[k] - k2_im[k] * _mean_O_square[k];
              phi_k2_im += k2_re[k] * _mean_O_square[k] - k2_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k2_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k2_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              k2_re = pinv(real(_Q), _lambda) * imag(_F);
              k2_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              k2_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              k2_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              k2_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              k2_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            k2_re = (real(_Q)).i() * imag(_F);
            k2_im = - (real(_Q)).i() * real(_F);

          }
          if(_if_phi){

            phi_k2_re = - _E.real();
            for(int k = 0; k < k2_re.n_elem; k++){

              phi_k2_re += - k2_re[k] * _mean_O_angled[k] - k2_im[k] * _mean_O_square[k];
              phi_k2_im += k2_re[k] * _mean_O_square[k] - k2_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + 0.5 * _delta * k2_re);
        new_alpha.set_imag(alpha_t_im + 0.5 * _delta * k2_im);
        if(_if_phi){

          new_phi.real(phi_t_re + 0.5 * _delta * phi_k2_re);
          new_phi.imag(phi_t_im + 0.5 * _delta * phi_k2_re);

        }

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /**************/
      /* THIRD STEP */
      /**************/
      //Makes a second 𝐭𝐕𝐌𝐂 step at time 𝑡 + δₜ
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k3_re = (real(_Q) + _eps * _I).i() * real(_F);
              k3_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              k3_re = pinv(real(_Q), _lambda) * real(_F);
              k3_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              k3_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              k3_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              k3_re = reg_SVD_inverse(real(_Q)) * real(_F);
              k3_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            k3_re = (real(_Q)).i() * real(_F);
            k3_im = (real(_Q)).i() * imag(_F);

          }
          if(_if_phi){

            phi_k3_im = - _E.real();
            for(int k = 0; k < k3_re.n_elem; k++){

              phi_k3_re += - k3_re[k] * _mean_O_angled[k] - k3_im[k] * _mean_O_square[k];
              phi_k3_im += k3_re[k] * _mean_O_square[k] - k3_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k3_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k3_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              k3_re = pinv(real(_Q), _lambda) * imag(_F);
              k3_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              k3_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              k3_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              k3_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              k3_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            k3_re = (real(_Q)).i() * imag(_F);
            k3_im = - (real(_Q)).i() * real(_F);

          }

          if(_if_phi){

            phi_k3_re = - _E.real();
            for(int k = 0; k < k3_re.n_elem; k++){

              phi_k3_re += - k3_re[k] * _mean_O_angled[k] - k3_im[k] * _mean_O_square[k];
              phi_k3_im += k3_re[k] * _mean_O_square[k] - k3_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Updates the variational parameters
        new_alpha.set_real(alpha_t_re + _delta * k3_re);
        new_alpha.set_imag(alpha_t_im + _delta * k3_im);
        if(_if_phi){

          new_phi.real(phi_t_re + _delta * phi_k3_re);
          new_phi.imag(phi_t_im + _delta * phi_k3_re);

        }

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

      /***************/
      /* FOURTH STEP */
      /***************/
      //Makes a second 𝐭𝐕𝐌𝐂 step at time 𝑡 + δₜ
      MPI_Barrier(common);
      this -> Reset_Moves_Statistics();
      this -> Reset();
      this -> tVMC_Step(common, p);

      if(rank == 0){

        //Solves the appropriate equations of motion
        if(_if_real_time){  // 𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k4_re = (real(_Q) + _eps * _I).i() * real(_F);
              k4_im = (real(_Q) + _eps * _I).i() * imag(_F);

            }
            else if(_regularization_method == 2){

              k4_re = pinv(real(_Q), _lambda) * real(_F);
              k4_im = pinv(real(_Q), _lambda) * imag(_F);

            }
            else if(_regularization_method == 3){

              k4_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);
              k4_im = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);

            }
            else if(_regularization_method == 4){

              k4_re = reg_SVD_inverse(real(_Q)) * real(_F);
              k4_im = reg_SVD_inverse(real(_Q)) * imag(_F);

            }

          }
          else{

            k4_re = (real(_Q)).i() * real(_F);
            k4_im = (real(_Q)).i() * imag(_F);

          }
          if(_if_phi){

            phi_k4_im = - _E.real();
            for(int k = 0; k < k4_re.n_elem; k++){

              phi_k4_re += - k4_re[k] * _mean_O_angled[k] - k4_im[k] * _mean_O_square[k];
              phi_k4_im += k4_re[k] * _mean_O_square[k] - k4_im[k] * _mean_O_angled[k];

            }

          }

        }
        else{  // 𝒊-𝐭𝐕𝐌𝐂

          if(_if_QGT_REG){

            if(_regularization_method == 1){

              k4_re = (real(_Q) + _eps * _I).i() * imag(_F);
              k4_im = - (real(_Q) + _eps * _I).i() * real(_F);

            }
            else if(_regularization_method == 2){

              k4_re = pinv(real(_Q), _lambda) * imag(_F);
              k4_im = - pinv(real(_Q), _lambda) * real(_F);

            }
            else if(_regularization_method == 3){

              k4_re = (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * imag(_F);
              k4_im = - (real(_Q) + std::max(_lambda0 * std::pow(_b, p), _lambda_min) * _I).i() * real(_F);

            }
            else if(_regularization_method == 4){

              k4_re = reg_SVD_inverse(real(_Q)) * imag(_F);
              k4_im = - reg_SVD_inverse(real(_Q)) * real(_F);

            }

          }
          else{

            k4_re = (real(_Q)).i() * imag(_F);
            k4_im = - (real(_Q)).i() * real(_F);

          }
          if(_if_phi){

            phi_k4_re = - _E.real();
            for(int k = 0; k < k4_re.n_elem; k++){

              phi_k4_re += - k4_re[k] * _mean_O_angled[k] - k4_im[k] * _mean_O_square[k];
              phi_k4_im += k4_re[k] * _mean_O_square[k] - k4_im[k] * _mean_O_angled[k];

            }

          }

        }

        //Final update of the variational parameters
        new_alpha.set_real(alpha_t_re + _delta * ((1.0 / 6.0) * k1_re + (1.0 / 3.0) * k2_re + (1.0 / 3.0) * k3_re + (1.0 / 6.0) * k4_re));
        new_alpha.set_imag(alpha_t_im + _delta * ((1.0 / 6.0) * k1_im + (1.0 / 3.0) * k2_im + (1.0 / 3.0) * k3_im + (1.0 / 6.0) * k4_im));
        if(_if_phi){

          new_phi.real(phi_t_re + _delta * ((1.0 / 6.0) * phi_k1_re + (1.0 / 3.0) * phi_k2_re + (1.0 / 3.0) * phi_k3_re + (1.0 / 6.0) * phi_k4_re));
          new_phi.imag(phi_t_im + _delta * ((1.0 / 6.0) * phi_k1_im + (1.0 / 3.0) * phi_k2_im + (1.0 / 3.0) * phi_k3_im + (1.0 / 6.0) * phi_k4_im));

        }

      }

      MPI_Barrier(common);

      //Updates parameters of all the nodes in the communicator
      MPI_Bcast(new_alpha.begin(), _vqs.n_alpha(), MPI_DOUBLE_COMPLEX, 0, common);
      _vqs.set_alpha(new_alpha);
      if(_if_phi){

        MPI_Bcast(&new_phi, 1, MPI_DOUBLE_COMPLEX, 0, common);
        _vqs.set_phi(new_phi);

      }

    }

  }
  else return;

}


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
