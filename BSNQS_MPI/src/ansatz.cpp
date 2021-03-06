#ifndef __ANSATZ__
#define __ANSATZ__


/********************************************************************************************************/
/***************************  Representation of the Many-Body Quantum State  ****************************/
/********************************************************************************************************/
/*

  We create several models in order to represent the quantum state of a many-body system defined
  in the discrete, on a ๐ and ๐ dimensional lattice ๐ฒ ฯต โคแต.
  The structure of the wave function ansatz is designed in a Variational Monte Carlo frameworks,
  that is, all the knowledge about the quantum state is encoded by a set of variational parameters
  that characterizes the generic interface of our classes (in the C++ language this is obtained
  through the use of an Abstract Class).
  These parameters should be optimized via a Variational Monte Carlo algorithm (๐ฌ๐๐ฆ๐ฉ๐ฅ๐๐ซ.๐๐ฉ๐ฉ).
  Moreover, we are interested in building variational quantum states that are Artificial Intelligence
  (๐จ๐ฐ) inspired, so we always consider the presence of a certain number of ๐๐๐๐๐๐๐ variables,
  (i.e. the actual quantum degrees of freedom of the systems), supported by a certain number of
  ๐๐๐๐๐๐ variables (auxiliary quantum degrees of freedom); the different types (visible or hidden)
  of variables are organized into distinct layers, according to a neural-inspired ansatz.
  Depending on the chosen architecture, there may be intra-layer interactions between variables
  of the same type and/or interactions between different variables that live in different layers.
  Even more, in some variational wave function the hidden variables will be traceable, and therefore
  we will have to worry only about the visible variables (as in the ๐๐๐); in the generic case however,
  the fictitious quantum variables will not be analytically integrable, and we should use more
  sophisticated sampling techniques (the ๐๐๐๐๐๐ case).
  However, we will consider complex variational parameters, and a generic form of the type

            ฮจ(๐,๐ฅ) = โฏ๐๐(๐)โขฮฃโ โฏ๐๐(ฮฃโ ๐โ(๐,๐)ฮฑโ) = โฏ๐๐(๐)โขฮฃโ ฮฆ(๐,๐,๐)

  with ๐โ(๐,๐) the so-called ๐๐๐๐ local operators, and ๐ฅ = {๐,๐} ฯต โโฟ-แตแตสณแตแตหข.

  NฬฒOฬฒTฬฒEฬฒ: we use the pseudo-random numbers generator device by [Percus & Kalos, 1989, NY University].
  NฬฒOฬฒTฬฒEฬฒ: we use the C++ Armadillo library to manage Linear Algebra calculations.

*/
/********************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etcโฆ
#include <cstdlib>  // <-- std::abort()
#include <cmath>  // <-- std::cosh(), std::log(), std::exp(), std::cos(), std::sin(), std::tanh()
#include <fstream>  // <-- std::ifstream, std::ofstream
#include <complex>  // <-- std::complex<>, .real(), .imag()
#include <armadillo>  // <-- arma::Mat, arma::Col
#include "random.h"  // <-- Random


using namespace arma;


  /*###########################################*/
 /*  ๐๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐ ๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐  */
/*###########################################*/
class WaveFunction {

  protected:

    //Geometric structure
    std::complex <double> _phi;  //The complex phase variational parameter ๐
    Col <std::complex <double>> _alpha;  //The variational parameters ๐ = {๐ผ๐ฃ,๐ผ๐ค,โฆ,๐ผโฟ-แตแตสณแตแตหข}
    Mat <std::complex <double>> _LocalOperators;  //The local operators ๐(๐,๐)

    //Architecture
    unsigned int _N;  //Number of visible variables ๐ = {๐ฃ๐ฃ,๐ฃ๐ค,โฆ,๐ฃ๐ญ}
    std::string _type;  //Type of ansatz
    bool _if_phi;  //Chooses ๐ โ? ๐ข (true) or ๐ = ๐ข (false)

    //Random device
    Random _rnd;

  public:

    //Constructor and Destructor
    virtual ~WaveFunction() = default;  //Necessary for dynamic allocation


    /*****************************/
    /*  ๐ฉโด๐-๐๐พ๐๐๐๐ถโ ๐ป๐๐๐ธ๐๐พโด๐  */
    /*****************************/
    //Access functions
    unsigned int n_visible() const {return _N;}  //Returns the number of visible variables ๐ = {๐ฃ๐ฃ,๐ฃ๐ค,โฆ,๐ฃ๐ญ}
    std::string type_of_ansatz() const {return _type;}  //Returns the type of the chosen ansatz architecture
    bool if_phi_neq_zero() const {return _if_phi;}  //Returns whether or not to use the phase ๐ in the ansatz
    unsigned int n_alpha() const {return _alpha.n_elem;}  //Returns the number of variational parameters ๐ = {๐ผ๐ฃ,๐ผ๐ค,โฆ,๐ผโฟ-แตแตสณแตแตหข}
    std::complex <double> phi() const {return _phi;}  //Returns the complex phase variational parameter ๐
    Col <std::complex <double>> alpha() const {return _alpha;}  //Returns the set of ๐ = {๐ผ๐ฃ,๐ผ๐ค,โฆ,๐ผโฟ-แตแตสณแตแตหข}
    Mat <std::complex <double>> O() const {return _LocalOperators;}  //Returns the local operators ๐(๐,๐)
    std::complex <double> alpha_at(unsigned int) const;  //Returns a selected variational parameter ๐ผ๐ฟ

    //Modifier functions
    void set_phi(std::complex <double> new_phi) {_phi = new_phi;}  //Changes the value of the complex phase variational parameter ๐
    void set_alpha(const Col <std::complex <double>>& new_alpha) {_alpha = new_alpha;}  //Changes the value of the variational parameters ๐ผ๐ฟ
    void set_alpha_at(unsigned int, std::complex <double>);  //Changes the value of a selected variational parameter ๐ผ๐ฟ
    void set_phi_real(double new_phi_real) {_phi.real(new_phi_real);}  //Changes the value of the real part of the phase variational parameter ๐แดฟ
    void set_phi_imag(double new_phi_imag) {_phi.imag(new_phi_imag);}  //Changes the value of the imaginary part of the phase variational parameter ๐แดต
    void set_alpha_real_at(unsigned int, double);  //Changes the value of the real part of a selected variational parameter ๐ผแดฟ๐ฟ
    void set_alpha_imag_at(unsigned int, double);  //Changes the value of the imaginary part of a selected variational parameter ๐ผแดต๐ฟ

    //Functional form of the ansatz
    Mat <int> generate_config(const Mat <int>&, const Mat <int>&) const;  //Creates a state of the type |๐โฉ, |๐โฉ or |๐หโฉ


    /*************************/
    /*  ๐ฑ๐พ๐๐๐๐ถโ ๐ป๐๐๐ธ๐๐พโด๐  */
    /************************/
    //Access function
    virtual unsigned int density() const = 0;  //Returns the density of the hidden variables ฮฝ = ๐ฌ/๐ญ

    //Modifier functions
    virtual void Init_on_Config(const Mat <int>&) = 0;  //Initializes properly the ansatz on a given quantum configuration
    virtual void Update_on_Config(const Mat <int>&, const Mat <int>&) = 0;  //Updates properly the ansatz on a given new sampled quantum configuration

    //Functional form of the ansatz
    virtual double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')
    virtual double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes cos[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    virtual double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes sin[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    virtual std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const = 0;  //Computes ๐๐๐(ฮฆ(๐,๐,๐))
    virtual std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const = 0;  //Computes ฮฆ(๐,๐,๐)
    virtual std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes ๐๐๐( ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐) )
    virtual std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐)
    virtual std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the natural logarithm of the 'Metropolis part' of the ansatz
    virtual std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the 'Metropolis part' of the ansatz
    virtual std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes the natural logarithm of the sqrt of
                                                        const Mat <int>&, const Mat <int>&,  //the Matropolis acceptance probability
                                                        const Mat <int>&, const Mat <int>&,
                                                        std::string option="") const = 0;
    virtual std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes the sqrt of the Matropolis acceptance probability
                                                     const Mat <int>&, const Mat <int>&,
                                                     const Mat <int>&, const Mat <int>&,
                                                     std::string option="") const = 0;
    virtual double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance
                                            const Mat <int>&, const Mat <int>&,  //probability || ฮจ(๐โฟแตสท,๐ฅ)/ฮจ(๐แตหกแต,๐ฅ) ||
                                            const Mat <int>&, const Mat <int>&,
                                            std::string option="") const = 0;
    virtual void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&) = 0;  //Computes the local operators ๐(๐,๐)

};


  /*####################################*/
 /*  ๐๐๐๐๐๐๐ ๐ฐ๐ข๐ญ๐ก ๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐  */
/*####################################*/
class JasNN : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    JasNN(unsigned int, bool, int);
    JasNN(std::string, bool, int);
    ~JasNN();

    //Access functions
    unsigned int density() const {return 1;}
    std::complex <double> eta() const {return _alpha(0);}  //Returns the nearest neighbors coupling parameter

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}
    double cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 1.0;}  //Computes ๐๐๐?[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    double sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}  //Computes ๐?๐๐[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮจ(๐,๐,๐)) on a given visible configuration
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes ฮจ(๐,๐,๐) on a given visible configuration
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ฮจ(๐โฟแตสท,๐,๐)/ฮจ(๐แตหกแต,๐,๐) )
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ฮจ(๐โฟแตสท,๐,๐)/ฮจ(๐แตหกแต,๐,๐)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮจ(๐,๐,๐))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ฮจ(๐,๐,๐)
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,    //Computes ๐๐๐( ฮจ(๐โฟแตสท,๐,๐)/ฮจ(๐แตหกแต,๐,๐) )
                                              const Mat <int>&, const Mat <int>&,
                                              const Mat <int>&, const Mat <int>&,
                                              std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes ฮจ(๐โฟแตสท,๐,๐)/ฮจ(๐แตหกแต,๐,๐)
                                           const Mat <int>&, const Mat <int>&,
                                           const Mat <int>&, const Mat <int>&,
                                           std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings
                                  const Mat <int>&, const Mat <int>&,  //acceptance probability || ฮจ(๐โฟแตสท,๐,๐)/ฮจ(๐แตหกแต,๐,๐) ||
                                  const Mat <int>&, const Mat <int>&,
                                  std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators ๐(๐) = โ๐๐๐(ฮจ(๐,๐))/โ๐

};


  /*####################################*/
 /*  ๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐  */
/*####################################*/
class RBM : public WaveFunction {

  private:

    //RBM Neural Network architecture
    unsigned int _M;  //Number of hidden neurons ๐ก = {๐ฝ๐ฃ,๐ฝ๐ค,โฆ,๐ฝ๐ฌ}

    //Look-up table for the effective angles ๐ณ(๐,๐)
    Col <std::complex<double>> _Theta;

    //Fast computation of the wave function
    const double _ln2;  //๐๐๐๐ค

  public:

    //Constructor and Destructor
    RBM(unsigned int, unsigned int, bool, int);
    RBM(std::string, bool, int);
    ~RBM();

    //Access functions
    unsigned int density() const {return _M/_N;}
    unsigned int n_hidden() const {return _M;}  //Returns the number of hidden neurons ๐ก = {๐ฝ๐ฃ,๐ฝ๐ค,โฆ,๐ฝ๐ฌ}
    std::complex <double> a_j(unsigned int) const;  //Returns the bias of the ๐ฟ-th visible neuron
    std::complex <double> b_k(unsigned int) const;  //Returns the bias of the ๐-th hidden neuron
    std::complex <double> W_jk(unsigned int, unsigned int) const;  //Returns the selected visible-hidden interaction strength
    std::complex <double> Theta_k(unsigned int) const;  //Returns the effective angles associated to the ๐-th hidden neuron
    Col <std::complex <double>> effective_angle() const {return _Theta;}  //Returns the set of ๐ณ(๐,๐)
    void print_a() const;  //Prints on standard output the set of visible bias ๐
    void print_b() const;  //Prints on standard output the set of hidden bias ๐
    void print_W() const;  //Prints on standard output the visible-hidden interaction strength matrix ๐
    void print_Theta() const;  //Prints on standard output the set of effective angles ๐ณ(๐,๐)

    //Modifier functions
    void Init_on_Config(const Mat <int>&);
    void Update_on_Config(const Mat <int>&, const Mat <int>&);

    //Wavefunction evaluation
    double lncosh(double) const;  //Computes ๐๐๐(๐๐๐?โ๐) of a real number ๐ ฯต โ
    std::complex <double> lncosh(std::complex <double>) const;  //Computes ๐๐๐(๐๐๐?โ๐) of a complex number ๐ ฯต โ
    void Init_Theta(const Mat <int>&);  //Initializes the effective angles ๐ณ(๐,๐) on the given visible configuration |๐โฉ
    void Update_Theta(const Mat <int>&, const Mat <int>&);  //Updates the effective angles ๐ณ(๐,๐) on a new sampled visible configuration |๐โฟแตสทโฉ
    double I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}
    double cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 1.0;}  //Computes ๐๐๐?[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    double sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}  //Computes ๐?๐๐[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮจ(๐,๐ฅ)) on a given visible configuration
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes ฮจ(๐,๐ฅ) on a given visible configuration
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮจ(๐โฟแตสท,๐ฅ) / ฮจ(๐แตหกแต,๐ฅ))
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ฮจ(๐โฟแตสท,๐ฅ) / ฮจ(๐แตหกแต,๐ฅ)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮจ(๐,๐ฅ))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ฮจ(๐,๐ฅ)
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,    //Computes ๐๐๐(ฮจ(๐โฟแตสท,๐ฅ) / ฮจ(๐แตหกแต,๐ฅ))
                                                const Mat <int>&, const Mat <int>&,
                                                const Mat <int>&, const Mat <int>&,
                                                std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes ฮจ(๐โฟแตสท,๐ฅ) / ฮจ(๐แตหกแต,๐ฅ)
                                             const Mat <int>&, const Mat <int>&,
                                             const Mat <int>&, const Mat <int>&,
                                             std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings
                                    const Mat <int>&, const Mat <int>&,  //acceptance probability || ฮจ(๐โฟแตสท,๐ฅ) / ฮจ(๐แตหกแต,๐ฅ) ||
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators ๐(๐) = โ๐๐๐(ฮจ(๐,๐))/โ๐

};


  /*##############################################*/
 /*  ๐๐๐๐๐๐๐-๐๐๐๐๐๐๐๐๐ (quasi)-๐ฎ๐๐๐ in ๐ฑ = ๐  */
/*##############################################*/
class quasi_uRBM : public WaveFunction{

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    quasi_uRBM(unsigned int, bool, int rank);
    quasi_uRBM(std::string, bool, int rank);
    ~quasi_uRBM();

    //Access functions
    unsigned int density() const {return 1;}
    std::complex <double> eta_j(unsigned int) const;  //Returns the selected visible-visible interaction strength
    std::complex <double> rho_j(unsigned int) const;  //Returns the selected hidden-hidden interaction strength
    std::complex <double> omega_j(unsigned int) const;  //Returns the selected visible-hidden interaction strength
    void print_eta() const;  //Prints on standard output the set of visible-visible interaction strength ๐
    void print_rho() const;  //Prints on standard output the set of hidden-hidden interaction strength ๐
    void print_omega() const;  //Prints on standard output the set of visible-hidden interaction strength ๐

    //Modifier functions
    void Init_on_Config(const Mat <int>& visible_config) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐?[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐?๐๐[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮฆ(๐,๐,๐))
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes ฮฆ(๐,๐,๐)
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐) )
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(๐(๐,๐,๐ห,๐ฅ))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐,๐,๐ห,๐ฅ)
    double logq_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ๐(๐โฟแตสท,๐,๐ห,๐ฅ)/๐(๐แตหกแต,๐,๐ห,๐ฅ) )
    double q_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐โฟแตสท,๐,๐ห,๐ฅ)/๐(๐แตหกแต,๐,๐ห,๐ฅ)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ๐(๐,๐โฟแตสท,๐ห,๐ฅ)/๐(๐,๐แตหกแต,๐ห,๐ฅ) )
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐,๐โฟแตสท,๐ห,๐ฅ)/๐(๐,๐แตหกแต,๐ห,๐ฅ)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ๐(๐,๐,๐หโฟแตสท,๐ฅ)/๐(๐,๐,๐หแตหกแต,๐ฅ) )
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐,๐,๐หโฟแตสท,๐ฅ)/๐(๐,๐,๐หแตหกแต,๐ฅ)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes ๐๐๐( ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ) ) on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes ๐๐๐( ๐(๐,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐,๐แตหกแต,๐หแตหกแต,๐ฅ) )
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes ๐(๐,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐,๐แตหกแต,๐หแตหกแต,๐ฅ)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes ๐๐๐( ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ) )
                                                const Mat <int>&, const Mat <int>&,
                                                const Mat <int>&, const Mat <int>&, std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ)
                                             const Mat <int>&, const Mat <int>&,
                                             const Mat <int>&, const Mat <int>&, std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || ฮจ(๐โฟแตสท,๐ฅ)/ฮจ(๐แตหกแต,๐ฅ) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&, std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators ๐(๐,๐) = โ๐๐๐(ฮฆ(๐,๐,๐))/โ๐

};


  /*#####################################*/
 /*  ๐๐๐๐๐๐๐๐๐-๐๐๐๐๐๐ ๐๐๐๐ in ๐ฑ = ๐  */
/*#####################################*/
class BS_NNQS : public WaveFunction{

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    BS_NNQS(unsigned int, bool, int rank);
    BS_NNQS(std::string, bool, int rank);
    ~BS_NNQS();

    //Access functions
    unsigned int density() const {return 1;}
    std::complex <double> eta() const {return _alpha(0);}  //Returns the visible-visible interaction strength ฮท
    std::complex <double> rho() const {return _alpha(1);}  //Returns the hidden-hidden interaction strength ฯ
    std::complex <double> omega() const {return _alpha(2);}  //Returns the visible-hidden interaction strength ฯ

    //Modifier functions
    void Init_on_Config(const Mat <int>& visible_config) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐?[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐?๐๐[โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(ฮฆ(๐,๐,๐))
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes ฮฆ(๐,๐,๐)
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐) )
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐(๐(๐,๐,๐ห,๐ฅ))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐,๐,๐ห,๐ฅ)
    double logq_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ๐(๐โฟแตสท,๐,๐ห,๐ฅ)/๐(๐แตหกแต,๐,๐ห,๐ฅ) )
    double q_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐โฟแตสท,๐,๐ห,๐ฅ)/๐(๐แตหกแต,๐,๐ห,๐ฅ)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ๐(๐,๐โฟแตสท,๐ห,๐ฅ)/๐(๐,๐แตหกแต,๐ห,๐ฅ) )
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐,๐โฟแตสท,๐ห,๐ฅ)/๐(๐,๐แตหกแต,๐ห,๐ฅ)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐๐๐( ๐(๐,๐,๐หโฟแตสท,๐ฅ)/๐(๐,๐,๐หแตหกแต,๐ฅ) )
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes ๐(๐,๐,๐หโฟแตสท,๐ฅ)/๐(๐,๐,๐หแตหกแต,๐ฅ)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes ๐๐๐( ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ) ) on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes ๐๐๐( ๐(๐,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐,๐แตหกแต,๐หแตหกแต,๐ฅ) )
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes ๐(๐,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐,๐แตหกแต,๐หแตหกแต,๐ฅ)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes ๐๐๐( ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ) )
                                                const Mat <int>&, const Mat <int>&,
                                                const Mat <int>&, const Mat <int>&,
                                                std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ)
                                             const Mat <int>&, const Mat <int>&,
                                             const Mat <int>&, const Mat <int>&,
                                             std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || ฮจ(๐โฟแตสท,๐ฅ)/ฮจ(๐แตหกแต,๐ฅ) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators ๐(๐,๐) = โ๐๐๐(ฮฆ(๐,๐,๐))/โ๐

};




/*******************************************************************************************************************************/
/******************************************  ๐๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐ ๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐  ******************************************/
/*******************************************************************************************************************************/
std::complex <double> WaveFunction :: alpha_at(unsigned int j) const {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to access the variational parameter set ๐." << std::endl;
    std::cerr << "   The variational parameter ๐ผ๐ฟ with ๐ฟ = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else
    return _alpha(j);

}


void WaveFunction :: set_alpha_at(unsigned int j, std::complex <double> new_param) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the variational parameter set ๐." << std::endl;
    std::cerr << "   The variational parameter ๐ผ๐ฟ with ๐ฟ = " << j << " does not exist." << std::endl;
    std::abort();

  }
  //Check passed
  else
    _alpha(j) = new_param;

}


void WaveFunction :: set_alpha_real_at(unsigned int j, double new_param_real) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the real part of the variational parameter set ๐." << std::endl;
    std::cerr << "   The variational parameter ๐ผแดฟ๐ฟ with ๐ฟ = " << j << " does not exist." << std::endl;
    std::abort();

  }
  //Check passed
  else
    _alpha(j).real(new_param_real);

}


void WaveFunction :: set_alpha_imag_at(unsigned int j, double new_param_imag) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the imaginary part of the variational parameter set ๐." << std::endl;
    std::cerr << "   The variational parameter ๐ผแดต๐ฟ with ๐ฟ = " << j << " does not exist." << std::endl;
    std::abort();

  }
  //Check passed
  else
    _alpha(j).imag(new_param_imag);

}


Mat <int> WaveFunction :: generate_config(const Mat <int>& old_config, const Mat <int>& flipped_site) const {

  //Function variables
  Mat <int> new_config = old_config;

  if(flipped_site.n_elem != 0){

    for(unsigned int j_row = 0; j_row < flipped_site.n_rows; j_row++)
      new_config(0, flipped_site(j_row, 0)) *= -1;
    return new_config;

  }
  else{

    return new_config;

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  ๐๐๐๐๐๐๐ ๐ฐ๐ข๐ญ๐ก ๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐  *********************************************/
/*******************************************************************************************************************************/
JasNN :: JasNN(unsigned int n_visible, bool phi_option, int rank) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the nearest-neighbors entangling Jatrow variational parameters
  //  ๐ฅ = {๐, ๐} = {๐, ๐} to some small random numbers.
  //
  //  In this case we have only ๐ญ parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        ๐ complex phase ๐
  //        ๐ nearest-neighbors visible-visible interaction strength ๐.
  //
  //  Note that in this case the number of variational parameters remains equal to ๐ for any system size ๐ญ.
  /*########################################################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a nearest-neighbors Jastrow wave function with randomly initialized variational parameters ๐ฅ = {๐, ๐}." << std::endl;

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _N = n_visible;
  _type = "Jastrow";
  _if_phi = phi_option;
  _alpha.set_size(1);
  _LocalOperators.zeros(1, 2);  //NฬฒOฬฒTฬฒEฬฒ: ๐_๐ = ๐, so we do not save it in memory
  if(_if_phi){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else
    _phi = 0.0;
  _alpha(0).real(_rnd.Gauss(0.0, 0.001));
  _alpha(0).imag(_rnd.Gauss(0.0, 0.001));

  if(rank == 0){

    std::cout << " Nearest-neighbors Jastrow ansatz correctly initialized with random interactions." << std::endl;
    std::cout << " Number of visible variables = " << _N << "." << std::endl;
    std::cout << " Number of hidden variables = " << _N << "." << std::endl;
    std::cout << " Density of the nearest-neighbors Jastrow ansatz = " << this -> density() << "." << std::endl << std::endl;

  }

}


JasNN :: JasNN(std::string file_wf, bool phi_option, int rank) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the nearest-neighbors Jastrow variational parameters
  //  ๐ฅ = {๐, ๐} = {๐, ๐} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a nearest-neighbors Jastrow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the nearest-neighbors Jastrow variational parameters ๐ฅ = {๐, ๐} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _if_phi = phi_option;
  input_wf >> _N;
  input_wf >> _phi;
  if(_if_phi == false && _phi != 0.0) {

    std::cerr << " ##ValueError: value for the variational phase not compatible with the ansatz construction option." << std::endl;
    std::cerr << "   Failed to construct the Jastrow variational quantum state." << std::endl;
    std::abort();

  }
  if(!input_wf.good() || _N < 0){

    std::cerr << " ##FileError: invalid construction of the nearest-neighbors Jastrow ansatz." << std::endl;
    std::abort();

  }
  _type = "Jastrow";
  _alpha.set_size(1);
  _LocalOperators.set_size(1, 2);  //NฬฒOฬฒTฬฒEฬฒ: ๐_๐ = ๐, so we do not save it in memory
  input_wf >> _alpha(0);

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Nearest-neighbors Jastrow Ansatz correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of visible neurons = " << _N << "." << std::endl;
      std::cout << " Number of hidden neurons = " << _N << "." << std::endl;
      std::cout << " Density of the nearest-neighbors Jastrow ansatz = " << this -> density() << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


JasNN :: ~JasNN() {

  //_rnd.SaveSeed();

}


std::complex <double> JasNN :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*####################################################*/
  //  Computes ๐๐๐(ฮจ(๐,๐,๐)) with
  //
  //        ฮจ(๐,๐,๐) = โฏ๐๐(๐) โขย?โฏ๐๐(ฮฃโ ๐โ(๐,๐)ฮฑโ)
  //                 = โฏ๐๐(๐) โขย?โฏ๐๐(๐ ฮฃ๐ฟ ๐ฃ๐ฟโข๐ฃ๐ฟ+๐ฃ).
  //
  //  Obviously, this ansatz is not of the Shadow type,
  //  and no auxiliary variables are introduced here.
  /*####################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐,๐,๐))." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_psi = 0.0;  //Storage variable for the nearest-neighbors interaction

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1)
      log_psi += double(visible_config(0, j) * visible_config(0, 0));  // ๐_๐ญ โข ๐_๐ข
    else
      log_psi += double(visible_config(0, j) * visible_config(0, j+1));  // ๐๐ฟ โข ๐๐ฟ+๐ฃ

  }

  return this -> phi() + this -> eta() * log_psi;

}


std::complex <double> JasNN :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


std::complex <double> JasNN :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                     const Mat <int>& hidden_config) const {

  /*##############################################################################*/
  //  Computes ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต)) at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the old visible configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the ๐๐ฉ๐๐๐ญ๐_๐๐ก๐๐ญ๐ function
  //  defined below in the ๐๐๐ class.
  //  Note that the ratio between the two evaluated wave function, which is the
  //  quantity related to the acceptance kernel of the Metropolis algorithm,
  //  is recovered by taking the exponential of the output of this function.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: once again we emphasize that in the specific case of the Jastrow
  //        ansatz the quantities calculated with the functions inherent to
  //        ฮฆ(๐,๐,๐) correspond to those calculated in the functions related
  //        to the Metropolis algorithm, since we have never introduced any
  //        auxiliary variable.
  //  NฬฒOฬฒTฬฒEฬฒ: the ๐๐๐๐๐๐_๐๐จ๐ง๐๐ข๐? argument is useless for the Jastrow ansatz,
  //        which does not depend upon any hidden variables.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต))." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |๐โฟแตสทโฉ
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสทโฉ = |๐แตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new visible configuration does not match with the number of visible variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |๐โฟแตสทโฉ
    double log_vv = 0.0;  //Storage variable for the visible-visible terms

    //Computes the visible-visible terms: ฮฃ๐ฟ (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
      else
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)

    }

    return this -> eta() * log_vv;

  }

}


std::complex <double> JasNN :: PhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                  const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_config));

}


std::complex <double> JasNN :: logPsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> logPhi(visible_config, hidden_ket);

}


std::complex <double> JasNN :: PsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> Phi(visible_config, hidden_ket);

}


std::complex <double> JasNN :: logPsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                     const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                     const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                     std::string option) const {

  return this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


std::complex <double> JasNN :: PsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                  const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                  const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                  std::string option) const {

  return this -> PhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


double JasNN :: PMetroNew_over_PMetroOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                           const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                           const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                           std::string option) const {

  return std::norm(this -> PsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                              hidden_ket, flipped_ket_site,
                                              hidden_bra, flipped_bra_site));

}


void JasNN :: LocalOperators(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameter
  //  ๐ on the sampled enlarged quantum configuration |๐ ๐ ๐หโฉ.
  //  In the case of the Jastrow ansatz the local operators ๐(๐,๐) is
  //
  //        โข ๐ โโ ๐(๐,๐) = ๐(๐) = ฮฃ๐ฟ ๐ฃ๐ฟโข๐ฃ๐ฟ+๐ฃ.
  //
  //  This operator is necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐)." << std::endl;
    std::abort();

  }

  //Function variables
  double O_vv = 0.0;  //Storage variable for the visible-visible terms

  //Computes the local operator assiociated to the only parameter ๐
  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1)
      O_vv += double(visible_config(0, j) * visible_config(0, 0));  // ๐_๐ญ โข ๐_๐ข
    else
      O_vv += double(visible_config(0, j) * visible_config(0, j+1));  // ๐๐ฟ โข ๐๐ฟ+๐ฃ

  }

  _LocalOperators(0, 0) = O_vv;  // ๐_ฮท(๐)

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  ๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐  *********************************************/
/*******************************************************************************************************************************/
RBM :: RBM(unsigned int n_visible, unsigned int density, bool phi_option, int rank)
     : _M(density*n_visible), _ln2(std::log(2.0)) {

  /*################################################################################*/
  //  Random-based constructor.
  //  Initializes the RBM variational parameters ๐ = {๐,๐,๐} to
  //  some small random numbers [G.Hinton, 2010].
  //  We have
  //
  //        ๐ญ visible neuron bias ๐ = {๐๐ข,๐๐ฃ,โฆ,๐๐ญ};
  //        ๐ฌ hidden neuron bias ๐ = {๐๐ข,๐๐ฃ,โฆ,๐๐ญ};
  //        ๐ญ โข ๐ฌ visible-hidden neuron interaction strength weights ๐ = [๐]๐ฟ๐
  //
  //  organized sequentially in the parameter vector data-member.
  //  Note that being ๐ a matrix, we 'unrolled' it row by row saving
  //  it in _alpha as a vector of ๐ญ โข ๐ฌ elements.
  //  We remember that the ๐ฟ-th row of ๐ represents the list of the interactions
  //  strength between the ๐ฟ-th visible variable and each of the ๐ฌ hidden neurons.
  /*################################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a RBM wave function with randomly initialized variational parameters ๐ = {๐,๐,๐ }." << std::endl;

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _if_phi = phi_option;
  if(_if_phi){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else
    _phi = 0.0;
  _type = "Neural Network";
  _N = n_visible;
  _alpha.set_size(_N + _M + _N * _M);
  _LocalOperators.zeros(_N + _M + _N * _M, 2);
  _Theta.set_size(_M);

  //Visible bias
  for(unsigned int j = 0; j < _N; j++)
    _alpha(j) = 0.0;  // ๐ผโฑผ โก ๐โฑผ

  //Hidden bias
  for(unsigned int k = _N; k < _N + _M; k++)
    _alpha(k) = 0.0;  // ๐ผโ โก ๐โ

  //Visible-hidden interaction weights
  for(unsigned int jk = _N + _M; jk < _alpha.n_elem; jk++){

    _alpha(jk).real(_rnd.Gauss(0.0, 0.1));  // ๐ผโฑผแดฟ โก [๐]แดฟ๐ฟ๐
    _alpha(jk).imag(_rnd.Gauss(0.0, 0.1));  // ๐ผโฑผแดต โก [๐]แดต๐ฟ๐

  }

  if(rank == 0){

    std::cout << " RBM ansatz correctly initialized with random weights." << std::endl;
    std::cout << " Number of visible neurons = " << _N << "." << std::endl;
    std::cout << " Number of hidden neurons = " << _M << "." << std::endl;
    std::cout << " Density of the RBM ansatz = " << this -> density() << "." << std::endl << std::endl;

  }

}


RBM :: RBM(std::string file_wf, bool phi_option, int rank)
     : _ln2(std::log(2.0)) {

  /*##############################################################*/
  //  File-based constructor.
  //  Initializes RBM variational parameters ๐ = {๐,๐,๐} from a
  //  given external file in '.wf' format; this can be useful
  //  in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent
  //  variational Monte Carlo with a previously optimized
  //  ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*##############################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a RBM wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the RBM variational parameters ๐ = { ๐,๐,๐ } from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  unsigned int density = 0;
  input_wf >> _N;
  input_wf >> density;
  _M = _N * density;
  if(!input_wf.good() || _N < 0 || _M < 0){

    std::cerr << " ##FileError: invalid construction of the RBM ansatz." << std::endl;
    std::cerr << "   Failed to initialize the RBM variational parameters ๐ = { ๐,๐,๐ } from file." << std::endl;
    std::abort();

  }
  _if_phi = phi_option;
  _type = "Neural Network";
  _alpha.set_size(_N + _M + _N * _M);
  _LocalOperators.zeros(_N + _M + _N * _M, 1);  // ๐โ(๐,๐) โก ๐โ(๐,๐ห) โก ๐โ(๐)
  _Theta.set_size(_M);
  input_wf >> _phi;
  if(_if_phi == false && _phi != 0.0) {

    std::cerr << " ##ValueError: value for the variational phase not compatible with the ansatz construction option." << std::endl;
    std::cerr << "   Failed to construct the Jastrow variational quantum state." << std::endl;
    std::abort();

  }
  for(unsigned int p = 0; p <_alpha.n_elem; p++)
    input_wf >> _alpha(p);

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " RBM ansatz correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of visible neurons = " << _N << "." << std::endl;
      std::cout << " Number of hidden neurons = " << _M << "." << std::endl;
      std::cout << " Density of the RBM ansatz = " << this -> density() << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


RBM :: ~RBM() {

  //_rnd.SaveSeed();

}


std::complex <double> RBM :: a_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the visible bias vector ๐ = {๐๐ข,๐๐ฃ,โฆ,๐๐ญ}." << std::endl;
    std::cerr << "   Element ๐โฑผ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(j);

}


std::complex <double> RBM :: b_k(unsigned int k) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(k >= _M || k < 0){

    std::cerr << " ##IndexError: failed to access the hidden bias vector ๐ = {๐๐ข,๐๐ฃ,โฆ,๐๐ญ}." << std::endl;
    std::cerr << "   Element ๐โ with k = " << k << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(_N + k);

}


std::complex <double> RBM :: W_jk(unsigned int j, unsigned int k) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || k >= _M || j < 0 || k < 0){

    std::cerr << " ##IndexError: failed to access the visible-hidden interaction strength matrix ๐." << std::endl;
    std::cerr << "   Element ๐โฑผโ with j = " << j << " and k = " << k << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(_N + _M + j * _M + k);

}


std::complex <double> RBM :: Theta_k(unsigned int k) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(k >= _M || k < 0){

    std::cerr << " ##IndexError: failed to access the effective angles ๐ณ(๐,๐)." << std::endl;
    std::cerr << "   Element ๐ณโ with k = " << k << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _Theta(k);

}


void RBM :: print_a() const {

  std::cout << "\n======================================" << std::endl;
  std::cout << "RBM visible bias vector ๐ = {๐๐ข,๐๐ฃ,โฆ,๐๐ญ}" << std::endl;
  std::cout << "======================================" << std::endl;
  for(unsigned int j = 0; j < _N; j++){

    std::cout << _alpha(j).real();
    if(_alpha(j).imag() >= 0)
      std::cout << " + i" << _alpha(j).imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha(j).imag() << "  " << std::endl;

  }

}


void RBM :: print_b() const {

  std::cout << "\n=====================================" << std::endl;
  std::cout << "RBM hidden bias vector ๐ = {๐๐ข,๐๐ฃ,โฆ,๐๐ญ}" << std::endl;
  std::cout << "=====================================" << std::endl;
  for(unsigned int k = 0; k < _M; k++){

    std::cout << _alpha(_N + k).real();
    if(_alpha(_N + k).imag() >= 0)
      std::cout << " + i" << _alpha(_N + k).imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha(_N + k).imag() << "  " << std::endl;

  }

}


void RBM :: print_W() const {

  std::cout << "\n=========================================================" << std::endl;
  std::cout << "RBM visible-hidden interaction strength matrix ๐ = [๐]๐ฟ๐" << std::endl;
  std::cout << "=========================================================" << std::endl;
  for(unsigned int j = 0; j < _N; j++){

    for(unsigned int k = 0; k < _M; k++){

      std::cout << _alpha(_N + _M + j * _M + k).real();
      if(_alpha(_N + _M + j * _M + k).imag() >= 0)
        std::cout << " + i" << _alpha(_N + _M + j * _M + k).imag() << "  ";
      else
        std::cout << " - i" << -1.0 * _alpha(_N + _M + j * _M + k).imag() << "  ";

    }
    std::cout << std::endl;

  }

}


void RBM :: print_Theta() const {

  std::cout << "\n==========================" << std::endl;
  std::cout << "RBM effective angles ๐ณ(๐,๐)" << std::endl;
  std::cout << "==========================" << std::endl;
  for(unsigned int k = 0; k < _Theta.n_elem; k++){

    std::cout << _Theta(k).real();
    if(_Theta(k).imag() >= 0)
      std::cout << " + i" << _Theta(k).imag() << std::endl;
    else
      std::cout << " + i" << -1.0 * _Theta(k).imag() << std::endl;

  }

}


void RBM :: Init_on_Config(const Mat <int>& visible_config) {

  this -> Init_Theta(visible_config);

}


void RBM :: Update_on_Config(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site) {

  this -> Update_Theta(visible_config, flipped_visible_site);

}


double RBM :: lncosh(double x) const {

  /*###########################################################*/
  //  Computes the natural logarithm of the hyperbolic cosine
  //  of a real number ๐ ฯต โ; we use the asymptotic expantion
  //  when the argument exceeds a certain threshold for
  //  computational efficiency reasons (see the appropriate
  //  section in the Jupyter Notebook ๐๐จ๐ญ๐๐ฌ.๐ข๐ฉ๐ฒ๐ง๐).
  //  The asymptotic expantion is:
  //
  //        ๐๐๐(๐๐๐?โ๐) ~ ๐ - ๐๐๐๐ค
  //
  /*###########################################################*/

  if(x < 6.0)
    return std::log(std::cosh(x));
  else
    return x - _ln2;

}


std::complex <double> RBM :: lncosh(std::complex <double> z) const {

  /*#################################################################################*/
  //  Computes the complex natural logarithm of
  //  the hyperbolic cosine of a generic complex
  //  number ๐ ฯต โ
  //
  //        ๐ = โe{๐} + iโขโm{๐} = ๐ + i๐
  //
  //  We can manipulate the complex logarithm in
  //  the following way:
  //
  //        ๐๐๐(๐๐๐?โ๐) = ๐๐๐[1/2 โข (eแถป + 1/eแถป)]
  //                  = ๐๐๐[1/2 โขย?(eแตโขeโฑแต + 1/eแตโขeโฑแต)]
  //                  = ๐๐๐{1/2 โขย?[eแตโข(cos(๐) + iโขsin(๐)) + 1/eแต(cos(๐) - iโขsin(๐))]}
  //                  = ๐๐๐{1/2 โขย?[cos(๐)โข(eแต + 1/eแต) + iโขsin(๐)โข(eแต - 1/eแต)]}
  //                  = ๐๐๐{cosh(๐)โขcos(๐) + iโขsinh(๐)โขsin(๐)}
  //                  = ๐๐๐{cosh(๐)โข[cos(๐) + iโขtanh(๐)โขsin(๐)]}
  //                  = lncosh(โe{๐}) + ๐๐๐{cos(โm{๐}) + iโขtanh(โe{๐})โขsin(โm{๐})}
  //
  //  where the first term in the final line is calculated through the
  //  function lncosh(x) defined above.
  /*#################################################################################*/

  double xr = z.real();
  double xi = z.imag();

  std::complex <double> result = this -> lncosh(xr);
  result += std::log(std::complex <double> (std::cos(xi), std::tanh(xr) * std::sin(xi)));
  return result;

}


void RBM :: Init_Theta(const Mat <int>& visible_config) {

  /*#######################################################################*/
  //  ๐ณ(๐,๐) Initialization  -->  ๐ณโ(๐,๐) = ๐โ + ฮฃโ [๐]โโโขฯโแถป
  //  Initializes the effective angles that appear thanks to the fact that
  //  in the particular functional form chosen for the this wave function
  //  the hidden degrees of freedom are traced out exactly.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: This analytical integration changes the generic form
  //        introduced above for the ansatz and consequently will also
  //        change the optimization algorithm (๐ฌ๐๐ฆ๐ฉ๐ฅ๐๐ซ.๐๐ฉ๐ฉ).
  //        In fact here the local operators associated with variational
  //        parameters become complex and no longer real, just as the angles
  //        ๐ณ(๐,๐).
  //
  //  The angles depend on the parameters {๐,๐} and
  //  on the visible variables (i.e. the quantum spin) that define the
  //  current quantum configuration of the associated quantum system.
  //  The effective angles serve both in the estimate of the Monte
  //  Carlo observables (via the Metropolis Algorithm) and in the
  //  stochastic optimization of the variational parameters
  //  (via imaginary-time and/or real-time VMC).
  //
  //  The (sampled) configuration ๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐๐จ๐ง๐๐ข๐? on which the effective
  //  angles are calculated can be either the configuration of a quantum
  //  spin system in ๐ dimension (๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐๐จ๐ง๐๐ข๐?.n_rows = ๐), or
  //  in ๐ dimensions (๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐๐จ๐ง๐๐ข๐?.n_rows โ? ๐), for example
  //
  //                                ๐ฉ
  //                     < -------------------- >        ^
  //                    | ฯแถป ฯแถป ฯแถป     โฆ       ฯแถป  \     |
  //                    | ฯแถป ฯแถป ฯแถป     โฆ       ฯแถป   \    |
  //        |๐๐ฃ โฆ ๐๐ญโฉ = | :  :  :      โฆ       ฯแถป    \   |  โณ
  //                    | :  :  :      โฆ       ฯแถป    /   |
  //                    | :  :  :      โฆ       ฯแถป   /    |
  //                    | ฯแถป ฯแถป ฯแถป     โฆ       ฯแถป  /     |
  //                                                     v
  //
  //  for a total of ๐ญ = ๐ฉโขโณ quantum degrees of freedom.
  /*#######################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to initialize the effective angles vector ๐ณ(๐,๐)." << std::endl;
    std::abort();

  }

  //Computes the effective angles
  for(unsigned int k = 0; k < _M; k++){

    _Theta(k) = _alpha(_N + k);
    for(unsigned int m_row = 0; m_row < visible_config.n_rows; m_row++){

      for(unsigned int m_col = 0; m_col < visible_config.n_cols; m_col++)
        _Theta(k) += _alpha(_N + _M + (m_row * visible_config.n_cols + m_col) * _M + k) * double(visible_config(m_row, m_col));

    }

  }

}


void RBM :: Update_Theta(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site) {

  /*##################################################################################*/
  //  Updates the effective angles by exploiting the look-up
  //  table _๐๐ก๐๐ญ๐ when a new quantum configuration is sampled during
  //  the Monte Carlo Markov Chain (MCMC).
  //  We efficiently represent the new configuration in which
  //  the quantum system is through the matrix ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐ฌ๐ข๐ญ๐,
  //  which contains the list of the indices (integer numbers)
  //  related to the lattice sites in which the spins of the old configuration
  //  |๐แตหกแตโฉ have been flipped compared to the new configuration |๐โฟแตสทโฉ
  //  proposed with the Metropolis algorithm (instead of saving
  //  the entire matrix related to the new quantum configuration).
  //  So in ๐ = ๐ this matrix will be reduced to a column vector of the type
  //
  //        โ  ๐ข  โ : we have flipped the 1st spin of the ๐d chain
  //        |  ๐ซ  | : we have flipped the 10th spin of the ๐d chain
  //        |  โข  | : โ                                           โ
  //        |  โข  | : โ                                           โ
  //        |  โข  | : โ                                           โ
  //        โ  โข  โ : โ                                           โ
  //
  //  while in ๐ = ๐ it will be a matrix in which each
  //  row represents the pair of indices which identifies the two dimensional lattice
  //  flipped spin site (๐ ๐) ฯต ๐ฒ, e.g. the spin in first position is represented
  //  with the pair (๐ข ๐ข) in this matrix.
  //  However, in any case, the effective angles are updated as follows:
  //
  //        ๐ณโ(๐โฟแตสท,๐) = ๐ณโ(๐แตหกแต,๐) - 2 โข ฮฃ๐ฟ [๐]๐ฟโโขฯ๐ฟแถป
  //
  //  where ๐ฟ is an index that runs only on the lattice sites where
  //  a spin is flipped, as described above.
  /*##################################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to update the effective angles vector ๐ณ(๐,๐)." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |๐โฟแตสทโฉ
  if(flipped_visible_site.n_elem == 0)
    return;
  else{

    //Check on the lattice dimensionality
    if(visible_config.n_rows == 1 && flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to update the effective angles vector ๐ณ(๐,๐)." << std::endl;
      std::abort();

    }
    if(visible_config.n_rows != 1 && flipped_visible_site.n_cols != 2){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to update the effective angles vector ๐ณ(๐,๐)." << std::endl;
      std::abort();

    }

    //Function variables
    std::complex <double> delta_theta;  //Storage variable

    //Updates ๐ณ(๐,๐)
    for(unsigned int k = 0; k < _M; k++){

      delta_theta = 0.0;
      for(unsigned int m_row = 0; m_row < flipped_visible_site.n_rows; m_row++){

        if(flipped_visible_site.n_cols == 1)  //๐ = ๐
          delta_theta += double(visible_config(0, flipped_visible_site(m_row, 0))) * _alpha(_N + _M + flipped_visible_site(m_row, 0) * _M + k);
        else if(flipped_visible_site.n_cols == 2)  //๐ = ๐
          delta_theta += double(visible_config(flipped_visible_site(m_row, 0), flipped_visible_site(m_row, 1))) *
                                _alpha(_N + _M + (flipped_visible_site(m_row, 0) * visible_config.n_cols + flipped_visible_site(m_row, 1)) * _M + k);
        else{

          std::cerr << " ##SizeError: bad construction of the new quantum configuration |๐โฟแตสทโฉ." << std::endl;
          std::cerr << "   Failed to updatet the effective angles vector ๐ณ(๐,๐)." << std::endl;
          std::abort();

        }

      }
      _Theta(k) -= 2.0 * delta_theta;  //Using the Look-up table for fast computation

    }

  }

}


std::complex <double> RBM :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*###################################################################*/
  //  Since we have managed to integrate exactly the hidden degrees
  //  of freedom for this ansatz, we can here interpret ฮฆ(๐,๐,๐) as
  //  the total wave function, which is defined as (we set ๐ = ๐ข):
  //
  //        ฮจ(๐,๐) = ฮฃโ โฏ๐๐(ฮฃโฑผ๐โฑผฯโฑผแถป + ฮฃโ๐โ๐ฝโ + ฮฃโฑผ[๐]โฑผโ๐ฝโฑผฯโแถป)
  //               = โฏ๐๐(ฮฃโฑผ๐โฑผฯโฑผแถป) โข ๐ทโ 2๐๐๐?โ(๐ณโ)
  //
  //  where the effective angles are defined above.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: the ๐๐๐๐๐๐_๐๐จ๐ง๐๐ข๐? argument is useless for the RBM ansatz,
  //        which does not depend explicitly on the hidden variables.
  /*###################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐,๐))." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_vv(0.0, 0.0);  //Storage variable for the visible terms
  std::complex <double> log_theta(0.0, 0.0);  //Storage variable for the theta angle terms

  //Computes the visible neurons terms: ฮฃโฑผ ๐โฑผฯโฑผแถป
  for(unsigned int j_row = 0; j_row < visible_config.n_rows; j_row++){

    for(unsigned int j_col = 0; j_col < visible_config.n_cols; j_col++)
      log_vv += _alpha(j_row * visible_config.n_cols + j_col) * double(visible_config(j_row, j_col));

  }

  //Computes the theta angles contribution: ฮฃโ ๐๐๐(๐๐๐?โ(๐ณโ))
  for(unsigned int k = 0; k < _M; k++){

    log_theta = _alpha(_N + k);
    for(unsigned int j_row = 0; j_row < visible_config.n_rows; j_row++){

      for(unsigned int j_col = 0; j_col < visible_config.n_cols; j_col++)
        log_theta += _alpha(_N + _M + (j_row * visible_config.n_cols + j_col) * _M + k) * double(visible_config(j_row, j_col));

    }
    log_vv += this -> lncosh(log_theta);

  }

  return log_vv + _M * _ln2;

}


std::complex <double> RBM :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


std::complex <double> RBM :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                   const Mat <int>& hidden_config) const {

  /*#############################################################################*/
  //  Computes ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต)) at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the
  //  ๐๐ฉ๐๐๐ญ๐_๐๐ก๐๐ญ๐ function defined above.
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: once again we emphasize that in the specific case of the RBM
  //        the quantities calculated with the functions inherent to ฮฆ(๐,๐,๐)
  //        correspond to those calculated in the functions related to the
  //        Metropolis algorithm, since we have traced away the fictitious
  //        degrees of freedom.
  //  NฬฒOฬฒTฬฒEฬฒ: the ๐๐๐๐๐๐_๐๐จ๐ง๐๐ข๐? argument is useless for the RBM ansatz,
  //        which does not depend explicitly on the hidden variables.
  /*#############################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐( ฮจ(๐โฟแตสท,๐)/ฮจ(๐แตหกแต,๐) )." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |๐โฟแตสทโฉ
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสทโฉ = |๐แตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    //๐ = ๐
    if(visible_config.n_rows == 1 && flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต))." << std::endl;
      std::abort();

    }
    //๐ = ๐
    if(visible_config.n_rows != 1 && flipped_visible_site.n_cols != 2){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต))." << std::endl;
      std::abort();

    }

    //Function variables
    std::complex <double> log_vv(0.0, 0.0);  //Storage variable for the visible terms
    std::complex <double> log_theta(0.0, 0.0);  //Storage variable for the old theta angles
    std::complex <double> log_theta_prime(0.0, 0.0);  //Storage variable for the new theta angles

    //Change due to the visible layer
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++){

      if(flipped_visible_site.n_cols == 1)  //๐ = ๐
        log_vv -= _alpha(flipped_visible_site(j_row, 0)) * double(visible_config(0, flipped_visible_site(j_row, 0)));
      else if(flipped_visible_site.n_cols == 2){  //๐ = ๐

        log_vv -= _alpha(flipped_visible_site(j_row, 0) * visible_config.n_cols + flipped_visible_site(j_row, 1)) *
                  double(visible_config(flipped_visible_site(j_row, 0), flipped_visible_site(j_row, 1)));
      }
      else{

        std::cerr << " ##SizeError: bad construction of the new quantum configuration |๐โฟแตสทโฉ." << std::endl;
        std::cerr << "   Failed to compute ๐๐๐(ฮจ(๐โฟแตสท) / ฮจ(๐แตหกแต))." << std::endl;
        std::abort();

      }

    }
    log_vv *= 2.0;

    //Change due to the visible-hidden interactions
    for(unsigned int k = 0; k < _M; k++){

      log_theta = _Theta(k);  //speed-up the calculation with the Look-up table
      log_theta_prime = log_theta;
      for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++){

        if(flipped_visible_site.n_cols == 1)  //๐ = ๐
          log_theta_prime -= 2.0 * double(visible_config(0, flipped_visible_site(j_row, 0))) * _alpha(_N + _M + flipped_visible_site(j_row, 0)*_M + k);
        else if(flipped_visible_site.n_cols == 2){  //๐ = ๐

          log_theta_prime -= 2.0 * double(visible_config(flipped_visible_site(j_row, 0), flipped_visible_site(j_row, 1))) *
                             _alpha(_N + _M + (flipped_visible_site(j_row, 0) * visible_config.n_cols + flipped_visible_site(j_row, 1)) * _M + k);

        }
        else{

          std::cerr << " ##SizeError: bad construction of the new quantum configuration |๐โฟแตสทโฉ." << std::endl;
          std::cerr << "   Failed to compute ๐๐๐( ฮจ(๐โฟแตสท,๐)/ฮจ(๐แตหกแต,๐) )." << std::endl;
          std::abort();

        }

      }
      log_vv += this -> lncosh(log_theta_prime) - this -> lncosh(log_theta);

    }

    return log_vv;

  }

}


std::complex <double> RBM :: PhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_config));

}


std::complex <double> RBM :: logPsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> logPhi(visible_config, hidden_ket);

}


std::complex <double> RBM :: PsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> Phi(visible_config, hidden_ket);

}


std::complex <double> RBM :: logPsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                   const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                   const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                   std::string option) const {

  return this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


std::complex <double> RBM :: PsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                std::string option) const {

  return this -> PhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


double RBM :: PMetroNew_over_PMetroOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                       const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                       const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                       std::string option) const {

  return std::norm(this -> PsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                              hidden_ket, flipped_ket_site,
                                              hidden_bra, flipped_bra_site));

}


void RBM :: LocalOperators(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) {

  /*#######################################################################*/
  //  Calculates the local operators associated to the
  //  variational parameters ๐ = {๐,๐,๐} on the sampled
  //  quantum configuration |๐โฉ.
  //  In the case of the RBM ansatz the local parameters are ๐(๐,๐) = ๐(๐)
  //
  //        โข ๐๐ฟ โโ ๐(๐) = ฯ๐ฟแถป
  //        โข ๐๐ โโ ๐(๐) = ๐ก๐๐โ(๐ณ๐(๐))
  //        โข [๐]๐ฟ๐ โโ ๐(๐) = ฯ๐ฟแถปโข๐ก๐๐โ(๐ณ๐(๐))
  //
  //  and are ๐๐จ๐ฆ๐ฉ๐ฅ๐๐ฑ number once evaluated!
  //  These operators are necessary to compute the Quantum
  //  Geometric Tensor and the Gradient during the
  //  stochastic optimization procedure.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: the ๐๐๐๐๐๐_๐ค๐๐ญ and ๐๐๐๐๐๐_๐๐ซ๐ arguments are useless for the
  //        RBM ansatz, which does not depend explicitly on the
  //        hidden variables.
  /*#######################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem !=_N ){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐)." << std::endl;
    std::abort();

  }

  //Local operators for the visible bias ๐
  for(unsigned int j_row = 0; j_row < visible_config.n_rows; j_row++){

    for(unsigned int j_col = 0; j_col < visible_config.n_cols; j_col++)
      _LocalOperators(j_row * visible_config.n_cols + j_col, 0) = double(visible_config(j_row, j_col));

  }

  //Local operators for the hidden bias ๐
  for(unsigned int k = 0; k < _M; k++)
    _LocalOperators(_N + k, 0) = std::tanh(_Theta(k));

  //Local operators for the visible-hidden interaction strength ๐
  for(unsigned int m_row = 0; m_row < visible_config.n_rows; m_row++){

    for(unsigned int m_col = 0; m_col < visible_config.n_cols; m_col++){

      for(unsigned int n = 0; n < _M; n++)
        _LocalOperators(_N + _M + (m_row * visible_config.n_cols + m_col) * _M + n, 0) = double(visible_config(m_row, m_col)) * std::tanh(_Theta(n));

    }

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/*****************************************  ๐๐๐๐๐๐๐-๐๐๐๐๐๐๐๐๐ (quasi)-๐ฎ๐๐๐ in ๐ฑ = ๐  ****************************************/
/*******************************************************************************************************************************/
quasi_uRBM :: quasi_uRBM(unsigned int n_visible, bool phi_option, int rank) {

  /*######################################################################################################*/
  //  Random-based constructor.
  //  Initializes the (quasi)-uRBM variational parameters ๐ฅ = {๐,๐,๐,๐} = {๐,๐} to
  //  some small random numbers.
  //
  //  Imposing periodic boundary conditions we have
  //
  //        ๐ complex phase ๐
  //        ๐ญ nearest-neighbors visible-visible interaction strength weights ๐ = {ฮทโฑผ};
  //        ๐ญ nearest-neighbors hidden-hidden interaction strength weights ๐ = {ฯโฑผ};
  //        ๐ญ local visible-hidden interaction strength weights ๐ = {๐คโฑผ};
  //
  //  We remember only in the special case of ๐ dimension the size of the sets of intra- and extra-layer
  //  connections is the same, since in ๐ dimension the number of nearest-neighbors site is ๐.
  /*######################################################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a 1D n.n. (quasi)-uRBM wave function with randomly initialized variational parameters ๐ฅ = { ๐,๐,๐,๐ }." << std::endl;

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Shadow";
  _if_phi = phi_option;
  _N = n_visible;
  _alpha.set_size(3 * _N);
  _LocalOperators.set_size(3 * _N, 2);  //NฬฒOฬฒTฬฒEฬฒ: ๐_๐ = ๐, so we do not save it in memory
  if(_if_phi) {

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else
    _phi = 0.0;
  for(unsigned int p = 0; p < _alpha.n_elem; p++){

    _alpha(p).real(_rnd.Gauss(0.0, 0.001));
    _alpha(p).imag(_rnd.Gauss(0.0, 0.001));

  }

  if(rank == 0){

    std::cout << " (quasi)-uRBM ansatz correctly initialized with random interactions." << std::endl;
    std::cout << " Number of visible variables = " << _N << "." << std::endl;
    std::cout << " Number of hidden variables = " << _N << "." << std::endl;
    std::cout << " Density of the (quasi)-uRBM ansatz = " << this -> density() << "." << std::endl << std::endl;

  }

}


quasi_uRBM :: quasi_uRBM(std::string file_wf, bool phi_option, int rank) {

  /*#############################################################################*/
  //  File-based constructor.
  //  Initializes the (quasi)-uRBM variational parameters ๐ฅ = {๐,๐,๐,๐} = {๐,๐}
  //  from a given external file in '.wf' format; this can be useful
  //  in a second moment during a check phase after the stochastic
  //  optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#############################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a 1D n.n. (quasi)-uRBM wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the (quasi)-uRBM variational parameters ๐ฅ = { ๐,๐,๐,๐ } from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _if_phi = phi_option;
  input_wf >> _N;
  input_wf >> _phi;
  if(_if_phi == false && _phi != 0.0) {

    std::cerr << " ##ValueError: value for the variational phase not compatible with the ansatz construction option." << std::endl;
    std::cerr << "   Failed to construct the variational quantum state." << std::endl;
    std::abort();

  }
  if(!input_wf.good() || _N < 0){

    std::cerr << " ##FileError: invalid construction of the 1D n.n. (quasi)-uRBM ansatz." << std::endl;
    std::abort();

  }
  _type = "Shadow";
  _alpha.set_size(3 * _N);
  _LocalOperators.set_size(3 * _N, 2);  //NฬฒOฬฒTฬฒEฬฒ: ๐_๐ = ๐, so we do not save it in memory
  for(unsigned int p = 0; p < _alpha.n_elem; p++)
    input_wf >> _alpha(p);

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " (quasi)-uRBM Ansatz correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of visible neurons = " << _N << "." << std::endl;
      std::cout << " Number of hidden neurons = " << _N << "." << std::endl;
      std::cout << " Density of the (quasi)-uRBM ansatz = " << this -> density() << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


quasi_uRBM :: ~quasi_uRBM() {

  //_rnd.SaveSeed();

}


std::complex <double> quasi_uRBM :: eta_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the visible-visible interaction strength vector ๐." << std::endl;
    std::cerr << "   Element ฮทโฑผ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(j);

}


std::complex <double> quasi_uRBM :: rho_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the hidden-hidden interaction strength vector ๐." << std::endl;
    std::cerr << " Element ฯโฑผโ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(_N + j);

}


std::complex <double> quasi_uRBM :: omega_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the visible-hidden interaction strength vector ๐." << std::endl;
    std::cerr << " Element ๐คโฑผ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(2 * _N + j);

}


void quasi_uRBM :: print_eta() const {

  std::cout << "\n=================================================" << std::endl;
  std::cout << "quasi_uRBM visible-visible interaction vector ๐" << std::endl;
  std::cout << "=================================================" << std::endl;
  for(unsigned int j = 0; j < _N; j++){

    std::cout << _alpha(j).real();
    if(_alpha(j).imag() >= 0)
      std::cout << " + i" << _alpha(j).imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha(j).imag() << "  " << std::endl;

  }

}


void quasi_uRBM :: print_rho() const {

  std::cout << "\n===============================================" << std::endl;
  std::cout << "quasi_uRBM Hidden-Hidden interaction vector ๐" << std::endl;
  std::cout << "===============================================" << std::endl;
  for(unsigned int j = 0; j < _N; j++){

    std::cout << _alpha(_N + j).real();
    if(_alpha(_N + j).imag() >= 0)
      std::cout << " + i" << _alpha(_N + j).imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha(_N + j).imag() << "  " << std::endl;

  }

}


void quasi_uRBM :: print_omega() const {

  std::cout << "\n================================================" << std::endl;
  std::cout << "quasi_uRBM visible-hidden interaction vector ๐" << std::endl;
  std::cout << "================================================" << std::endl;
  for(unsigned int j = 0; j < _N; j++){

    std::cout << _alpha(2 * _N + j).real();
    if(_alpha(2 * _N + j).imag() >= 0)
      std::cout << " + i" << _alpha(2 * _N + j).imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha(2 * _N + j).imag() << "  " << std::endl;

  }

}


double quasi_uRBM :: I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*######################################################################*/
  //  Computes the value of the angle
  //
  //        โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ') = ฮฃโ (๐โ(๐,๐) - ๐โ(๐,๐กห)) โข ๐ผโแดต
  //
  //  on the given sampled configuration |๐ ๐ ๐หโฉ. This angle enters
  //  in the determination of the Monte Carlo averages estimation
  //  for the quantum observable during the stochastic optimization.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: the contribution of the variational parameter ๐
  //        is not to be included in the sum defining โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ').
  /*######################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh = 0.0;  //Storage variable for the hidden-hidden terms
  double II_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      II_hh += this -> rho_j(j).imag() * double(hidden_ket(0, j) * hidden_ket(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // ฯแดต_๐ญ โข (๐ฝ_๐ญโข๐ฝ_๐ข - ๐ฝห_๐ญโข๐ฝห_๐ข)
      II_vh += this -> omega_j(j).imag() * double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // ฯแดต_๐ญโข๐_๐ญ โข (๐ฝ_๐ญ - ๐ฝห_๐ญ)

    }
    else{

      II_hh += this -> rho_j(j).imag() * double(hidden_ket(0, j) * hidden_ket(0, j+1) - hidden_bra(0, j) * hidden_ket(0, j+1));  // ฯแดต๐ฟ โขย?(๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ - ๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ)
      II_vh += this -> omega_j(j).imag() * double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // ฯแดต๐ฟโข๐๐ฟ โข (๐ฝ๐ฟ - ๐ฝห๐ฟ)

    }

  }

  return II_hh + II_vh;

}


double quasi_uRBM :: cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::cos(this -> I_minus_I(visible_config, hidden_ket, hidden_bra));

}


double quasi_uRBM :: sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::sin(this -> I_minus_I(visible_config, hidden_ket, hidden_bra));

}


std::complex <double> quasi_uRBM :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*##########################################################################################################*/
  //  Computes ๐๐๐(ฮฆ(๐,๐,๐)) with
  //
  //        ฮฆ(๐,๐,๐) = โฏ๐๐(ฮฃโ ๐โ(๐,๐)ฮฑโ)
  //
  //  ฮฆ is that part of variational Shadow ansatz that appears in the ๐๐๐ calculation
  //  of a local quantum observables, i.e.
  //
  //        ๐ช(๐, ๐) = โจฮจ(๐, ๐)| ๐ช |ฮจ(๐, ๐)โฉ
  //                = ฮฃ๐ฃ ฮจโ(๐, ๐, ๐) โข โจ๐| ๐ช |ฮจ(๐, ๐)โฉ
  //                = ฮฃ๐ฃ โฏ๐๐(๐) โข ฮฃโ ฮฆโ(๐,๐,๐) โข โจ๐| ๐ช |ฮจ(๐, ๐)โฉ
  //                = ฮฃ๐ฃฮฃโฮฃโห โฏ๐๐(2โ{๐}) โข ฮฆโ(๐,๐,๐) โข ฮฆ(๐,๐ห,๐) โข ฮฃ๐ฃห โจ๐| ๐ช |๐หโฉโข(ฮฆ(๐ห,๐ห,๐) / ฮฆ(๐,๐ห,๐))
  //                = ฮฃ๐ฃฮฃโฮฃโห ๐(๐ฃ, ๐ฝ, ๐ฝห) โข ๐ชหกแตแถ(๐ฃ, ๐ฝห)
  //
  //  and plays the same role as, for example, the entire wave function in the ๐๐๐ case,
  //  appearing as the ratio
  //
  //        ฮฆ(๐ห,๐ห,๐) / ฮฆ(๐,๐ห,๐)
  //
  //  in the calculation of ๐ชหกแตแถ(๐ฃ, ๐ฝ').
  //
  //  NฬฒOฬฒTฬฒEฬฒ: the ๐๐๐๐๐๐_๐๐จ๐ง๐๐ข๐? argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                ฮฆ(๐,๐,๐)
  //                   or
  //                ฮฆ(๐,๐ห,๐).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ฮฆ(๐,๐,๐)." << std::endl;
    std::abort();

  }
  // |๐โฉ or โจ๐ห|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ฮฆ(๐,๐,๐)." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_vv = 0.0;  //Storage variable for the visible-visible terms
  std::complex <double> log_hh = 0.0;  //Storage variable for the hidden-hidden terms
  std::complex <double> log_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      log_vv += this -> eta_j(j) * double(visible_config(0, j) * visible_config(0, 0));  // ฮท_๐ญ โขย?๐_๐ญโข๐_๐ข
      log_hh += this -> rho_j(j) * double(hidden_config(0, j) * hidden_config(0, 0));  // ฯ_๐ญ โขย?๐ฝ_๐ญโข๐ฝ_๐ข or ฯ_๐ญ โขย?๐ฝห_๐ญโข๐ฝห_๐ข
      log_vh += this -> omega_j(j) * double(visible_config(0, j) * hidden_config(0, j));  // ฯ_๐ญ โขย?๐_๐ญโข๐ฝ_๐ญ or ฯ_๐ญ โขย?๐_๐ญโข๐ฝห_๐ญ

    }
    else{

      log_vv += this -> eta_j(j) * double(visible_config(0, j) * visible_config(0, j+1));  // ฮท๐ฟ โข ๐๐ฟโข๐๐ฟ+๐ฃ
      log_hh += this -> rho_j(j) * double(hidden_config(0, j) * hidden_config(0, j+1));  // ฯ๐ฟ โขย?๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ or ฯ๐ฟ โขย?๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ
      log_vh += this -> omega_j(j) * double(visible_config(0, j) * hidden_config(0, j));  // ฯ๐ฟ โขย?๐๐ฟโข๐ฝ๐ฟ or ฯ๐ฟ โข ๐๐ฟโข๐ฝห๐ฟ

    }

  }

  return log_vv + log_hh + log_vh;

}


std::complex <double> quasi_uRBM :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


std::complex <double> quasi_uRBM :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                          const Mat <int>& hidden_config) const {

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐( ฮฆ(๐โฟแตสท,๐,๐)/ฮฆ(๐แตหกแต,๐,๐) )." << std::endl;
    std::abort();

  }
  // |๐โฉ or โจ๐ห|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(ฮฆ(๐โฟแตสท) / ฮฆ(๐แตหกแต))." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |๐โฟแตสทโฉ
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสทโฉ = |๐แตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new visible configuration does not match with the number of visible variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(ฮฆ(๐โฟแตสท) / ฮฆ(๐แตหกแต))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |๐โฟแตสทโฉ
    std::complex <double> log_vh = 0.0;  //Storage variable for the visible-hidden terms
    std::complex <double> log_vv = 0.0;  //Storage variable for the visible-visible terms

    //Computes the visible-hidden terms: ฮฃ๐ฟ ฯ๐ฟโข๐๐ฟแตหกแตโข๐ฝ๐ฟ with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_visible_site(j_row, 0)) * double(visible_config(0, flipped_visible_site(j_row, 0)) * hidden_config(0, flipped_visible_site(j_row, 0)));

    //Computes the visible-visible terms: ฮฃ๐ฟ ฮท๐ฟโข(๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += this -> eta_j(j) * double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // ฮท_๐ญโข(๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
      else
        log_vv += this -> eta_j(j) * double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // ฮท๐ฟโข(๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)

    }

    return -2.0 * log_vh + log_vv;

  }

}


std::complex <double> quasi_uRBM :: PhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                       const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_config));

}


std::complex <double> quasi_uRBM :: logPsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*################################################################################################*/
  //  Computes the value of the complex natural logarithm of the 'classical' part ๐ of the total
  //  probability distribution
  //
  //        ๐ซ(๐,๐,๐ห,๐ฅ) = ๐(๐,๐,๐ห,๐ฅ) โข [๐๐๐?(โ(๐,๐)-โ(๐,๐ห)) + i๐?๐๐(โ(๐,๐)-โ(๐,๐ห))]
  //
  //  of the enlarged sampling space, i.e. ๐(๐,๐,๐ห,๐ฅ).
  //  The total probability distribution is defined through the sum
  //
  //        ฮฃ๐ฃฮฃ๐ฝฮฃ๐ฝห ๐ซ(๐,๐,๐ห,๐ฅ) = ฮฃ๐ฃ |ฮจ(๐,๐ฅ)|^2 = ๐
  //
  //  where
  //
  //        ฮจ(๐,๐ฅ) = ฮจ(๐,๐,๐) = โฏ๐๐(๐)โขฮฃโ โฏ๐๐(ฮฃโ ๐โ(๐,๐)ฮฑโ)
  //                = โฏ๐๐(๐) โข โฏ๐๐{ฮฃโฑผโ ฮทโฑผโ๐ฃโฑผ๐ฃโ} โข ฮฃโ โฏ๐๐(ฮฃโฑผโ ฯโฑผโ๐ฝโฑผ๐ฝโ + ฮฃโฑผ ๐โฑผ๐ฃโฑผ๐ฝโ}
  //
  //  is the variational Shadow wave function characterized by the variational
  //  parameters {๐, ๐} = {๐, ๐, ๐, ๐}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        ฮฃ๐ฃฮฃ๐ฝฮฃ๐ฝ' ๐(๐,๐,๐ห,๐ฅ) ๐ป(๐,๐,๐ห) = โจ๐ป(๐,๐,๐ห)โฉ๐ / โจ๐๐๐?(โ(๐,๐)-โ(๐,๐ห))โฉ๐.
  //
  //  So it is clear that the classical probability part ๐(๐,๐,๐ห,๐ฅ) plays the role of
  //  square modulus of the wave function with which to sample the shadow configurations |๐, ๐, ๐หโฉ
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        ๐(๐,๐,๐ห,๐ฅ) = โฏ๐๐(2โ{๐}) โข โฏ๐๐(โ(๐ฃ, ๐ฝ) + โ(๐ฃ, ๐ฝห))
  //
  //  where
  //
  //        โ(๐ฃ, ๐ฝ) + โ(๐ฃ, ๐ฝห) = ฮฃโ (๐โ(๐,๐) + ๐โ(๐,๐ห)) โขย?ฮฑแดฟโ
  //
  //  and it has to be calculated on the current visible configuration |๐โฉ and the sampled
  //  hidden configuration ket |๐โฉ and bra โจ๐ห|.
  /*################################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐ห))." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐ห))." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐ห))." << std::endl;
    std::abort();

  }

  //Function variables
  double log_vv = 0.0;  //Storage variable for the visible-visible terms
  double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
  double log_vh = 0.0;  //Storage variable for the visible-hidden terms
  std::complex <double> log_psi;

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      log_vv += this -> eta_j(j).real() * double(visible_config(0, j) * visible_config(0, 0));  // ฮทแดฟ_๐ญ โข ๐_๐ญโข๐_๐ข
      log_hh += this -> rho_j(j).real() * double(hidden_ket(0, j) * hidden_ket(0, 0) + hidden_bra(0, j) * hidden_bra(0, 0));  // ฯแดฟ_๐ญ โขย?๐ฝ_๐ญโข๐ฝ_๐ข + ๐ฝห_๐ญโข๐ฝห_๐ข
      log_vh += this -> omega_j(j).real() * double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // ฯแดฟ_๐ญ โขย?๐_๐ญโข(๐ฝ_๐ญ + ๐ฝห_๐ญ)

    }
    else{

      log_vv += this -> eta_j(j).real() * double(visible_config(0, j) * visible_config(0, j+1));  // ฮฃ๐ฟ ฮทแดฟ๐ฟ โข ๐๐ฟโข๐๐ฟ+๐ฃ
      log_hh += this -> rho_j(j).real() * double(hidden_ket(0, j) * hidden_ket(0, j+1) + hidden_bra(0, j) * hidden_bra(0, j+1));  // ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ + ๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ
      log_vh += this -> omega_j(j).real() * double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?๐๐ฟโข(๐ฝ๐ฟ + ๐ฝห๐ฟ)

    }

  }

  log_psi.real(2.0 * this -> phi().real() + 2.0 * log_vv + log_hh + log_vh);
  log_psi.imag(0.0);
  return log_psi;

}


std::complex <double> quasi_uRBM :: PsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  //Function variables
  std::complex <double> P;
  P.imag(0.0);
  P.real(std::exp(this -> logPsiMetro(visible_config, hidden_ket, hidden_bra)).real());

  return P;

}


double quasi_uRBM :: logq_over_q_visible(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                         const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*##############################################################################*/
  //  Computes ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))
  //  evaluated in a new proposed configuration |๐โฟแตสท ๐ ๐หโฉ wrt
  //  the current configuration |๐แตหกแต ๐ ๐หโฉ (at fixed variational parameters ๐ฅ),
  //  where only the visible variables have been changed.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current configuration.
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐โฟแตสท ๐ ๐หโฉ
  if(flipped_visible_site.n_elem==0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสท ๐ ๐หโฉ = |๐แตหกแต ๐ ๐หโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |๐โฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |๐โฟแตสทโฉ
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: ฮฃ๐ฟ ฮทแดฟ๐ฟ โข (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // ฮทแดฟ_๐ญ โขย?(๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
      else
        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // ฮทแดฟ๐ฟ โขย?(๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)

    }

    //Computes the visible-hidden term: ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?๐๐ฟแตหกแตโข(๐ฝ๐ฟ + ๐ฝห๐ฟ) with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_visible_site(j_row, 0)).real() * double(visible_config(0, flipped_visible_site(j_row, 0)) *
                (hidden_ket(0, flipped_visible_site(j_row, 0)) + hidden_bra(0, flipped_visible_site(j_row, 0))));

    return 2.0 * log_vv - 2.0 * log_vh;

  }

}


double quasi_uRBM :: q_over_q_visible(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                      const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::exp(this -> logq_over_q_visible(visible_config, flipped_visible_site, hidden_ket, hidden_bra));

}


double quasi_uRBM :: logq_over_q_ket(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site) const {

  /*#################################################################################*/
  //  Computes ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))
  //  evaluated in a new proposed configuration |๐ ๐โฟแตสท ๐หโฉ wrt
  //  the current configuration |๐ ๐แตหกแต ๐หโฉ (at fixed variational parameters ๐ฅ),
  //  where only the hidden variables ket have been changed.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed ket configuration is a configuration with a certain number of
  //  flipped spins wrt the old ket configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*#################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐ ๐โฟแตสท ๐หโฉ
  if(flipped_ket_site.n_elem==0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐ ๐โฟแตสท ๐หโฉ = |๐ ๐แตหกแต ๐หโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |๐โฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐ห)/๐(๐,๐แตหกแต,๐ห))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |๐โฟแตสทโฉ
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?(๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // ฯแดฟ_๐ญ โขย?(๐ฝ_๐ญโฟแตสทโข๐ฝ_๐ขโฟแตสท - ๐ฝ_๐ญแตหกแตโข๐ฝ_๐ขแตหกแต)
      else
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // ฯแดฟ๐ฟ โขย?(๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)

    }

    //Computes the visible-hidden term: ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?๐๐ฟแตหกแตโข๐ฝ๐ฟแตหกแต with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ค๐๐ญ_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_ket_site(j_row, 0)).real() * double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));

    return log_hh - 2.0 * log_vh;

  }

}


double quasi_uRBM :: q_over_q_ket(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site) const {

  return std::exp(this -> logq_over_q_ket(visible_config, hidden_ket, flipped_ket_site));

}


double quasi_uRBM :: logq_over_q_bra(const Mat <int>& visible_config, const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  /*#################################################################################*/
  //  Computes ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))
  //  evaluated in a new proposed configuration |๐ ๐ ๐หโฟแตสทโฉ wrt
  //  the current configuration |๐ ๐ ๐หแตหกแตโฉ (at fixed variational parameters ๐ฅ),
  //  where only the hidden variables bra have been changed.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed bra configuration is a configuration with a certain number of
  //  flipped spins wrt the old bra configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*#################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐ ๐ ๐หโฟแตสทโฉ
  if(flipped_bra_site.n_elem==0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐ ๐ ๐หโฟแตสทโฉ = |๐ ๐ ๐หแตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration โจ๐หโฟแตสท|." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // โจ๐หโฟแตสท|
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: ฮฃ๐ฟ ฯแดฟ๐ฟ โข (๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // ฯแดฟ_๐ญ โข (๐ฝห_๐ญโฟแตสทโข๐ฝห_๐ขโฟแตสท - ๐ฝห_๐ญแตหกแตโข๐ฝห_๐ขแตหกแต)
      else
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // ฯแดฟ๐ฟ โข (๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)

    }

    //Computes the visible-hidden term: ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?๐๐ฟแตหกแตโข๐ฝห๐ฟแตหกแต with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐๐ซ๐_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_bra_site(j_row, 0)).real() * double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    return log_hh - 2.0 * log_vh;

  }

}


double quasi_uRBM :: q_over_q_bra(const Mat <int>& visible_config, const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_bra(visible_config, hidden_bra, flipped_bra_site));

}


double quasi_uRBM :: logq_over_q_equal_site(const Mat <int>& visible_config, const Mat <int>& flipped_equal_site,
                                            const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*###################################################################################*/
  //  Computes ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต))
  //  evaluated in a new proposed configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ wrt
  //  the current configuration |๐แตหกแต ๐แตหกแต ๐หแตหกแตโฉ (at fixed variational parameters ๐ฅ).
  //  In this case we decide to move the spins located at the same (randomly
  //  choosen) lattice sites for all the three variables ๐, ๐, ๐ห.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current spin configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*###################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ
  if(flipped_equal_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ = |๐แตหกแต ๐แตหกแต ๐หแตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_equal_site);  // |๐โฟแตสทโฉ
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_equal_site);  // |๐โฟแตสทโฉ
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_equal_site);  // |๐หโฟแตสทโฉ
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms

    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1){

        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // ฮทแดฟ_๐ญ โขย?(๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // ฯแดฟ_๐ญ โขย?(๐ฝ_๐ญโฟแตสทโข๐ฝ_๐ขโฟแตสท - ๐ฝ_๐ญแตหกแตโข๐ฝ_๐ขแตหกแต)
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // ฯแดฟ_๐ญ โขย?(๐ฝห_๐ญโฟแตสทโข๐ฝห_๐ขโฟแตสท - ๐ฝห_๐ญแตหกแตโข๐ฝห_๐ขแตหกแต)

      }
      else{

        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // ฮทแดฟ๐ฟ โข (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // ฯแดฟ๐ฟ โขย?(๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // ฯแดฟ๐ฟ โขย?(๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)

      }

    }

    return 2.0 * log_vv + log_hh;

  }

}


double quasi_uRBM :: q_over_q_equal_site(const Mat <int>& visible_config, const Mat <int>& flipped_equal_site,
                                         const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::exp(this -> logq_over_q_equal_site(visible_config, flipped_equal_site, hidden_ket, hidden_bra));

}


double quasi_uRBM :: logq_over_q_braket(const Mat <int>& visible_config,
                                        const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                        const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  /*################################################################################*/
  //  Computes ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต))
  //  evaluated in a new proposed configuration |๐ ๐โฟแตสท ๐หโฟแตสทโฉ wrt
  //  the current configuration |๐ ๐แตหกแต ๐หแตหกแตโฉ (at fixed variational parameters ๐ฅ).
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped auxiliary spins wrt the current spin configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐ ๐โฟแตสท ๐หโฟแตสทโฉ
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐ ๐โฟแตสท ๐หโฟแตสทโฉ = |๐ ๐แตหกแต ๐หแตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |๐ ๐โฟแตสท ๐หโฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |๐โฟแตสทโฉ
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // โจ๐หโฟแตสท|
    double log_ket = 0.0;  //Storage variable for the hidden-hidden ket terms
    double log_bra = 0.0;  //Storage variable for the hidden-hidden bra terms
    double log_vk = 0.0;  //Storage variable for the visible-hidden ket terms
    double log_vb = 0.0;  //Storage variable for the visible-hidden bra terms

    //๐ฃ๐๐ ๐ธ๐ถ๐โฏ:  |๐โฟแตสทโฉ โ? |๐แตหกแตโฉ & |๐หโฟแตสทโฉ = |๐หแตหกแตโฉ
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the hidden-hidden terms only for the ket: ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?(๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_ket += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));
        else
          log_ket += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: ฮฃ๐ฟ ฯแดฟ๐ฟ โข ๐๐ฟแตหกแตโข๐ฝ๐ฟแตหกแต
      for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += this -> omega_j(flipped_ket_site(j_row, 0)).real() * double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));

    }

    //๐ค๐๐ ๐ธ๐ถ๐โฏ:  |๐โฟแตสทโฉ = |๐แตหกแตโฉ & |๐หโฟแตสทโฉ โ? |๐หแตหกแตโฉ
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the hidden-hidden terms only for the ket: ฮฃ๐ฟ ฯแดฟ๐ฟ โขย?(๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_bra += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));
        else
          log_bra += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: ฮฃ๐ฟ ฯแดฟ๐ฟ โข ๐๐ฟแตหกแตโข๐ฝ๐ฟแตหกแต
      for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += this -> omega_j(flipped_bra_site(j_row, 0)).real() * double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    }

    //๐ฅ๐๐ ๐ธ๐ถ๐โฏ:  |๐โฟแตสทโฉ โ? |๐แตหกแตโฉ & |๐หโฟแตสทโฉ โ? |๐หแตหกแตโฉ
    else if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem != 0){

      //Computes the hidden-hidden terms
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1){

          log_ket += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));
          log_bra += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));

        }
        else{

          log_ket += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));
          log_bra += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));

        }

      }

      //Computes the visible-hidden terms
      for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += this -> omega_j(flipped_ket_site(j_row, 0)).real() * double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));
      for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += this -> omega_j(flipped_bra_site(j_row, 0)).real() * double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    }

    else{

      std::cerr << " ##OptionError: something went wrong in the determination of ๐๐๐( ๐(๐,๐โฟแตสท,๐หโฟแตสท)/๐(๐,๐แตหกแต,๐หแตหกแต) )." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต))." << std::endl;
      std::abort();

    }

    return log_ket + log_bra - 2.0 * (log_vk + log_vb);

  }

}


double quasi_uRBM :: q_over_q_braket(const Mat <int>& visible_config,
                                     const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                     const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_braket(visible_config,
                                             hidden_ket, flipped_ket_site,
                                             hidden_bra, flipped_bra_site));

}


std::complex <double> quasi_uRBM :: logPsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                          const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                          const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                          std::string option) const {

  //Function variables
  std::complex <double> logPoP;
  logPoP.imag(0.0);  //In this case the acceptance is a pure real number

  if(option == "visible")
    logPoP.real(this -> logq_over_q_visible(visible_config, flipped_visible_site, hidden_ket, hidden_bra));
  else if(option == "ket")
    logPoP.real(this -> logq_over_q_ket(visible_config, hidden_ket, flipped_ket_site));
  else if(option == "bra")
    logPoP.real(this -> logq_over_q_bra(visible_config, hidden_bra, flipped_bra_site));
  else if(option == "equal site")
    logPoP.real(this -> logq_over_q_equal_site(visible_config, flipped_visible_site, hidden_ket, hidden_bra));
  else if(option == "braket")
    logPoP.real(this -> logq_over_q_braket(visible_config, hidden_ket, flipped_ket_site, hidden_bra, flipped_bra_site));
  else{

    std::cerr << " ##OptionError: no available option as function argument." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต))." << std::endl;
    std::abort();

  }

  return logPoP;

}


std::complex <double> quasi_uRBM :: PsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                       const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                       const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                       std::string option) const {

  //Function variables
  std::complex <double> PoP;
  PoP.imag(0.0);

  PoP.real(std::exp(this -> logPsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                                  hidden_ket, flipped_ket_site,
                                                  hidden_bra, flipped_bra_site,
                                                  option).real()));
  return PoP;

}


double quasi_uRBM :: PMetroNew_over_PMetroOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                              const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                              const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                              std::string option) const {

  /*######################################################################*/
  //  NฬฒOฬฒTฬฒEฬฒ: in the Shadow ansatz the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)
  //
  /*######################################################################*/

  std::complex <double> p = this -> PsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                                       hidden_ket, flipped_ket_site,
                                                       hidden_bra, flipped_bra_site,
                                                       option);
  if(p.imag() != 0.0){

    std::cerr << " ##ValueError: the imaginary part of the Metropolis-Hastings acceptance probability must be zero!" << std::endl;
    std::cerr << "   Failed to compute the Metropolis-Hastings acceptance probability." << std::endl;
    std::abort();

  }

  return p.real();

}


void quasi_uRBM :: LocalOperators(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameters
  //  ๐ on the sampled enlarged quantum configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ.
  //  In the case of the (quasi)-uRBM ansatz the local parameters are ๐(๐,๐)
  //
  //        โข ฮท๐ฟ โน--โบ ๐(๐,๐) = ๐(๐) = ๐ฃ๐ฟ โขย?๐ฃ๐ฟ+๐ฃ
  //        โข ฯ๐ฟ โน--โบ ๐(๐,๐) = ๐(๐) = ๐ฝ๐ฟ โขย?๐ฝ๐ฟ+๐ฃ      ๐(๐,๐ห) = ๐(๐ห) = ๐ฝห๐ฟ โขย?๐ฝห๐ฟ+๐ฃ
  //        โข ฯ๐ฟ โน--โบ ๐(๐,๐) = ๐ฝ๐ฟ โข ๐ฃ๐ฟ                ๐(๐,๐ห) = ๐ฝห๐ฟ โข ๐ฃ๐ฟ
  //
  //  It is important to note that in the Shadow wave function the local operators
  //  (which are a geometric properties of the wave function itself) related to
  //  the hidden-hidden interactions and the hidden-visible interaction, respectively
  //  depend also on the auxiliary variables, and not only on the actual quantum degrees
  //  of freedom of the system.
  //  These operators are necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  //  We remember that in the Shadow case the local operators are real.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐) and ๐(๐,๐ห)." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐) and ๐(๐,๐ห)." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐) and ๐(๐,๐ห)." << std::endl;
    std::abort();

  }

  //Local operators for the visible-visible interactions strength ๐
  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      _LocalOperators(j, 0) = double(visible_config(0, j) * visible_config(0, 0));
      _LocalOperators(j, 1) = double(visible_config(0, j) * visible_config(0, 0));

    }
    else{

    _LocalOperators(j, 0) = double(visible_config(0, j) * visible_config(0, j+1));
    _LocalOperators(j, 1) = double(visible_config(0, j) * visible_config(0, j+1));

    }

  }

  //Local operators for the hidden-hidden interactions strength ๐
  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      _LocalOperators(_N + j, 0) = double(hidden_ket(0, j) * hidden_ket(0, 0));
      _LocalOperators(_N + j, 1) = double(hidden_bra(0, j) * hidden_bra(0, 0));

    }
    else{

      _LocalOperators(_N + j, 0) = double(hidden_ket(0, j) * hidden_ket(0, j+1));
      _LocalOperators(_N + j, 1) = double(hidden_bra(0, j) * hidden_bra(0, j+1));

    }

  }

  //Local operators for the hidden-hidden interactions strength ๐
  for(unsigned int j = 0; j < _N; j++){

    _LocalOperators(2 + _N + j, 0) = double(visible_config(0, j) * hidden_ket(0, j));
    _LocalOperators(2 + _N + j, 1) = double(visible_config(0, j) * hidden_bra(0, j));

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  ๐๐๐๐๐๐๐๐๐-๐๐๐๐๐๐ ๐๐๐๐ in ๐ฑ = ๐  ********************************************/
/*******************************************************************************************************************************/
BS_NNQS :: BS_NNQS(unsigned int n_visible, bool phi_option, int rank) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the Baeriswyl-Shadow variational parameters ๐ฅ = {๐, ฮท, ฯ, ฯ} = {๐, ๐} to
  //  some small random numbers.
  //  In particular we set the imaginary part of the parameters to exactly zero, while the
  //  real part is chosen randomly.
  //
  //  In this case we have only ๐ฏ parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        ๐ complex phase ๐
  //        ๐ nearest-neighbors visible-visible interaction strength weights ฮท;
  //        ๐ nearest-neighbors hidden-hidden interaction strength weights ฯ;
  //        ๐ local visible-hidden interaction strength weights ฯ.
  //
  //  Note that in this case the number of variational parameters remains equal to ๐ฏ for any system size ๐ญ.
  /*########################################################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a 1D Baeriswyl-Shadow wave function with randomly initialized variational parameters ๐ฅ = { ๐, ฮท, ฯ, ฯ }." << std::endl;

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Shadow";
  _if_phi = phi_option;
  _N = n_visible;
  _alpha.set_size(3);
  _LocalOperators.set_size(3, 2);  //NฬฒOฬฒTฬฒEฬฒ: ๐_๐ = ๐, so we do not save it in memory
  if(_if_phi){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(0.0);

  }
  else
    _phi = 0.0;
  for(unsigned int p = 0; p < _alpha.n_elem; p++){

    _alpha(p).real(_rnd.Gauss(0.0, 0.001));
    _alpha(p).imag(0.0);

  }

  if(rank == 0){

    std::cout << " Baeriswyl-Shadow NNQS ansatz correctly initialized with random interactions." << std::endl;
    std::cout << " Number of visible variables = " << _N << "." << std::endl;
    std::cout << " Number of hidden variables = " << _N << "." << std::endl;
    std::cout << " Density of the Baeriswyl-Shadow NNQS ansatz = " << this -> density() << "." << std::endl << std::endl;

  }

}


BS_NNQS :: BS_NNQS(std::string file_wf, bool phi_option, int rank) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the Baeriswyl-Shadow variational parameters
  //  ๐ฅ = {๐, ฮท, ฯ, ฯ} = {๐, ๐} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0)
    std::cout << "#Create a 1D Baeriswyl-Shadow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the Baeriswyl-Shadow NNQS variational parameters ๐ฅ = { ๐, ฮท, ฯ, ฯ } from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
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
  std::ifstream input("./input_random_device/seed1.in");
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

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _if_phi = phi_option;
  input_wf >> _N;
  input_wf >> _phi;
  if(_if_phi == false && _phi != 0.0) {

    std::cerr << " ##ValueError: value for the variational phase not compatible with the ansatz construction option." << std::endl;
    std::cerr << "   Failed to construct the variational quantum state." << std::endl;
    std::abort();

  }
  if(!input_wf.good() || _N < 0){

    std::cerr << " ##FileError: invalid construction of the 1D Baeriswyl-Shadow ansatz." << std::endl;
    std::abort();

  }
  _type = "Shadow";
  _alpha.set_size(3);
  _LocalOperators.set_size(3, 2);  //NฬฒOฬฒTฬฒEฬฒ: ๐_๐ = ๐, so we do not save it in memory
  for(unsigned int p = 0; p < _alpha.n_elem; p++)
    input_wf >> _alpha(p);

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Baeriswyl-Shadow NNQS Ansatz correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of visible neurons = " << _N << "." << std::endl;
      std::cout << " Number of hidden neurons = " << _N << "." << std::endl;
      std::cout << " Density of the Baeriswyl-Shadow NNQS ansatz = " << this -> density() << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


BS_NNQS :: ~BS_NNQS() {

  //_rnd.SaveSeed();

}


double BS_NNQS :: I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*######################################################################*/
  //  Computes the value of the angle
  //
  //        โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ') = ฮฃโ (๐โ(๐,๐) - ๐โ(๐,๐กห)) โข ๐ผโแดต
  //
  //  on the given sampled configuration |๐ ๐ ๐หโฉ.
  //  This angle enters in the determination of the Monte Carlo averages
  //  estimation for the quantum observable during the stochastic
  //  optimization.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: the contribution of the variational parameter ๐
  //        is not to be included in the sum defining โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ').
  /*######################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')." << std::endl;
    std::abort();

  }
  // |๐หโฉ
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the angle โ(๐ฃ, ๐ฝ) - โ(๐ฃ, ๐ฝ')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh = 0.0;  //Storage variable for the hidden-hidden terms
  double II_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      II_hh += double(hidden_ket(0, j) * hidden_ket(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // (๐ฝ_๐ญโข๐ฝ_๐ข - ๐ฝห_๐ญโข๐ฝห_๐ข)
      II_vh += double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // ๐_๐ญโข(๐ฝ_๐ญ - ๐ฝห_๐ญ)

    }
    else{

      II_hh += double(hidden_ket(0, j) * hidden_ket(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // (๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ - ๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ)
      II_vh += double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // ๐๐ฟโข(๐ฝ๐ฟ - ๐ฝห๐ฟ)

    }

  }

  return this -> rho().imag() * II_hh + this -> omega().imag() * II_vh;

}


double BS_NNQS :: cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::cos(this -> I_minus_I(visible_config, hidden_ket, hidden_bra));

}


double BS_NNQS :: sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::sin(this -> I_minus_I(visible_config, hidden_ket, hidden_bra));

}


std::complex <double> BS_NNQS :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*##########################################################################################################*/
  //  Computes ๐๐๐(ฮฆ(๐,๐,๐)) with
  //
  //        ฮฆ(๐,๐,๐) = โฏ๐๐(ฮฃโ ๐โ(๐,๐)ฮฑโ)
  //
  //  ฮฆ is that part of variational Shadow ansatz that appears in the ๐๐๐ calculation
  //  of a local quantum observables, i.e.
  //
  //        ๐(๐,๐) = โจฮจ(๐,๐)| ๐ธ |ฮจ(๐,๐)โฉ
  //                = ฮฃ๐ฃ ฮจโ(๐,๐,๐) โข โจ๐| ๐ธ |ฮจ(๐,๐)โฉ
  //                = ฮฃ๐ฃ โฏ๐๐(๐) โข ฮฃโ ฮฆโ(๐,๐,๐) โข โจ๐| ๐ธ |ฮจ(๐,๐)โฉ
  //                = ฮฃ๐ฃฮฃโฮฃโห โฏ๐๐(2๐แดฟ) โข ฮฆโ(๐,๐,๐) โข ฮฆ(๐,๐ห,๐) โข ฮฃ๐ฃห โจ๐| ๐ธ |๐หโฉ โข ฮฆ(๐ห,๐ห,๐) / ฮฆ(๐,๐ห,๐)
  //                = ฮฃ๐ฃฮฃโฮฃโห ๐(๐ฃ, ๐ฝ, ๐ฝห) โข ๐(๐ฃ,๐ฝห)
  //
  //  with ๐ธ a generic quantum observable operator, and plays the same role as, for example, the entire wave
  //  function in the ๐๐๐ case, appearing as the ratio
  //
  //        ฮฆ(๐ห,๐ห,๐) / ฮฆ(๐,๐ห,๐)
  //
  //  in the above calculation.
  //
  //  NฬฒOฬฒTฬฒEฬฒ: the ๐๐๐๐๐๐_๐๐จ๐ง๐๐ข๐? argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                ฮฆ(๐,๐,๐)
  //                   or
  //                ฮฆ(๐,๐ห,๐).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ฮฆ(๐,๐,๐)." << std::endl;
    std::abort();

  }
  // |๐โฉ or โจ๐ห|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ฮฆ(๐,๐,๐)." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_vv = 0.0;  //Storage variable for the visible-visible terms
  std::complex <double> log_hh = 0.0;  //Storage variable for the hidden-hidden terms
  std::complex <double> log_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      log_vv += double(visible_config(0, j) * visible_config(0, 0));  // ๐_๐ญโข๐_๐ข
      log_hh += double(hidden_config(0, j) * hidden_config(0, 0));  // ๐ฝ_๐ญโข๐ฝ_๐ข or ๐ฝห_๐ญโข๐ฝห_๐ข
      log_vh += double(visible_config(0, j) * hidden_config(0, j));  // ๐_๐ญโข๐ฝ_๐ญ or ๐_๐ญโข๐ฝห_๐ญ

    }
    else{

      log_vv += double(visible_config(0, j) * visible_config(0, j+1));  // ๐๐ฟโข๐๐ฟ+๐ฃ
      log_hh += double(hidden_config(0, j) * hidden_config(0, j+1));  // ๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ or ๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ
      log_vh += double(visible_config(0, j) * hidden_config(0, j));  // ๐๐ฟโข๐ฝ๐ฟ or ๐๐ฟโข๐ฝห๐ฟ

    }

  }

  return this -> eta() * log_vv + this -> rho() * log_hh + this -> omega() * log_vh;

}


std::complex <double> BS_NNQS :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


std::complex <double> BS_NNQS :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                       const Mat <int>& hidden_config) const {

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(ฮฆ(๐โฟแตสท,๐) / ฮฆ(๐แตหกแต,๐))." << std::endl;
    std::abort();

  }
  // |๐โฉ or โจ๐ห|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(ฮฆ(๐โฟแตสท,๐) / ฮฆ(๐แตหกแต,๐))." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |๐โฟแตสทโฉ
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสทโฉ = |๐แตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new visible configuration does not match with the number of visible variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(ฮฆ(๐โฟแตสท,๐) / ฮฆ(๐แตหกแต,๐))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |๐โฟแตสทโฉ
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms
    double log_vv = 0.0;  //Storage variable for the visible-visible terms

    //Computes the visible-hidden terms: ฮฃ๐ฟ ๐๐ฟแตหกแตโข๐ฝ๐ฟ with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++)
      log_vh += double(visible_config(0, flipped_visible_site(j_row, 0)) * hidden_config(0, flipped_visible_site(j_row, 0)));

    //Computes the visible-visible terms: ฮฃ๐ฟ (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
      else
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)

    }

    return -2.0 * this -> omega() * log_vh + this -> eta() * log_vv;

  }

}


std::complex <double> BS_NNQS :: PhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                    const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_config));

}


std::complex <double> BS_NNQS :: logPsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*################################################################################################*/
  //  Computes the value of the real natural logarithm of the 'classical' part ๐ of the total
  //  probability distribution
  //
  //        ๐ซ(๐,๐,๐ห) = ๐(๐,๐,๐ห) โข [๐๐๐?(โ(๐,๐)-โ(๐,๐ห)) + ๐๐?๐๐(โ(๐,๐)-โ(๐,๐ห))]
  //                  = ๐ข(๐,๐,๐ห) +  ๐โข๐ฒ(๐,๐,๐ห)
  //
  //  of the enlarged sampling space, i.e. ๐(๐,๐,๐ห).
  //  The total probability distribution is defined through the sum
  //
  //        ฮฃ๐ฃฮฃ๐ฝฮฃ๐ฝห ๐ซ(๐,๐,๐ห) = ฮฃ๐ฃ |ฮจ(๐,๐,๐)|^2 = ๐
  //
  //  where
  //
  //        ฮจ(๐,๐,๐) = โฏ๐๐(๐) โข ฮฃโ โฏ๐๐(ฮฃโ ๐โ(๐,๐)ฮฑโ)
  //                 = โฏ๐๐(๐) โข โฏ๐๐( ฮท โข ฮฃ๐ฟ ๐๐ฟโข๐๐ฟ+๐ฃ ) โข
  //                           โข ฮฃ๐ฝ โฏ๐๐( ฯ โข ฮฃ๐ฟ (๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ) + ฯ โขย?ฮฃ๐ฟ (๐๐ฟโข๐ฝ๐ฟ) )
  //
  //  is the variational Baeriswyl-Shadow wave function characterized by the variational
  //  parameters {๐, ๐} = {๐, ฮท, ฯ, ฯ}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        ฮฃ๐ฃฮฃ๐ฝฮฃ๐ฝ' ๐(๐,๐,๐ห) ๐ป(๐,๐,๐ห) = โจ๐ป(๐,๐,๐ห)โฉ๐ / โจ๐๐๐?(โ(๐,๐)-โ(๐,๐ห))โฉ๐
  //
  //  So it is clear that the classical probability part ๐(๐,๐,๐ห) plays the role of
  //  square modulus of the wave function with which to sample the shadow configurations |๐, ๐, ๐หโฉ
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        ๐(๐,๐,๐ห) = โฏ๐๐(2๐แดฟ) โข โฏ๐๐(โ(๐ฃ, ๐ฝ) + โ(๐ฃ, ๐ฝห))
  //
  //  where
  //
  //        โ(๐ฃ, ๐ฝ) + โ(๐ฃ, ๐ฝห) = ฮฃโ (๐โ(๐,๐) + ๐โ(๐,๐ห)) โขย?๐ผแดฟโ
  //
  //  and it has to be calculated on the current configuration |๐ ๐ ๐หโฉ.
  /*################################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐ห))." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐ห))." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐ห))." << std::endl;
    std::abort();

  }

  //Function variables
  double log_vv = 0.0;  //Storage variable for the visible-visible terms
  double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
  double log_vh = 0.0;  //Storage variable for the visible-hidden terms
  std::complex <double> log_q;

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      log_vv += double(visible_config(0, j) * visible_config(0, 0));  // ๐_๐ญโข๐_๐ข
      log_hh += double(hidden_ket(0, j) * hidden_ket(0, 0) + hidden_bra(0, j) * hidden_bra(0, 0));  // ๐ฝ_๐ญโข๐ฝ_๐ข + ๐ฝห_๐ญโข๐ฝห_๐ข
      log_vh += double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // ๐_๐ญโข(๐ฝ_๐ญ + ๐ฝห_๐ญ)

    }
    else{

      log_vv += double(visible_config(0, j) * visible_config(0, j+1));  // ฮฃ๐ฟ ๐๐ฟโข๐๐ฟ+๐ฃ
      log_hh += double(hidden_ket(0, j) * hidden_ket(0, j+1) + hidden_bra(0, j) * hidden_bra(0, j+1));  // ฮฃ๐ฟ ๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ + ๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ
      log_vh += double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // ฮฃ๐ฟ ๐๐ฟโข(๐ฝ๐ฟ + ๐ฝห๐ฟ)

    }

  }

  log_q.imag(0.0);
  log_q.real(2.0 * this -> phi().real() + 2.0 * this -> eta().real() * log_vv + this -> rho().real() * log_hh + this -> omega().real() * log_vh);
  return log_q;

}


std::complex <double> BS_NNQS :: PsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  //Function variables
  std::complex <double> P;
  P.imag(0.0);
  P.real(std::exp(this -> logPsiMetro(visible_config, hidden_ket, hidden_bra)).real());

  return P;

}


double BS_NNQS :: logq_over_q_visible(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                      const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*##############################################################################*/
  //  Computes ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))
  //  evaluated in a new proposed configuration |๐โฟแตสท ๐ ๐หโฉ wrt
  //  the current configuration |๐แตหกแต ๐ ๐หโฉ (at fixed variational parameters ๐ฅ),
  //  where only the visible variables have been changed.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current configuration.
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐โฟแตสท ๐ ๐หโฉ
  if(flipped_visible_site.n_elem==0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสท ๐ ๐หโฉ = |๐แตหกแต ๐ ๐หโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |๐โฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐,๐ห) / ๐(๐แตหกแต,๐,๐ห))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |๐โฟแตสทโฉ
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: ฮฃ๐ฟ (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
      else
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)

    }

    //Computes the visible-hidden term: ฮฃ๐ฟ ๐๐ฟแตหกแตโข(๐ฝ๐ฟ + ๐ฝห๐ฟ) with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ฏ๐ข๐ฌ๐ข๐๐ฅ๐_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++)
      log_vh += double(visible_config(0, flipped_visible_site(j_row, 0)) * (hidden_ket(0, flipped_visible_site(j_row, 0)) + hidden_bra(0, flipped_visible_site(j_row, 0))));

    return 2.0 * this -> eta().real() * log_vv - 2.0 * this -> omega().real() * log_vh;

  }

}


double BS_NNQS :: q_over_q_visible(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                   const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::exp(this -> logq_over_q_visible(visible_config, flipped_visible_site, hidden_ket, hidden_bra));

}


double BS_NNQS :: logq_over_q_ket(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site) const {

  /*#################################################################################*/
  //  Computes ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))
  //  evaluated in a new proposed configuration |๐ ๐โฟแตสท ๐หโฉ wrt
  //  the current configuration |๐ ๐แตหกแต ๐หโฉ (at fixed variational parameters ๐ฅ),
  //  where only the hidden variables ket have been changed.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed ket configuration is a configuration with a certain number of
  //  flipped spins wrt the old ket configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*#################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐ ๐โฟแตสท ๐หโฉ
  if(flipped_ket_site.n_elem==0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐ ๐โฟแตสท ๐หโฉ = |๐ ๐แตหกแต ๐หโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |๐โฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐ห) / ๐(๐,๐แตหกแต,๐ห))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |๐โฟแตสทโฉ
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: ฮฃ๐ฟ (๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // (๐ฝ_๐ญโฟแตสทโข๐ฝ_๐ขโฟแตสท - ๐ฝ_๐ญแตหกแตโข๐ฝ_๐ขแตหกแต)
      else
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // (๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)

    }

    //Computes the visible-hidden term: ฮฃ๐ฟ ๐๐ฟแตหกแตโข๐ฝ๐ฟแตหกแต with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ค๐๐ญ_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
      log_vh += double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));

    return this -> rho().real() * log_hh - 2.0 * this -> omega().real() * log_vh;

  }

}


double BS_NNQS :: q_over_q_ket(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site) const {

  return std::exp(this -> logq_over_q_ket(visible_config, hidden_ket, flipped_ket_site));

}


double BS_NNQS :: logq_over_q_bra(const Mat <int>& visible_config, const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  /*#################################################################################*/
  //  Computes ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))
  //  evaluated in a new proposed configuration |๐ ๐ ๐หโฟแตสทโฉ wrt
  //  the current configuration |๐ ๐ ๐หแตหกแตโฉ (at fixed variational parameters ๐ฅ),
  //  where only the hidden variables ket have been changed.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed ket configuration is a configuration with a certain number of
  //  flipped spins wrt the old ket configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*#################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐ ๐ ๐หโฟแตสทโฉ
  if(flipped_bra_site.n_elem==0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐ ๐ ๐หโฟแตสทโฉ = |๐ ๐ ๐หแตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration โจ๐หโฟแตสท|." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐,๐หโฟแตสท) / ๐(๐,๐,๐หแตหกแต))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // โจ๐หโฟแตสท|
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: ฮฃ๐ฟ (๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // (๐ฝห_๐ญโฟแตสทโข๐ฝห_๐ขโฟแตสท - ๐ฝห_๐ญแตหกแตโข๐ฝห_๐ขแตหกแต)
      else
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // (๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)

    }

    //Computes the visible-hidden term: ฮฃ๐ฟ ๐๐ฟแตหกแตโข๐ฝห๐ฟแตหกแต with ๐ฟ ฯต ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐๐ซ๐_๐ฌ๐ข๐ญ๐
    for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
      log_vh += double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    return this -> rho().real() * log_hh - 2.0 * this -> omega().real() * log_vh;

  }

}


double BS_NNQS :: q_over_q_bra(const Mat <int>& visible_config, const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_bra(visible_config, hidden_bra, flipped_bra_site));

}


double BS_NNQS :: logq_over_q_equal_site(const Mat <int>& visible_config, const Mat <int>& flipped_equal_site,
                                         const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*###################################################################################*/
  //  Computes ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต))
  //  evaluated in a new proposed configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ wrt
  //  the current configuration |๐แตหกแต ๐แตหกแต ๐หแตหกแตโฉ (at fixed variational parameters ๐ฅ).
  //  In this case we decide to move the spins located at the same (randomly
  //  choosen) lattice sites for all the three variables ๐, ๐, ๐ห.
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current spin configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*###################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ
  if(flipped_equal_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ = |๐แตหกแต ๐แตหกแต ๐หแตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |๐โฟแตสท ๐โฟแตสท ๐หโฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_equal_site);  // |๐โฟแตสทโฉ
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_equal_site);  // |๐โฟแตสทโฉ
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_equal_site);  // |๐หโฟแตสทโฉ
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms

    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1){

        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (๐_๐ญโฟแตสทโข๐_๐ขโฟแตสท - ๐_๐ญแตหกแตโข๐_๐ขแตหกแต)
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // (๐ฝ_๐ญโฟแตสทโข๐ฝ_๐ขโฟแตสท - ๐ฝ_๐ญแตหกแตโข๐ฝ_๐ขแตหกแต)
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // (๐ฝห_๐ญโฟแตสทโข๐ฝห_๐ขโฟแตสท - ๐ฝห_๐ญแตหกแตโข๐ฝห_๐ขแตหกแต)

      }
      else{

        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (๐๐ฟโฟแตสทโข๐๐ฟ+๐ฃโฟแตสท - ๐๐ฟแตหกแตโข๐๐ฟ+๐ฃแตหกแต)
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // (๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // (๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)

      }

    }

    return 2.0 * this -> eta().real() * log_vv + this -> rho().real() * log_hh;

  }

}


double BS_NNQS :: q_over_q_equal_site(const Mat <int>& visible_config, const Mat <int>& flipped_equal_site,
                                      const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return std::exp(this -> logq_over_q_equal_site(visible_config, flipped_equal_site, hidden_ket, hidden_bra));

}


double BS_NNQS :: logq_over_q_braket(const Mat <int>& visible_config,
                                     const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                     const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  /*################################################################################*/
  //  Computes ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต))
  //  evaluated in a new proposed configuration |๐ ๐โฟแตสท ๐หโฟแตสทโฉ wrt
  //  the current configuration |๐ ๐แตหกแต ๐หแตหกแตโฉ (at fixed variational parameters ๐ฅ).
  //  As mentioned before, this quantity will be used in the
  //  determination of the acceptance probability in the Metropolis Algorithm.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped auxiliary spins wrt the current spin configuration;
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  /*################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |๐ ๐โฟแตสท ๐หโฟแตสทโฉ
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0)
    return 0.0;  //๐๐๐(1) = 0, the case |๐ ๐โฟแตสท ๐หโฟแตสทโฉ = |๐ ๐แตหกแต ๐หแตหกแตโฉ
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |๐ ๐โฟแตสท ๐หโฟแตสทโฉ." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |๐โฟแตสทโฉ
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // โจ๐หโฟแตสท|
    double log_ket = 0.0;  //Storage variable for the hidden-hidden ket terms
    double log_bra = 0.0;  //Storage variable for the hidden-hidden bra terms
    double log_vk = 0.0;  //Storage variable for the visible-hidden ket terms
    double log_vb = 0.0;  //Storage variable for the visible-hidden bra terms

    //๐ฃ๐๐ ๐ธ๐ถ๐โฏ:  |๐โฟแตสทโฉ โ? |๐แตหกแตโฉ & โจ๐หโฟแตสท| = โจ๐หแตหกแต|
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the hidden-hidden terms only for the ket: ฮฃ๐ฟ (๐ฝ๐ฟโฟแตสทโข๐ฝ๐ฟ+๐ฃโฟแตสท - ๐ฝ๐ฟแตหกแตโข๐ฝ๐ฟ+๐ฃแตหกแต)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_ket += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));
        else
          log_ket += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: ฮฃ๐ฟ ๐๐ฟแตหกแตโข๐ฝ๐ฟแตหกแต
      for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));

    }

    //๐ค๐๐ ๐ธ๐ถ๐โฏ:  |๐โฟแตสทโฉ = |๐แตหกแตโฉ & โจ๐หโฟแตสท| โ? โจ๐หแตหกแต|
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the hidden-hidden terms only for the ket: ฮฃ๐ฟ (๐ฝห๐ฟโฟแตสทโข๐ฝห๐ฟ+๐ฃโฟแตสท - ๐ฝห๐ฟแตหกแตโข๐ฝห๐ฟ+๐ฃแตหกแต)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_bra += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));
        else
          log_bra += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: ฮฃ๐ฟ ๐๐ฟแตหกแตโข๐ฝ๐ฟแตหกแต
      for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    }

    //๐ฅ๐๐ ๐ธ๐ถ๐โฏ:  |๐โฟแตสทโฉ โ? |๐แตหกแตโฉ & โจ๐หโฟแตสท| โ? โจ๐หแตหกแต|
    else if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem != 0){

      //Computes the hidden-hidden terms
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1){

          log_ket += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));
          log_bra += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));

        }
        else{

          log_ket += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));
          log_bra += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));

        }

      }

      //Computes the visible-hidden terms
      for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));
      for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    }

    else{

      std::cerr << " ##OptionError: something went wrong in the determination of ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต))." << std::endl;
      std::cerr << "   Failed to compute ๐๐๐(๐(๐,๐โฟแตสท,๐หโฟแตสท) / ๐(๐,๐แตหกแต,๐หแตหกแต))." << std::endl;
      std::abort();

    }

    return this -> rho().real() * (log_ket + log_bra) - 2.0 * this -> omega().real() * (log_vk + log_vb);

  }

}


double BS_NNQS :: q_over_q_braket(const Mat <int>& visible_config,
                                  const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                  const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_braket(visible_config, hidden_ket, flipped_ket_site, hidden_bra, flipped_bra_site));

}


std::complex <double> BS_NNQS :: logPsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                       const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                       const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                       std::string option) const {

  //Function variables
  std::complex <double> logPoP;
  logPoP.imag(0.0);  //In this case the acceptance is a pure real number

  if(option == "visible")
    logPoP.real( this -> logq_over_q_visible(visible_config, flipped_visible_site, hidden_ket, hidden_bra));
  else if(option == "ket")
    logPoP.real(this -> logq_over_q_ket(visible_config, hidden_ket, flipped_ket_site));
  else if(option == "bra")
    logPoP.real(this -> logq_over_q_bra(visible_config, hidden_bra, flipped_bra_site));
  else if(option == "equal site")
    logPoP.real(this -> logq_over_q_equal_site(visible_config, flipped_visible_site, hidden_ket, hidden_bra));
  else if(option == "braket")
    logPoP.real(this -> logq_over_q_braket(visible_config, hidden_ket, flipped_ket_site, hidden_bra, flipped_bra_site));
  else{

    std::cerr << " ##OptionError: no available option as function argument." << std::endl;
    std::cerr << "   Failed to compute ๐๐๐(๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท) / ๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต))." << std::endl;
    std::abort();

  }

  return logPoP;

}


std::complex <double> BS_NNQS :: PsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                    const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                                    const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                                    std::string option) const {

  //Function variables
  std::complex <double> PoP;

  PoP = std::exp(this -> logPsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                               hidden_ket, flipped_ket_site,
                                               hidden_bra, flipped_bra_site,
                                               option));
  PoP.imag(0.0);
  return PoP;

}


double BS_NNQS :: PMetroNew_over_PMetroOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                           const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                           const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                           std::string option) const {

  /*######################################################################*/
  //  NฬฒOฬฒTฬฒEฬฒ: in the Shadow ansatz the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely ๐(๐โฟแตสท,๐โฟแตสท,๐หโฟแตสท,๐ฅ)/๐(๐แตหกแต,๐แตหกแต,๐หแตหกแต,๐ฅ)
  //
  /*######################################################################*/

  std::complex <double> p = this -> PsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                                       hidden_ket, flipped_ket_site,
                                                       hidden_bra, flipped_bra_site,
                                                       option);
  if(p.imag() != 0.0){

    std::cerr << " ##ValueError: the imaginary part of the Metropolis-Hastings acceptance probability must be zero!" << std::endl;
    std::cerr << "   Failed to compute the Metropolis-Hastings acceptance probability." << std::endl;
    std::abort();

  }

  return p.real();

}


void BS_NNQS :: LocalOperators(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameters
  //  ๐ on the sampled enlarged quantum configuration |๐ ๐ ๐หโฉ.
  //  In the case of the Baeriswyl-Shadow NNQS ansatz the local parameters ๐(๐,๐) are
  //
  //        โข ฮท โน--โบ ๐(๐,๐) = ๐(๐) = ฮฃ๐ฟ ๐ฃ๐ฟโข๐ฃ๐ฟ+๐ฃ
  //        โข ฯ โน--โบ ๐(๐,๐) = ๐(๐) = ฮฃ๐ฟ ๐ฝ๐ฟโข๐ฝ๐ฟ+๐ฃ       ๐(๐,๐ห) = ๐(๐ห) = ฮฃ๐ฟ ๐ฝห๐ฟโข๐ฝห๐ฟ+๐ฃ
  //        โข ฯ โน--โบ ๐(๐,๐) = ฮฃ๐ฟ ๐ฝ๐ฟโข๐ฃ๐ฟ                 ๐(๐,๐ห) = ฮฃ๐ฟ ๐ฃ๐ฟโข๐ฝห๐ฟ
  //
  //  It is important to note that in the Shadow wave function the local operators
  //  (which are a geometric properties of the wave function itself) related to
  //  the hidden-hidden interactions and the hidden-visible interaction, respectively
  //  depend also on the auxiliary variables, and not only on the actual quantum degrees
  //  of freedom of the system.
  //  These operators are necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  //  We remember that in the Shadow case the local operators are real.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |๐โฉ
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐) and ๐(๐,๐ห)." << std::endl;
    std::abort();

  }
  // |๐โฉ
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐) and ๐(๐,๐ห)." << std::endl;
    std::abort();

  }
  // โจ๐ห|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (๐ฝ = ๐)." << std::endl;
    std::cerr << "   Failed to compute the local operators ๐(๐,๐) and ๐(๐,๐ห)." << std::endl;
    std::abort();

  }

  //Function variables
  double O_vv = 0.0;  //Storage variable for the visible-visible terms
  double O_hh_ket = 0.0;  //Storage variable for the hidden-hidden terms
  double O_hh_bra = 0.0;  //Storage variable for the hidden-hidden terms
  double O_vh_ket = 0.0;  //Storage variable for the visible-hidden terms
  double O_vh_bra = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      O_vv += double(visible_config(0, j) * visible_config(0, 0));
      O_hh_ket += double(hidden_ket(0, j) * hidden_ket(0, 0));
      O_hh_bra += double(hidden_bra(0, j) * hidden_bra(0, 0));
      O_vh_ket += double(visible_config(0, j) * hidden_ket(0, j));
      O_vh_bra += double(visible_config(0, j) * hidden_bra(0, j));

    }
    else{

      O_vv += double(visible_config(0, j) * visible_config(0, j+1));
      O_hh_ket += double(hidden_ket(0, j) * hidden_ket(0, j+1));
      O_hh_bra += double(hidden_bra(0, j) * hidden_bra(0, j+1));
      O_vh_ket += double(visible_config(0, j) * hidden_ket(0, j));
      O_vh_bra += double(visible_config(0, j) * hidden_bra(0, j));

    }

  }

  _LocalOperators(0, 0) = O_vv;  // ๐_ฮท(๐)
  _LocalOperators(0, 1) = O_vv;  // ๐_ฮท(๐)
  _LocalOperators(1, 0) = O_hh_ket;  // ๐_ฯ(๐)
  _LocalOperators(1, 1) = O_hh_bra;  // ๐_ฯ(๐ห)
  _LocalOperators(2, 0) = O_vh_ket;  // ๐_ฯ(๐,๐)
  _LocalOperators(2, 1) = O_vh_bra;  // ๐_ฯ(๐,๐ห)

}


#endif
