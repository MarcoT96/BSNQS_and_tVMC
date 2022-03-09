#ifndef __ANSATZ__
#define __ANSATZ__


/********************************************************************************************************/
/***************************  Representation of the Many-Body Quantum State  ****************************/
/********************************************************************************************************/
/*

  We create several models in order to represent the quantum state of a many-body system defined
  in the discrete, on a 𝟏 and 𝟐 dimensional lattice 𝚲 ϵ ℤᵈ.
  The structure of the wave function ansatz is designed in a Variational Monte Carlo frameworks,
  that is, all the knowledge about the quantum state is encoded by a set of variational parameters
  that characterizes the generic interface of our classes (in the C++ language this is obtained
  through the use of an Abstract Class).
  These parameters should be optimized via a Variational Monte Carlo algorithm (𝐬𝐚𝐦𝐩𝐥𝐞𝐫.𝐜𝐩𝐩).
  Moreover, we are interested in building variational quantum states that are Artificial Intelligence
  (𝑨𝑰) inspired, so we always consider the presence of a certain number of 𝒗𝒊𝒔𝒊𝒃𝒍𝒆 variables,
  (i.e. the actual quantum degrees of freedom of the systems), supported by a certain number of
  𝒉𝒊𝒅𝒅𝒆𝒏 variables (auxiliary quantum degrees of freedom); the different types (visible or hidden)
  of variables are organized into distinct layers, according to a neural-inspired ansatz.
  Depending on the chosen architecture, there may be intra-layer interactions between variables
  of the same type and/or interactions between different variables that live in different layers.
  Even more, in some variational wave function the hidden variables will be traceable, and therefore
  we will have to worry only about the visible variables (as in the 𝐑𝐁𝐌); in the generic case however,
  the fictitious quantum variables will not be analytically integrable, and we should use more
  sophisticated sampling techniques (the 𝐒𝐇𝐀𝐃𝐎𝐖 case).
  However, we will consider complex variational parameters, and a generic form of the type

            Ψ(𝒗,𝓥) = ℯ𝓍𝓅(𝜙)•Σₕ ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ) = ℯ𝓍𝓅(𝜙)•Σₕ Φ(𝒗,𝒉,𝛂)

  with 𝕆ₖ(𝒗,𝒉) the so-called 𝐑𝐄𝐀𝐋 local operators, and 𝓥 = {𝜙,𝛂} ϵ ℂⁿ-ᵖᵃʳᵃᵐˢ.

  N̲O̲T̲E̲: we use the pseudo-random numbers generator device by [Percus & Kalos, 1989, NY University].
  N̲O̲T̲E̲: we use the C++ Armadillo library to manage Linear Algebra calculations.

*/
/********************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etc…
#include <cstdlib>  // <-- std::abort()
#include <cmath>  // <-- std::cosh(), std::log(), std::exp(), std::cos(), std::sin(), std::tanh()
#include <fstream>  // <-- std::ifstream, std::ofstream
#include <complex>  // <-- std::complex<>, .real(), .imag()
#include <armadillo>  // <-- arma::Mat, arma::Col
#include "random.h"  // <-- Random


using namespace arma;


  /*###########################################*/
 /*  𝐕𝐀𝐑𝐈𝐀𝐓𝐈𝐎𝐍𝐀𝐋 𝐖𝐀𝐕𝐄 𝐅𝐔𝐍𝐂𝐓𝐈𝐎𝐍 𝐈𝐍𝐓𝐄𝐑𝐅𝐀𝐂𝐄  */
/*###########################################*/
class WaveFunction {

  protected:

    //Geometric structure
    std::complex <double> _phi;  //The complex phase variational parameter 𝜙
    Col <std::complex <double>> _alpha;  //The variational parameters 𝛂 = {𝛼𝟣,𝛼𝟤,…,𝛼ⁿ-ᵖᵃʳᵃᵐˢ}
    Mat <std::complex <double>> _LocalOperators;  //The local operators 𝕆(𝒗,𝒉)

    //Architecture
    unsigned int _N;  //Number of visible variables 𝒗 = {𝑣𝟣,𝑣𝟤,…,𝑣𝖭}
    std::string _type;  //Type of ansatz
    bool _if_phi;  //Chooses 𝜙 ≠ 𝟢 (true) or 𝜙 = 𝟢 (false)

    //Random device
    Random _rnd;

  public:

    //Constructor and Destructor
    virtual ~WaveFunction() = default;  //Necessary for dynamic allocation


    /*****************************/
    /*  𝒩ℴ𝓃-𝓋𝒾𝓇𝓉𝓊𝒶ℒ 𝒻𝓊𝓃𝒸𝓉𝒾ℴ𝓃  */
    /*****************************/
    //Access functions
    unsigned int n_visible() const {return _N;}  //Returns the number of visible variables 𝒗 = {𝑣𝟣,𝑣𝟤,…,𝑣𝖭}
    std::string type_of_ansatz() const {return _type;}  //Returns the type of the chosen ansatz architecture
    bool if_phi_neq_zero() const {return _if_phi;}  //Returns whether or not to use the phase 𝜙 in the ansatz
    unsigned int n_alpha() const {return _alpha.n_elem;}  //Returns the number of variational parameters 𝛂 = {𝛼𝟣,𝛼𝟤,…,𝛼ⁿ-ᵖᵃʳᵃᵐˢ}
    std::complex <double> phi() const {return _phi;}  //Returns the complex phase variational parameter 𝜙
    Col <std::complex <double>> alpha() const {return _alpha;}  //Returns the set of 𝛂 = {𝛼𝟣,𝛼𝟤,…,𝛼ⁿ-ᵖᵃʳᵃᵐˢ}
    Mat <std::complex <double>> O() const {return _LocalOperators;}  //Returns the local operators 𝕆(𝒗,𝒉)
    std::complex <double> alpha_at(unsigned int) const;  //Returns a selected variational parameter 𝛼𝒿

    //Modifier functions
    void set_phi(std::complex <double> new_phi) {_phi = new_phi;}  //Changes the value of the complex phase variational parameter 𝜙
    void set_alpha_at(unsigned int, std::complex <double>);  //Changes the value of a selected variational parameter 𝛼𝒿
    void set_phi_real(double new_phi_real) {_phi.real(new_phi_real);}  //Changes the value of the real part of the phase variational parameter 𝜙ᴿ
    void set_phi_imag(double new_phi_imag) {_phi.imag(new_phi_imag);}  //Changes the value of the imaginary part of the phase variational parameter 𝜙ᴵ
    void set_alpha_real_at(unsigned int, double);  //Changes the value of the real part of a selected variational parameter 𝛼ᴿ𝒿
    void set_alpha_imag_at(unsigned int, double);  //Changes the value of the imaginary part of a selected variational parameter 𝛼ᴵ𝒿

    //Functional form of the ansatz
    Mat <int> generate_config(const Mat <int>&, const Mat <int>&) const;  //Creates a state of the type |𝒗⟩, |𝒉⟩ or |𝒉ˈ⟩


    /*************************/
    /*  𝒱𝒾𝓇𝓉𝓊𝒶ℒ 𝒻𝓊𝓃𝒸𝓉𝒾ℴ𝓃  */
    /************************/
    //Access function
    virtual unsigned int density() const = 0;  //Returns the density of the hidden variables ν = 𝖬/𝖭

    //Modifier functions
    virtual void Init_on_Config(const Mat <int>&) = 0;  //Initializes properly the ansatz on a given quantum configuration
    virtual void Update_on_Config(const Mat <int>&, const Mat <int>&) = 0;  //Updates properly the ansatz on a given new sampled quantum configuration

    //Functional form of the ansatz
    virtual double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    virtual double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    virtual double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    virtual std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const = 0;  //Computes 𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))
    virtual std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const = 0;  //Computes Φ(𝒗,𝒉,𝛂)
    virtual std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes 𝑙𝑜𝑔( Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂) )
    virtual std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
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
                                            const Mat <int>&, const Mat <int>&,  //probability || Ψ(𝒗ⁿᵉʷ,𝓥)/Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                            const Mat <int>&, const Mat <int>&,
                                            std::string option="") const = 0;
    virtual void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&) = 0;  //Computes the local operators 𝕆(𝒗,𝒉)

};


  /*####################################*/
 /*  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐰𝐢𝐭𝐡 𝐍𝐄𝐀𝐑𝐄𝐒𝐓 𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒  */
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
    JasNN(unsigned int, bool);
    JasNN(std::string, bool);
    ~JasNN();

    //Access functions
    unsigned int density() const {return 1;}
    std::complex <double> eta() const {return _alpha(0);}  //Returns the nearest neighbors coupling parameter

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}
    double cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Ψ(𝒗,𝜙,𝜂)) on a given visible configuration
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝜂) on a given visible configuration
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( Ψ(𝒗ⁿᵉʷ,𝜙,𝜂)/Ψ(𝒗ᵒˡᵈ,𝜙,𝜂) )
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝜂)/Ψ(𝒗ᵒˡᵈ,𝜙,𝜂)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Ψ(𝒗,𝜙,𝜂))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝜂)
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,    //Computes 𝑙𝑜𝑔( Ψ(𝒗ⁿᵉʷ,𝜙,𝜂)/Ψ(𝒗ᵒˡᵈ,𝜙,𝜂) )
                                              const Mat <int>&, const Mat <int>&,
                                              const Mat <int>&, const Mat <int>&,
                                              std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝜂)/Ψ(𝒗ᵒˡᵈ,𝜙,𝜂)
                                           const Mat <int>&, const Mat <int>&,
                                           const Mat <int>&, const Mat <int>&,
                                           std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings
                                  const Mat <int>&, const Mat <int>&,  //acceptance probability || Ψ(𝒗ⁿᵉʷ,𝜙,𝜂)/Ψ(𝒗ᵒˡᵈ,𝜙,𝜂) ||
                                  const Mat <int>&, const Mat <int>&,
                                  std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝕆(𝒗) = ∂𝑙𝑜𝑔(Ψ(𝒗,𝛂))/∂𝛂

};


  /*####################################*/
 /*  𝐑𝐄𝐒𝐓𝐑𝐈𝐂𝐓𝐄𝐃 𝐁𝐎𝐋𝐓𝐙𝐌𝐀𝐍𝐍 𝐌𝐀𝐂𝐇𝐈𝐍𝐄  */
/*####################################*/
class RBM : public WaveFunction {

  private:

    //RBM Neural Network architecture
    unsigned int _M;  //Number of hidden neurons 𝐡 = {𝒽𝟣,𝒽𝟤,…,𝒽𝖬}

    //Look-up table for the effective angles 𝛳(𝒗,𝛂)
    Col <std::complex<double>> _Theta;

    //Fast computation of the wave function
    const double _ln2;  //𝑙𝑜𝑔𝟤

  public:

    //Constructor and Destructor
    RBM(unsigned int, unsigned int);
    RBM(std::string);
    ~RBM();

    //Access functions
    unsigned int density() const {return _M/_N;}
    unsigned int n_hidden() const {return _M;}  //Returns the number of hidden neurons 𝐡 = {𝒽𝟣,𝒽𝟤,…,𝒽𝖬}
    std::complex <double> a_j(unsigned int) const;  //Returns the bias of the 𝒿-th visible neuron
    std::complex <double> b_k(unsigned int) const;  //Returns the bias of the 𝓀-th hidden neuron
    std::complex <double> W_jk(unsigned int, unsigned int) const;  //Returns the selected visible-hidden interaction strength
    std::complex <double> Theta_k(unsigned int) const;  //Returns the effective angles associated to the 𝓀-th hidden neuron
    Col <std::complex <double>> effective_angle() const {return _Theta;}  //Returns the set of 𝛳(𝒗,𝛂)
    void print_a() const;  //Prints on standard output the set of visible bias 𝐚
    void print_b() const;  //Prints on standard output the set of hidden bias 𝐛
    void print_W() const;  //Prints on standard output the visible-hidden interaction strength matrix 𝕎
    void print_Theta() const;  //Prints on standard output the set of effective angles 𝛳(𝒗,𝛂)

    //Modifier functions
    void Init_on_Config(const Mat <int>&);
    void Update_on_Config(const Mat <int>&, const Mat <int>&);

    //Wavefunction evaluation
    double lncosh(double) const;  //Computes 𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓍) of a real number 𝓍 ϵ ℝ
    std::complex <double> lncosh(std::complex <double>) const;  //Computes 𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓏) of a complex number 𝓏 ϵ ℂ
    void Init_Theta(const Mat <int>&);  //Initializes the effective angles 𝛳(𝒗,𝛂) on the given visible configuration |𝒗⟩
    void Update_Theta(const Mat <int>&, const Mat <int>&);  //Updates the effective angles 𝛳(𝒗,𝛂) on a new sampled visible configuration |𝒗ⁿᵉʷ⟩
    double I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}
    double cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Ψ(𝒗,𝓥)) on a given visible configuration
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝓥) on a given visible configuration
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥))
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Ψ(𝒗,𝓥))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝓥)
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,    //Computes 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥))
                                                const Mat <int>&, const Mat <int>&,
                                                const Mat <int>&, const Mat <int>&,
                                                std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥)
                                             const Mat <int>&, const Mat <int>&,
                                             const Mat <int>&, const Mat <int>&,
                                             std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings
                                    const Mat <int>&, const Mat <int>&,  //acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝕆(𝒗) = ∂𝑙𝑜𝑔(Ψ(𝒗,𝛂))/∂𝛂

};


  /*##############################################*/
 /*  𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐔𝐑 (quasi)-𝐮𝐑𝐁𝐌 in 𝗱 = 𝟏  */
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
    quasi_uRBM(unsigned int, bool);
    quasi_uRBM(std::string, bool);
    ~quasi_uRBM();

    //Access functions
    unsigned int density() const {return 1;}
    std::complex <double> eta_j(unsigned int) const;  //Returns the selected visible-visible interaction strength
    std::complex <double> rho_j(unsigned int) const;  //Returns the selected hidden-hidden interaction strength
    std::complex <double> omega_j(unsigned int) const;  //Returns the selected visible-hidden interaction strength
    void print_eta() const;  //Prints on standard output the set of visible-visible interaction strength 𝛈
    void print_rho() const;  //Prints on standard output the set of hidden-hidden interaction strength 𝛒
    void print_omega() const;  //Prints on standard output the set of visible-hidden interaction strength 𝒘

    //Modifier functions
    void Init_on_Config(const Mat <int>& visible_config) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗,𝒉,𝛂)
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂) )
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ,𝓥))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥) )
    double q_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥) )
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥) )
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔( 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) ) on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes 𝑙𝑜𝑔( 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) )
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔( 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) )
                                                const Mat <int>&, const Mat <int>&,
                                                const Mat <int>&, const Mat <int>&, std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                                             const Mat <int>&, const Mat <int>&,
                                             const Mat <int>&, const Mat <int>&, std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥)/Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&, std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝕆(𝒗,𝒉) = ∂𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))/∂𝛂

};


  /*#####################################*/
 /*  𝐁𝐀𝐄𝐑𝐈𝐒𝐖𝐘𝐋-𝐒𝐇𝐀𝐃𝐎𝐖 𝐍𝐍𝐐𝐒 in 𝗱 = 𝟏  */
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
    BS_NNQS(unsigned int, bool);
    BS_NNQS(std::string, bool);
    ~BS_NNQS();

    //Access functions
    unsigned int density() const {return 1;}
    std::complex <double> eta() const {return _alpha(0);}  //Returns the visible-visible interaction strength η
    std::complex <double> rho() const {return _alpha(1);}  //Returns the hidden-hidden interaction strength ρ
    std::complex <double> omega() const {return _alpha(2);}  //Returns the visible-hidden interaction strength ω

    //Modifier functions
    void Init_on_Config(const Mat <int>& visible_config) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    std::complex <double> logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))
    std::complex <double> Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗,𝒉,𝛂)
    std::complex <double> logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂) )
    std::complex <double> PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
    std::complex <double> logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ,𝓥))
    std::complex <double> PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥) )
    double q_over_q_visible(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥) )
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔( 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥) )
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔( 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) ) on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes 𝑙𝑜𝑔( 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) )
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    std::complex <double> logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔( 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) )
                                                const Mat <int>&, const Mat <int>&,
                                                const Mat <int>&, const Mat <int>&,
                                                std::string option="") const;
    std::complex <double> PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                                             const Mat <int>&, const Mat <int>&,
                                             const Mat <int>&, const Mat <int>&,
                                             std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥)/Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝕆(𝒗,𝒉) = ∂𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂))/∂𝛂

};




/*******************************************************************************************************************************/
/******************************************  𝐕𝐀𝐑𝐈𝐀𝐓𝐈𝐎𝐍𝐀𝐋 𝐖𝐀𝐕𝐄 𝐅𝐔𝐍𝐂𝐓𝐈𝐎𝐍 𝐈𝐍𝐓𝐄𝐑𝐅𝐀𝐂𝐄  ******************************************/
/*******************************************************************************************************************************/
std::complex <double> WaveFunction :: alpha_at(unsigned int j) const {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to access the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter 𝛼𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else
    return _alpha(j);

}


void WaveFunction :: set_alpha_at(unsigned int j, std::complex <double> new_param) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter 𝛼𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    std::abort();

  }
  //Check passed
  else
    _alpha(j) = new_param;

}


void WaveFunction :: set_alpha_real_at(unsigned int j, double new_param_real) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the real part of the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter 𝛼ᴿ𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    std::abort();

  }
  //Check passed
  else
    _alpha(j).real(new_param_real);

}


void WaveFunction :: set_alpha_imag_at(unsigned int j, double new_param_imag) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the imaginary part of the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter 𝛼ᴵ𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
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
/**********************************************  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐰𝐢𝐭𝐡 𝐍𝐄𝐀𝐑𝐄𝐒𝐓 𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒  *********************************************/
/*******************************************************************************************************************************/
JasNN :: JasNN(unsigned int n_visible, bool phi_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the nearest-neighbors entangling Jatrow variational parameters
  //  𝓥 = {𝜙, 𝜂} = {𝜙, 𝛂} to some small random numbers.
  //
  //  In this case we have only 𝟭 parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝟏 nearest-neighbors visible-visible interaction strength 𝜂.
  //
  //  Note that in this case the number of variational parameters remains equal to 𝟏 for any system size 𝖭.
  /*########################################################################################################*/

  //Information
  std::cout << "#Create a nearest-neighbors Jastrow wave function with randomly initialized variational parameters 𝓥 = {𝜙, 𝜂}." << std::endl;

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
  _LocalOperators.zeros(1, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  if(_if_phi){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else
    _phi = 0.0;
  _alpha(0).real(_rnd.Gauss(0.0, 0.001));
  _alpha(0).imag(_rnd.Gauss(0.0, 0.001));

  std::cout << " Nearest-neighbors Jastrow ansatz correctly initialized with random interactions." << std::endl;
  std::cout << " Number of visible variables = " << _N << "." << std::endl;
  std::cout << " Number of hidden variables = " << _N << "." << std::endl;
  std::cout << " Density of the nearest-neighbors Jastrow ansatz = " << this -> density() << "." << std::endl << std::endl;

}


JasNN :: JasNN(std::string file_wf, bool phi_option) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the nearest-neighbors Jastrow variational parameters
  //  𝓥 = {𝜙, 𝜂} = {𝜙, 𝛂} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  std::cout << "#Create a nearest-neighbors Jastrow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the nearest-neighbors Jastrow variational parameters 𝓥 = {𝜙, 𝜂} from file." << std::endl;
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
  _LocalOperators.set_size(1, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  input_wf >> _alpha(0);

  if(input_wf.good()){

    std::cout << " Nearest-neighbors Jastrow Ansatz correctly loaded from file " << file_wf << "." << std::endl;
    std::cout << " Number of visible neurons = " << _N << "." << std::endl;
    std::cout << " Number of hidden neurons = " << _N << "." << std::endl;
    std::cout << " Density of the nearest-neighbors Jastrow ansatz = " << this -> density() << "." << std::endl << std::endl;

  }
  input_wf.close();

}


JasNN :: ~JasNN() {

  _rnd.SaveSeed();

}


std::complex <double> JasNN :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*####################################################*/
  //  Computes 𝑙𝑜𝑔(Ψ(𝒗,𝜙,𝜂)) with
  //
  //        Ψ(𝒗,𝜙,𝜂) = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(𝜂 Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣).
  //
  //  Obviously, this ansatz is not of the Shadow type,
  //  and no auxiliary variables are introduced here.
  /*####################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗,𝜙,𝜂))." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_psi = 0.0;  //Storage variable for the nearest-neighbors interaction

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1)
      log_psi += double(visible_config(0, j) * visible_config(0, 0));  // 𝓋_𝖭 • 𝓋_𝟢
    else
      log_psi += double(visible_config(0, j) * visible_config(0, j+1));  // 𝓋𝒿 • 𝓋𝒿+𝟣

  }

  return this -> phi() + this -> eta() * log_psi;

}


std::complex <double> JasNN :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


std::complex <double> JasNN :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                                     const Mat <int>& hidden_config) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)) at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the old visible configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the 𝐔𝐩𝐝𝐚𝐭𝐞_𝐓𝐡𝐞𝐭𝐚 function
  //  defined below in the 𝐑𝐁𝐌 class.
  //  Note that the ratio between the two evaluated wave function, which is the
  //  quantity related to the acceptance kernel of the Metropolis algorithm,
  //  is recovered by taking the exponential of the output of this function.
  //
  //  N̲O̲T̲E̲: once again we emphasize that in the specific case of the Jastrow
  //        ansatz the quantities calculated with the functions inherent to
  //        Φ(𝒗,𝒉,𝛂) correspond to those calculated in the functions related
  //        to the Metropolis algorithm, since we have never introduced any
  //        auxiliary variable.
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the Jastrow ansatz,
  //        which does not depend upon any hidden variables.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new visible configuration does not match with the number of visible variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the visible-visible terms

    //Computes the visible-visible terms: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
      else
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)

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
  //  𝜂 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the Jastrow ansatz the local operators 𝕆(𝒗,𝒉) is
  //
  //        • 𝜂 ←→ 𝕆(𝒗,𝒉) = 𝕆(𝒗) = Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣.
  //
  //  This operator is necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉)." << std::endl;
    std::abort();

  }

  //Function variables
  double O_vv = 0.0;  //Storage variable for the visible-visible terms

  //Computes the local operator assiociated to the only parameter 𝜂
  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1)
      O_vv += double(visible_config(0, j) * visible_config(0, 0));  // 𝓋_𝖭 • 𝓋_𝟢
    else
      O_vv += double(visible_config(0, j) * visible_config(0, j+1));  // 𝓋𝒿 • 𝓋𝒿+𝟣

  }

  _LocalOperators(0, 0) = O_vv;  // 𝕆_η(𝒗)
  //_LocalOperators(0, 1) = O_vv;  // 𝕆_η(𝒗)

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  𝐑𝐄𝐒𝐓𝐑𝐈𝐂𝐓𝐄𝐃 𝐁𝐎𝐋𝐓𝐙𝐌𝐀𝐍𝐍 𝐌𝐀𝐂𝐇𝐈𝐍𝐄  *********************************************/
/*******************************************************************************************************************************/
RBM :: RBM(unsigned int n_visible, unsigned int density)
     : _M(density*n_visible), _ln2(std::log(2.0)) {

  /*################################################################################*/
  //  Random-based constructor.
  //  Initializes the RBM variational parameters 𝛂 = {𝐚,𝐛,𝕎} to
  //  some small random numbers [G.Hinton, 2010].
  //  We have
  //
  //        𝖭 visible neuron bias 𝐚 = {𝑎𝟢,𝑎𝟣,…,𝑎𝖭};
  //        𝖬 hidden neuron bias 𝐛 = {𝑏𝟢,𝑏𝟣,…,𝑏𝖭};
  //        𝖭 • 𝖬 visible-hidden neuron interaction strength weights 𝕎 = [𝕎]𝒿𝓀
  //
  //  organized sequentially in the parameter vector data-member.
  //  Note that being 𝕎 a matrix, we 'unrolled' it row by row saving
  //  it in _alpha as a vector of 𝖭 • 𝖬 elements.
  //  We remember that the 𝒿-th row of 𝕎 represents the list of the interactions
  //  strength between the 𝒿-th visible variable and each of the 𝖬 hidden neurons.
  /*################################################################################*/

  //Information
  std::cout << "#Create a RBM wave function with randomly initialized variational parameters 𝛂 = {𝐚,𝐛,𝕎 }." << std::endl;

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
    std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _if_phi = false;
  _phi = 0.0;
  _type = "Neural Network";
  _N = n_visible;
  _alpha.set_size(_N + _M + _N * _M);
  _LocalOperators.zeros(_N + _M + _N * _M, 2);
  _Theta.set_size(_M);

  //Visible bias
  for(unsigned int j = 0; j < _N; j++)
    _alpha(j) = 0.0;  // 𝛼ⱼ ≡ 𝑎ⱼ

  //Hidden bias
  for(unsigned int k = _N; k < _N + _M; k++)
    _alpha(k) = 0.0;  // 𝛼ₖ ≡ 𝑏ₖ

  //Visible-hidden interaction weights
  for(unsigned int jk = _N + _M; jk < _alpha.n_elem; jk++){

    _alpha(jk).real(_rnd.Gauss(0.0, 0.1));  // 𝛼ⱼᴿ ≡ [𝕎]ᴿ𝒿𝓀
    _alpha(jk).imag(_rnd.Gauss(0.0, 0.1));  // 𝛼ⱼᴵ ≡ [𝕎]ᴵ𝒿𝓀

  }

  std::cout << " RBM ansatz correctly initialized with random weights." << std::endl;
  std::cout << " Number of visible neurons = " << _N << "." << std::endl;
  std::cout << " Number of hidden neurons = " << _M << "." << std::endl;
  std::cout << " Density of the RBM ansatz = " << this -> density() << "." << std::endl << std::endl;

}


RBM :: RBM(std::string file_wf)
     : _ln2(std::log(2.0)) {

  /*##############################################################*/
  //  File-based constructor.
  //  Initializes RBM variational parameters 𝛂 = {𝐚,𝐛,𝕎} from a
  //  given external file in '.wf' format; this can be useful
  //  in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent
  //  variational Monte Carlo with a previously optimized
  //  ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*##############################################################*/

  //Information
  std::cout << "#Create a RBM wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the RBM variational parameters 𝛂 = { 𝐚,𝐛,𝕎 } from file." << std::endl;
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
    std::cerr << "   Failed to initialize the RBM variational parameters 𝛂 = { 𝐚,𝐛,𝕎 } from file." << std::endl;
    std::abort();

  }
  _if_phi = false;
  _phi = 0.0;
  _type = "Neural Network";
  _alpha.set_size(_N + _M + _N * _M);
  _LocalOperators.zeros(_N + _M + _N * _M, 1);  // 𝕆ₖ(𝒗,𝒉) ≡ 𝕆ₖ(𝒗,𝒉ˈ) ≡ 𝕆ₖ(𝒗)
  _Theta.set_size(_M);
  for(unsigned int p = 0; p <_alpha.n_elem; p++)
    input_wf >> _alpha(p);

  if(input_wf.good()){

    std::cout << " RBM ansatz correctly loaded from file " << file_wf << "." << std::endl;
    std::cout << " Number of visible neurons = " << _N << "." << std::endl;
    std::cout << " Number of hidden neurons = " << _M << "." << std::endl;
    std::cout << " Density of the RBM ansatz = " << this -> density() << "." << std::endl << std::endl;

  }
  input_wf.close();

}


RBM :: ~RBM() {

  _rnd.SaveSeed();

}


std::complex <double> RBM :: a_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the visible bias vector 𝐚 = {𝑎𝟢,𝑎𝟣,…,𝑎𝖭}." << std::endl;
    std::cerr << "   Element 𝑎ⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(j);

}


std::complex <double> RBM :: b_k(unsigned int k) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(k >= _M || k < 0){

    std::cerr << " ##IndexError: failed to access the hidden bias vector 𝐛 = {𝑏𝟢,𝑏𝟣,…,𝑏𝖭}." << std::endl;
    std::cerr << "   Element 𝑏ₖ with k = " << k << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(_N + k);

}


std::complex <double> RBM :: W_jk(unsigned int j, unsigned int k) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || k >= _M || j < 0 || k < 0){

    std::cerr << " ##IndexError: failed to access the visible-hidden interaction strength matrix 𝕎." << std::endl;
    std::cerr << "   Element 𝕎ⱼₖ with j = " << j << " and k = " << k << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(_N + _M + j * _M + k);

}


std::complex <double> RBM :: Theta_k(unsigned int k) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(k >= _M || k < 0){

    std::cerr << " ##IndexError: failed to access the effective angles 𝛳(𝒗,𝛂)." << std::endl;
    std::cerr << "   Element 𝛳ₖ with k = " << k << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _Theta(k);

}


void RBM :: print_a() const {

  std::cout << "\n======================================" << std::endl;
  std::cout << "RBM visible bias vector 𝐚 = {𝑎𝟢,𝑎𝟣,…,𝑎𝖭}" << std::endl;
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
  std::cout << "RBM hidden bias vector 𝐛 = {𝑏𝟢,𝑏𝟣,…,𝑏𝖭}" << std::endl;
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
  std::cout << "RBM visible-hidden interaction strength matrix 𝕎 = [𝕎]𝒿𝓀" << std::endl;
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
  std::cout << "RBM effective angles 𝛳(𝒗,𝛂)" << std::endl;
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
  //  of a real number 𝓍 ϵ ℝ; we use the asymptotic expantion
  //  when the argument exceeds a certain threshold for
  //  computational efficiency reasons (see the appropriate
  //  section in the Jupyter Notebook 𝐍𝐨𝐭𝐞𝐬.𝐢𝐩𝐲𝐧𝐛).
  //  The asymptotic expantion is:
  //
  //        𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓍) ~ 𝓍 - 𝑙𝑜𝑔𝟤
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
  //  number 𝓏 ϵ ℂ
  //
  //        𝓏 = ℜe{𝓏} + i•ℑm{𝓏} = 𝓊 + i𝓋
  //
  //  We can manipulate the complex logarithm in
  //  the following way:
  //
  //        𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓏) = 𝑙𝑜𝑔[1/2 • (eᶻ + 1/eᶻ)]
  //                  = 𝑙𝑜𝑔[1/2 • (eᵘ•eⁱᵛ + 1/eᵘ•eⁱᵛ)]
  //                  = 𝑙𝑜𝑔{1/2 • [eᵘ•(cos(𝓋) + i•sin(𝓋)) + 1/eᵘ(cos(𝓋) - i•sin(𝓋))]}
  //                  = 𝑙𝑜𝑔{1/2 • [cos(𝓋)•(eᵘ + 1/eᵘ) + i•sin(𝓋)•(eᵘ - 1/eᵘ)]}
  //                  = 𝑙𝑜𝑔{cosh(𝓊)•cos(𝓋) + i•sinh(𝓊)•sin(𝓋)}
  //                  = 𝑙𝑜𝑔{cosh(𝓊)•[cos(𝓋) + i•tanh(𝓊)•sin(𝓋)]}
  //                  = lncosh(ℜe{𝓏}) + 𝑙𝑜𝑔{cos(ℑm{𝓏}) + i•tanh(ℜe{𝓏})•sin(ℑm{𝓏})}
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
  //  𝛳(𝒗,𝛂) Initialization  -->  𝛳ₖ(𝒗,𝛂) = 𝑏ₖ + Σₘ [𝕎]ₖₘ•σₘᶻ
  //  Initializes the effective angles that appear thanks to the fact that
  //  in the particular functional form chosen for the this wave function
  //  the hidden degrees of freedom are traced out exactly.
  //
  //  N̲O̲T̲E̲: This analytical integration changes the generic form
  //        introduced above for the ansatz and consequently will also
  //        change the optimization algorithm (𝐬𝐚𝐦𝐩𝐥𝐞𝐫.𝐜𝐩𝐩).
  //        In fact here the local operators associated with variational
  //        parameters become complex and no longer real, just as the angles
  //        𝛳(𝒗,𝛂).
  //
  //  The angles depend on the parameters {𝐛,𝕎} and
  //  on the visible variables (i.e. the quantum spin) that define the
  //  current quantum configuration of the associated quantum system.
  //  The effective angles serve both in the estimate of the Monte
  //  Carlo observables (via the Metropolis Algorithm) and in the
  //  stochastic optimization of the variational parameters
  //  (via imaginary-time and/or real-time VMC).
  //
  //  The (sampled) configuration 𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐜𝐨𝐧𝐟𝐢𝐠 on which the effective
  //  angles are calculated can be either the configuration of a quantum
  //  spin system in 𝟏 dimension (𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐜𝐨𝐧𝐟𝐢𝐠.n_rows = 𝟏), or
  //  in 𝟐 dimensions (𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐜𝐨𝐧𝐟𝐢𝐠.n_rows ≠ 𝟏), for example
  //
  //                                𝒩
  //                     < -------------------- >        ^
  //                    | σᶻ σᶻ σᶻ     …       σᶻ  \     |
  //                    | σᶻ σᶻ σᶻ     …       σᶻ   \    |
  //        |𝒗𝟣 … 𝒗𝖭⟩ = | :  :  :      …       σᶻ    \   |  ℳ
  //                    | :  :  :      …       σᶻ    /   |
  //                    | :  :  :      …       σᶻ   /    |
  //                    | σᶻ σᶻ σᶻ     …       σᶻ  /     |
  //                                                     v
  //
  //  for a total of 𝖭 = 𝒩•ℳ quantum degrees of freedom.
  /*#######################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to initialize the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
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
  //  table _𝐓𝐡𝐞𝐭𝐚 when a new quantum configuration is sampled during
  //  the Monte Carlo Markov Chain (MCMC).
  //  We efficiently represent the new configuration in which
  //  the quantum system is through the matrix 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐬𝐢𝐭𝐞,
  //  which contains the list of the indices (integer numbers)
  //  related to the lattice sites in which the spins of the old configuration
  //  |𝒗ᵒˡᵈ⟩ have been flipped compared to the new configuration |𝒗ⁿᵉʷ⟩
  //  proposed with the Metropolis algorithm (instead of saving
  //  the entire matrix related to the new quantum configuration).
  //  So in 𝐝 = 𝟏 this matrix will be reduced to a column vector of the type
  //
  //        ⌈  𝟢  ⌉ : we have flipped the 1st spin of the 𝟏d chain
  //        |  𝟫  | : we have flipped the 10th spin of the 𝟏d chain
  //        |  •  | : “                                           ”
  //        |  •  | : “                                           ”
  //        |  •  | : “                                           ”
  //        ⌊  •  ⌋ : “                                           ”
  //
  //  while in 𝐝 = 𝟐 it will be a matrix in which each
  //  row represents the pair of indices which identifies the two dimensional lattice
  //  flipped spin site (𝓍 𝓎) ϵ 𝚲, e.g. the spin in first position is represented
  //  with the pair (𝟢 𝟢) in this matrix.
  //  However, in any case, the effective angles are updated as follows:
  //
  //        𝛳ₖ(𝒗ⁿᵉʷ,𝛂) = 𝛳ₖ(𝒗ᵒˡᵈ,𝛂) - 2 • Σ𝒿 [𝕎]𝒿ₖ•σ𝒿ᶻ
  //
  //  where 𝒿 is an index that runs only on the lattice sites where
  //  a spin is flipped, as described above.
  /*##################################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to update the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0)
    return;
  else{

    //Check on the lattice dimensionality
    if(visible_config.n_rows == 1 && flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to update the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
      std::abort();

    }
    if(visible_config.n_rows != 1 && flipped_visible_site.n_cols != 2){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to update the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
      std::abort();

    }

    //Function variables
    std::complex <double> delta_theta;  //Storage variable

    //Updates 𝛳(𝒗,𝛂)
    for(unsigned int k = 0; k < _M; k++){

      delta_theta = 0.0;
      for(unsigned int m_row = 0; m_row < flipped_visible_site.n_rows; m_row++){

        if(flipped_visible_site.n_cols == 1)  //𝐝 = 𝟏
          delta_theta += double(visible_config(0, flipped_visible_site(m_row, 0))) * _alpha(_N + _M + flipped_visible_site(m_row, 0) * _M + k);
        else if(flipped_visible_site.n_cols == 2)  //𝐝 = 𝟐
          delta_theta += double(visible_config(flipped_visible_site(m_row, 0), flipped_visible_site(m_row, 1))) *
                                _alpha(_N + _M + (flipped_visible_site(m_row, 0) * visible_config.n_cols + flipped_visible_site(m_row, 1)) * _M + k);
        else{

          std::cerr << " ##SizeError: bad construction of the new quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
          std::cerr << "   Failed to updatet the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
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
  //  of freedom for this ansatz, we can here interpret Φ(𝒗,𝒉,𝛂) as
  //  the total wave function, which is defined as (we set 𝜙 = 𝟢):
  //
  //        Ψ(𝒗,𝛂) = Σₕ ℯ𝓍𝓅(Σⱼ𝑎ⱼσⱼᶻ + Σₖ𝑏ₖ𝒽ₖ + Σⱼ[𝕎]ⱼₖ𝒽ⱼσₖᶻ)
  //               = ℯ𝓍𝓅(Σⱼ𝑎ⱼσⱼᶻ) • 𝚷ₖ 2𝑐𝑜𝑠ℎ(𝛳ₖ)
  //
  //  where the effective angles are defined above.
  //
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the RBM ansatz,
  //        which does not depend explicitly on the hidden variables.
  /*###################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗,𝛂))." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_vv(0.0, 0.0);  //Storage variable for the visible terms
  std::complex <double> log_theta(0.0, 0.0);  //Storage variable for the theta angle terms

  //Computes the visible neurons terms: Σⱼ 𝑎ⱼσⱼᶻ
  for(unsigned int j_row = 0; j_row < visible_config.n_rows; j_row++){

    for(unsigned int j_col = 0; j_col < visible_config.n_cols; j_col++)
      log_vv += _alpha(j_row * visible_config.n_cols + j_col) * double(visible_config(j_row, j_col));

  }

  //Computes the theta angles contribution: Σₖ 𝑙𝑜𝑔(𝑐𝑜𝑠ℎ(𝛳ₖ))
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
  //  Computes 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)) at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the
  //  𝐔𝐩𝐝𝐚𝐭𝐞_𝐓𝐡𝐞𝐭𝐚 function defined above.
  //  Note that the ratio between the two evaluated wave function,
  //  which is the quantity related to the acceptance kernel of the
  //  Metropolis algorithm is recovered by taking the exponential
  //  function of the output of this function.
  //
  //  N̲O̲T̲E̲: once again we emphasize that in the specific case of the RBM
  //        the quantities calculated with the functions inherent to Φ(𝒗,𝒉,𝛂)
  //        correspond to those calculated in the functions related to the
  //        Metropolis algorithm, since we have traced away the fictitious
  //        degrees of freedom.
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the RBM ansatz,
  //        which does not depend explicitly on the hidden variables.
  /*#############################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _N){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔( Ψ(𝒗ⁿᵉʷ,𝛂)/Ψ(𝒗ᵒˡᵈ,𝛂) )." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    //𝐝 = 𝟏
    if(visible_config.n_rows == 1 && flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ))." << std::endl;
      std::abort();

    }
    //𝐝 = 𝟐
    if(visible_config.n_rows != 1 && flipped_visible_site.n_cols != 2){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ))." << std::endl;
      std::abort();

    }

    //Function variables
    std::complex <double> log_vv(0.0, 0.0);  //Storage variable for the visible terms
    std::complex <double> log_theta(0.0, 0.0);  //Storage variable for the old theta angles
    std::complex <double> log_theta_prime(0.0, 0.0);  //Storage variable for the new theta angles

    //Change due to the visible layer
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++){

      if(flipped_visible_site.n_cols == 1)  //𝐝 = 𝟏
        log_vv -= _alpha(flipped_visible_site(j_row, 0)) * double(visible_config(0, flipped_visible_site(j_row, 0)));
      else if(flipped_visible_site.n_cols == 2){  //𝐝 = 𝟐

        log_vv -= _alpha(flipped_visible_site(j_row, 0) * visible_config.n_cols + flipped_visible_site(j_row, 1)) *
                  double(visible_config(flipped_visible_site(j_row, 0), flipped_visible_site(j_row, 1)));
      }
      else{

        std::cerr << " ##SizeError: bad construction of the new quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
        std::cerr << "   Failed to compute 𝑙𝑜𝑔(Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ))." << std::endl;
        std::abort();

      }

    }
    log_vv *= 2.0;

    //Change due to the visible-hidden interactions
    for(unsigned int k = 0; k < _M; k++){

      log_theta = _Theta(k);  //speed-up the calculation with the Look-up table
      log_theta_prime = log_theta;
      for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++){

        if(flipped_visible_site.n_cols == 1)  //𝐝 = 𝟏
          log_theta_prime -= 2.0 * double(visible_config(0, flipped_visible_site(j_row, 0))) * _alpha(_N + _M + flipped_visible_site(j_row, 0)*_M + k);
        else if(flipped_visible_site.n_cols == 2){  //𝐝 = 𝟐

          log_theta_prime -= 2.0 * double(visible_config(flipped_visible_site(j_row, 0), flipped_visible_site(j_row, 1))) *
                             _alpha(_N + _M + (flipped_visible_site(j_row, 0) * visible_config.n_cols + flipped_visible_site(j_row, 1)) * _M + k);

        }
        else{

          std::cerr << " ##SizeError: bad construction of the new quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
          std::cerr << "   Failed to compute 𝑙𝑜𝑔( Ψ(𝒗ⁿᵉʷ,𝛂)/Ψ(𝒗ᵒˡᵈ,𝛂) )." << std::endl;
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
  //  variational parameters 𝛂 = {𝐚,𝐛,𝕎} on the sampled
  //  quantum configuration |𝒗⟩.
  //  In the case of the RBM ansatz the local parameters are 𝕆(𝒗,𝒉) = 𝕆(𝒗)
  //
  //        • 𝑎𝒿 ←→ 𝕆(𝒗) = σ𝒿ᶻ
  //        • 𝑏𝓀 ←→ 𝕆(𝒗) = 𝑡𝑎𝑛ℎ(𝛳𝓀(𝒗))
  //        • [𝕎]𝒿𝓀 ←→ 𝕆(𝒗) = σ𝒿ᶻ•𝑡𝑎𝑛ℎ(𝛳𝓀(𝒗))
  //
  //  and are 𝐜𝐨𝐦𝐩𝐥𝐞𝐱 number once evaluated!
  //  These operators are necessary to compute the Quantum
  //  Geometric Tensor and the Gradient during the
  //  stochastic optimization procedure.
  //
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐤𝐞𝐭 and 𝒉𝒊𝒅𝒅𝒆𝒏_𝐛𝐫𝐚 arguments are useless for the
  //        RBM ansatz, which does not depend explicitly on the
  //        hidden variables.
  /*#######################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem !=_N ){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of visible neurons." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗)." << std::endl;
    std::abort();

  }

  //Local operators for the visible bias 𝐚
  for(unsigned int j_row = 0; j_row < visible_config.n_rows; j_row++){

    for(unsigned int j_col = 0; j_col < visible_config.n_cols; j_col++)
      _LocalOperators(j_row * visible_config.n_cols + j_col, 0) = double(visible_config(j_row, j_col));

  }

  //Local operators for the hidden bias 𝐛
  for(unsigned int k = 0; k < _M; k++)
    _LocalOperators(_N + k, 0) = std::tanh(_Theta(k));

  //Local operators for the visible-hidden interaction strength 𝕎
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
/*****************************************  𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐔𝐑 (quasi)-𝐮𝐑𝐁𝐌 in 𝗱 = 𝟏  ****************************************/
/*******************************************************************************************************************************/
quasi_uRBM :: quasi_uRBM(unsigned int n_visible, bool phi_option) {

  /*######################################################################################################*/
  //  Random-based constructor.
  //  Initializes the (quasi)-uRBM variational parameters 𝓥 = {𝜙,𝛈,𝛒,𝒘} = {𝜙,𝛂} to
  //  some small random numbers.
  //
  //  Imposing periodic boundary conditions we have
  //
  //        𝟏 complex phase 𝜙
  //        𝖭 nearest-neighbors visible-visible interaction strength weights 𝛈 = {ηⱼ};
  //        𝖭 nearest-neighbors hidden-hidden interaction strength weights 𝛒 = {ρⱼ};
  //        𝖭 local visible-hidden interaction strength weights 𝒘 = {𝑤ⱼ};
  //
  //  We remember only in the special case of 𝟏 dimension the size of the sets of intra- and extra-layer
  //  connections is the same, since in 𝟏 dimension the number of nearest-neighbors site is 𝟐.
  /*######################################################################################################*/

  //Information
  std::cout << "#Create a 1D n.n. (quasi)-uRBM wave function with randomly initialized variational parameters 𝓥 = { 𝜙,𝛈,𝛒,𝒘 }." << std::endl;

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
  _LocalOperators.set_size(3 * _N, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
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

  std::cout << " (quasi)-uRBM ansatz correctly initialized with random interactions." << std::endl;
  std::cout << " Number of visible variables = " << _N << "." << std::endl;
  std::cout << " Number of hidden variables = " << _N << "." << std::endl;
  std::cout << " Density of the (quasi)-uRBM ansatz = " << this -> density() << "." << std::endl << std::endl;

}


quasi_uRBM :: quasi_uRBM(std::string file_wf, bool phi_option) {

  /*#############################################################################*/
  //  File-based constructor.
  //  Initializes the (quasi)-uRBM variational parameters 𝓥 = {𝜙,𝛈,𝛒,𝒘} = {𝜙,𝛂}
  //  from a given external file in '.wf' format; this can be useful
  //  in a second moment during a check phase after the stochastic
  //  optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#############################################################################*/

  //Information
  std::cout << "#Create a 1D n.n. (quasi)-uRBM wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the (quasi)-uRBM variational parameters 𝓥 = { 𝜙,𝛈,𝛒,𝒘 } from file." << std::endl;
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
  _LocalOperators.set_size(3 * _N, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  for(unsigned int p = 0; p < _alpha.n_elem; p++)
    input_wf >> _alpha(p);

  if(input_wf.good()){

    std::cout << " (quasi)-uRBM Ansatz correctly loaded from file " << file_wf << "." << std::endl;
    std::cout << " Number of visible neurons = " << _N << "." << std::endl;
    std::cout << " Number of hidden neurons = " << _N << "." << std::endl;
    std::cout << " Density of the (quasi)-uRBM ansatz = " << this -> density() << "." << std::endl << std::endl;

  }
  input_wf.close();

}


quasi_uRBM :: ~quasi_uRBM() {

  _rnd.SaveSeed();

}


std::complex <double> quasi_uRBM :: eta_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the visible-visible interaction strength vector 𝛈." << std::endl;
    std::cerr << "   Element ηⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(j);

}


std::complex <double> quasi_uRBM :: rho_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the hidden-hidden interaction strength vector 𝛒." << std::endl;
    std::cerr << " Element ρⱼₖ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(_N + j);

}


std::complex <double> quasi_uRBM :: omega_j(unsigned int j) const {

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _N || j < 0){

    std::cerr << " ##IndexError: failed to access the visible-hidden interaction strength vector 𝒘." << std::endl;
    std::cerr << " Element 𝑤ⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }
  //Check passed
  else
    return _alpha(2 * _N + j);

}


void quasi_uRBM :: print_eta() const {

  std::cout << "\n=================================================" << std::endl;
  std::cout << "quasi_uRBM visible-visible interaction vector 𝛈" << std::endl;
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
  std::cout << "quasi_uRBM Hidden-Hidden interaction vector 𝛒" << std::endl;
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
  std::cout << "quasi_uRBM visible-hidden interaction vector 𝒘" << std::endl;
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
  //        ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽') = Σₖ (𝕆ₖ(𝒗,𝒉) - 𝕆ₖ(𝒗,𝐡ˈ)) • 𝛼ₖᴵ
  //
  //  on the given sampled configuration |𝒗 𝒉 𝒉ˈ⟩. This angle enters
  //  in the determination of the Monte Carlo averages estimation
  //  for the quantum observable during the stochastic optimization.
  //
  //  N̲O̲T̲E̲: the contribution of the variational parameter 𝜙
  //        is not to be included in the sum defining ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽').
  /*######################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh = 0.0;  //Storage variable for the hidden-hidden terms
  double II_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      II_hh += this -> rho_j(j).imag() * double(hidden_ket(0, j) * hidden_ket(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // ρᴵ_𝖭 • (𝒽_𝖭•𝒽_𝟢 - 𝒽ˈ_𝖭•𝒽ˈ_𝟢)
      II_vh += this -> omega_j(j).imag() * double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // ωᴵ_𝖭•𝓋_𝖭 • (𝒽_𝖭 - 𝒽ˈ_𝖭)

    }
    else{

      II_hh += this -> rho_j(j).imag() * double(hidden_ket(0, j) * hidden_ket(0, j+1) - hidden_bra(0, j) * hidden_ket(0, j+1));  // ρᴵ𝒿 • (𝒽𝒿•𝒽𝒿+𝟣 - 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣)
      II_vh += this -> omega_j(j).imag() * double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // ωᴵ𝒿•𝓋𝒿 • (𝒽𝒿 - 𝒽ˈ𝒿)

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
  //  Computes 𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂)) with
  //
  //        Φ(𝒗,𝒉,𝛂) = ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //
  //  Φ is that part of variational Shadow ansatz that appears in the 𝐐𝐌𝐂 calculation
  //  of a local quantum observables, i.e.
  //
  //        𝒪(𝜙, 𝛂) = ⟨Ψ(𝜙, 𝛂)| 𝒪 |Ψ(𝜙, 𝛂)⟩
  //                = Σ𝑣 Ψ⋆(𝒗, 𝜙, 𝛂) • ⟨𝒗| 𝒪 |Ψ(𝜙, 𝛂)⟩
  //                = Σ𝑣 ℯ𝓍𝓅(𝜙) • Σₕ Φ⋆(𝒗,𝒉,𝛂) • ⟨𝒗| 𝒪 |Ψ(𝜙, 𝛂)⟩
  //                = Σ𝑣ΣₕΣₕˈ ℯ𝓍𝓅(2ℜ{𝜙}) • Φ⋆(𝒗,𝒉,𝛂) • Φ(𝒗,𝒉ˈ,𝛂) • Σ𝑣ˈ ⟨𝒗| 𝒪 |𝒗ˈ⟩•(Φ(𝒗ˈ,𝒉ˈ,𝛂) / Φ(𝒗,𝒉ˈ,𝛂))
  //                = Σ𝑣ΣₕΣₕˈ 𝓆(𝑣, 𝒽, 𝒽ˈ) • 𝒪ˡᵒᶜ(𝑣, 𝒽ˈ)
  //
  //  and plays the same role as, for example, the entire wave function in the 𝐑𝐁𝐌 case,
  //  appearing as the ratio
  //
  //        Φ(𝒗ˈ,𝒉ˈ,𝛂) / Φ(𝒗,𝒉ˈ,𝛂)
  //
  //  in the calculation of 𝒪ˡᵒᶜ(𝑣, 𝒽').
  //
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                Φ(𝒗,𝒉,𝛂)
  //                   or
  //                Φ(𝒗,𝒉ˈ,𝛂).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_vv = 0.0;  //Storage variable for the visible-visible terms
  std::complex <double> log_hh = 0.0;  //Storage variable for the hidden-hidden terms
  std::complex <double> log_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      log_vv += this -> eta_j(j) * double(visible_config(0, j) * visible_config(0, 0));  // η_𝖭 • 𝓋_𝖭•𝓋_𝟢
      log_hh += this -> rho_j(j) * double(hidden_config(0, j) * hidden_config(0, 0));  // ρ_𝖭 • 𝒽_𝖭•𝒽_𝟢 or ρ_𝖭 • 𝒽ˈ_𝖭•𝒽ˈ_𝟢
      log_vh += this -> omega_j(j) * double(visible_config(0, j) * hidden_config(0, j));  // ω_𝖭 • 𝓋_𝖭•𝒽_𝖭 or ω_𝖭 • 𝓋_𝖭•𝒽ˈ_𝖭

    }
    else{

      log_vv += this -> eta_j(j) * double(visible_config(0, j) * visible_config(0, j+1));  // η𝒿 • 𝓋𝒿•𝓋𝒿+𝟣
      log_hh += this -> rho_j(j) * double(hidden_config(0, j) * hidden_config(0, j+1));  // ρ𝒿 • 𝒽𝒿•𝒽𝒿+𝟣 or ρ𝒿 • 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
      log_vh += this -> omega_j(j) * double(visible_config(0, j) * hidden_config(0, j));  // ω𝒿 • 𝓋𝒿•𝒽𝒿 or ω𝒿 • 𝓋𝒿•𝒽ˈ𝒿

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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔( Φ(𝒗ⁿᵉʷ,𝒉,𝛂)/Φ(𝒗ᵒˡᵈ,𝒉,𝛂) )." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(Φ(𝒗ⁿᵉʷ) / Φ(𝒗ᵒˡᵈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new visible configuration does not match with the number of visible variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(Φ(𝒗ⁿᵉʷ) / Φ(𝒗ᵒˡᵈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |𝒗ⁿᵉʷ⟩
    std::complex <double> log_vh = 0.0;  //Storage variable for the visible-hidden terms
    std::complex <double> log_vv = 0.0;  //Storage variable for the visible-visible terms

    //Computes the visible-hidden terms: Σ𝒿 ω𝒿•𝓋𝒿ᵒˡᵈ•𝒽𝒿 with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐬𝐢𝐭𝐞
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_visible_site(j_row, 0)) * double(visible_config(0, flipped_visible_site(j_row, 0)) * hidden_config(0, flipped_visible_site(j_row, 0)));

    //Computes the visible-visible terms: Σ𝒿 η𝒿•(𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += this -> eta_j(j) * double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // η_𝖭•(𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
      else
        log_vv += this -> eta_j(j) * double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // η𝒿•(𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)

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
  //  Computes the value of the complex natural logarithm of the 'classical' part 𝓆 of the total
  //  probability distribution
  //
  //        𝒫(𝒗,𝒉,𝒉ˈ,𝓥) = 𝓆(𝒗,𝒉,𝒉ˈ,𝓥) • [𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ)) + i𝑠𝑖𝑛(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))]
  //
  //  of the enlarged sampling space, i.e. 𝓆(𝒗,𝒉,𝒉ˈ,𝓥).
  //  The total probability distribution is defined through the sum
  //
  //        Σ𝑣Σ𝒽Σ𝒽ˈ 𝒫(𝒗,𝒉,𝒉ˈ,𝓥) = Σ𝑣 |Ψ(𝒗,𝓥)|^2 = 𝟏
  //
  //  where
  //
  //        Ψ(𝒗,𝓥) = Ψ(𝒗,𝜙,𝛂) = ℯ𝓍𝓅(𝜙)•Σₕ ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅{Σⱼₖ ηⱼₖ𝑣ⱼ𝑣ₖ} • Σₕ ℯ𝓍𝓅(Σⱼₖ ρⱼₖ𝒽ⱼ𝒽ₖ + Σⱼ 𝓌ⱼ𝑣ⱼ𝒽ₖ}
  //
  //  is the variational Shadow wave function characterized by the variational
  //  parameters {𝜙, 𝛂} = {𝜙, 𝛈, 𝛒, 𝒘}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        Σ𝑣Σ𝒽Σ𝒽' 𝓆(𝒗,𝒉,𝒉ˈ,𝓥) 𝒻(𝒗,𝒉,𝒉ˈ) = ⟨𝒻(𝒗,𝒉,𝒉ˈ)⟩𝓆 / ⟨𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))⟩𝓆.
  //
  //  So it is clear that the classical probability part 𝓆(𝒗,𝒉,𝒉ˈ,𝓥) plays the role of
  //  square modulus of the wave function with which to sample the shadow configurations |𝒗, 𝒉, 𝒉ˈ⟩
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        𝓆(𝒗,𝒉,𝒉ˈ,𝓥) = ℯ𝓍𝓅(2ℜ{𝜙}) • ℯ𝓍𝓅(ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ))
  //
  //  where
  //
  //        ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ) = Σₖ (𝕆ₖ(𝒗,𝒉) + 𝕆ₖ(𝒗,𝒉ˈ)) • αᴿₖ
  //
  //  and it has to be calculated on the current visible configuration |𝒗⟩ and the sampled
  //  hidden configuration ket |𝒉⟩ and bra ⟨𝒉ˈ|.
  /*################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ))." << std::endl;
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

      log_vv += this -> eta_j(j).real() * double(visible_config(0, j) * visible_config(0, 0));  // ηᴿ_𝖭 • 𝓋_𝖭•𝓋_𝟢
      log_hh += this -> rho_j(j).real() * double(hidden_ket(0, j) * hidden_ket(0, 0) + hidden_bra(0, j) * hidden_bra(0, 0));  // ρᴿ_𝖭 • 𝒽_𝖭•𝒽_𝟢 + 𝒽ˈ_𝖭•𝒽ˈ_𝟢
      log_vh += this -> omega_j(j).real() * double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // ωᴿ_𝖭 • 𝓋_𝖭•(𝒽_𝖭 + 𝒽ˈ_𝖭)

    }
    else{

      log_vv += this -> eta_j(j).real() * double(visible_config(0, j) * visible_config(0, j+1));  // Σ𝒿 ηᴿ𝒿 • 𝓋𝒿•𝓋𝒿+𝟣
      log_hh += this -> rho_j(j).real() * double(hidden_ket(0, j) * hidden_ket(0, j+1) + hidden_bra(0, j) * hidden_bra(0, j+1));  // Σ𝒿 ρᴿ𝒿 • 𝒽𝒿•𝒽𝒿+𝟣 + 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
      log_vh += this -> omega_j(j).real() * double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // Σ𝒿 ωᴿ𝒿 • 𝓋𝒿•(𝒽𝒿 + 𝒽ˈ𝒿)

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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
  if(flipped_visible_site.n_elem==0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ = |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: Σ𝒿 ηᴿ𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // ηᴿ_𝖭 • (𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
      else
        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // ηᴿ𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)

    }

    //Computes the visible-hidden term: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•(𝒽𝒿 + 𝒽ˈ𝒿) with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐬𝐢𝐭𝐞
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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
  if(flipped_ket_site.n_elem==0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |𝒉ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: Σ𝒿 ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // ρᴿ_𝖭 • (𝒽_𝖭ⁿᵉʷ•𝒽_𝟢ⁿᵉʷ - 𝒽_𝖭ᵒˡᵈ•𝒽_𝟢ᵒˡᵈ)
      else
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)

    }

    //Computes the visible-hidden term: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐤𝐞𝐭_𝐬𝐢𝐭𝐞
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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))
  //  evaluated in a new proposed configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥),
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
  if(flipped_bra_site.n_elem==0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: Σ𝒿 ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // ρᴿ_𝖭 • (𝒽ˈ_𝖭ⁿᵉʷ•𝒽ˈ_𝟢ⁿᵉʷ - 𝒽ˈ_𝖭ᵒˡᵈ•𝒽ˈ_𝟢ᵒˡᵈ)
      else
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)

    }

    //Computes the visible-hidden term: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽ˈ𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐛𝐫𝐚_𝐬𝐢𝐭𝐞
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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥).
  //  In this case we decide to move the spins located at the same (randomly
  //  choosen) lattice sites for all the three variables 𝒗, 𝒉, 𝒉ˈ.
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_equal_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_equal_site);  // |𝒗ⁿᵉʷ⟩
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_equal_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_equal_site);  // |𝒉ˈⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms

    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1){

        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // ηᴿ_𝖭 • (𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // ρᴿ_𝖭 • (𝒽_𝖭ⁿᵉʷ•𝒽_𝟢ⁿᵉʷ - 𝒽_𝖭ᵒˡᵈ•𝒽_𝟢ᵒˡᵈ)
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // ρᴿ_𝖭 • (𝒽ˈ_𝖭ⁿᵉʷ•𝒽ˈ_𝟢ⁿᵉʷ - 𝒽ˈ_𝖭ᵒˡᵈ•𝒽ˈ_𝟢ᵒˡᵈ)

      }
      else{

        log_vv += this -> eta_j(j).real() * double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // ηᴿ𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
        log_hh += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
        log_hh += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)

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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥).
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_ket = 0.0;  //Storage variable for the hidden-hidden ket terms
    double log_bra = 0.0;  //Storage variable for the hidden-hidden bra terms
    double log_vk = 0.0;  //Storage variable for the visible-hidden ket terms
    double log_vb = 0.0;  //Storage variable for the visible-hidden bra terms

    //𝟣𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & |𝒉ˈⁿᵉʷ⟩ = |𝒉ˈᵒˡᵈ⟩
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the hidden-hidden terms only for the ket: Σ𝒿 ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_ket += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));
        else
          log_ket += this -> rho_j(j).real() * double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += this -> omega_j(flipped_ket_site(j_row, 0)).real() * double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));

    }

    //𝟤𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ = |𝒉ᵒˡᵈ⟩ & |𝒉ˈⁿᵉʷ⟩ ≠ |𝒉ˈᵒˡᵈ⟩
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the hidden-hidden terms only for the ket: Σ𝒿 ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_bra += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));
        else
          log_bra += this -> rho_j(j).real() * double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += this -> omega_j(flipped_bra_site(j_row, 0)).real() * double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    }

    //𝟥𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & |𝒉ˈⁿᵉʷ⟩ ≠ |𝒉ˈᵒˡᵈ⟩
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

      std::cerr << " ##OptionError: something went wrong in the determination of 𝑙𝑜𝑔( 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ) )." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))." << std::endl;
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
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))." << std::endl;
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
  //  N̲O̲T̲E̲: in the Shadow ansatz the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)
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
  //  𝛂 on the sampled enlarged quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩.
  //  In the case of the (quasi)-uRBM ansatz the local parameters are 𝕆(𝒗,𝒉)
  //
  //        • η𝒿 ‹--› 𝕆(𝒗,𝒉) = 𝕆(𝒗) = 𝑣𝒿 • 𝑣𝒿+𝟣
  //        • ρ𝒿 ‹--› 𝕆(𝒗,𝒉) = 𝕆(𝒉) = 𝒽𝒿 • 𝒽𝒿+𝟣      𝕆(𝒗,𝒉ˈ) = 𝕆(𝒉ˈ) = 𝒽ˈ𝒿 • 𝒽ˈ𝒿+𝟣
  //        • ω𝒿 ‹--› 𝕆(𝒗,𝒉) = 𝒽𝒿 • 𝑣𝒿                𝕆(𝒗,𝒉ˈ) = 𝒽ˈ𝒿 • 𝑣𝒿
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }

  //Local operators for the visible-visible interactions strength 𝛈
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

  //Local operators for the hidden-hidden interactions strength 𝛒
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

  //Local operators for the hidden-hidden interactions strength 𝒘
  for(unsigned int j = 0; j < _N; j++){

    _LocalOperators(2 + _N + j, 0) = double(visible_config(0, j) * hidden_ket(0, j));
    _LocalOperators(2 + _N + j, 1) = double(visible_config(0, j) * hidden_bra(0, j));

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  𝐁𝐀𝐄𝐑𝐈𝐒𝐖𝐘𝐋-𝐒𝐇𝐀𝐃𝐎𝐖 𝐍𝐍𝐐𝐒 in 𝗱 = 𝟏  ********************************************/
/*******************************************************************************************************************************/
BS_NNQS :: BS_NNQS(unsigned int n_visible, bool phi_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the Baeriswyl-Shadow variational parameters 𝓥 = {𝜙, η, ρ, ω} = {𝜙, 𝛂} to
  //  some small random numbers.
  //  In particular we set the imaginary part of the parameters to exactly zero, while the
  //  real part is chosen randomly.
  //
  //  In this case we have only 𝟯 parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝟏 nearest-neighbors visible-visible interaction strength weights η;
  //        𝟏 nearest-neighbors hidden-hidden interaction strength weights ρ;
  //        𝟏 local visible-hidden interaction strength weights ω.
  //
  //  Note that in this case the number of variational parameters remains equal to 𝟯 for any system size 𝖭.
  /*########################################################################################################*/

  //Information
  std::cout << "#Create a 1D Baeriswyl-Shadow wave function with randomly initialized variational parameters 𝓥 = { 𝜙, η, ρ, ω }." << std::endl;

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
  _LocalOperators.set_size(3, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
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

  std::cout << " Baeriswyl-Shadow NNQS ansatz correctly initialized with random interactions." << std::endl;
  std::cout << " Number of visible variables = " << _N << "." << std::endl;
  std::cout << " Number of hidden variables = " << _N << "." << std::endl;
  std::cout << " Density of the Baeriswyl-Shadow NNQS ansatz = " << this -> density() << "." << std::endl << std::endl;

}


BS_NNQS :: BS_NNQS(std::string file_wf, bool phi_option) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the Baeriswyl-Shadow variational parameters
  //  𝓥 = {𝜙, η, ρ, ω} = {𝜙, 𝛂} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  std::cout << "#Create a 1D Baeriswyl-Shadow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the Baeriswyl-Shadow NNQS variational parameters 𝓥 = { 𝜙, η, ρ, ω } from file." << std::endl;
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
  _LocalOperators.set_size(3, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  for(unsigned int p = 0; p < _alpha.n_elem; p++)
    input_wf >> _alpha(p);

  if(input_wf.good()){

    std::cout << " Baeriswyl-Shadow NNQS Ansatz correctly loaded from file " << file_wf << "." << std::endl;
    std::cout << " Number of visible neurons = " << _N << "." << std::endl;
    std::cout << " Number of hidden neurons = " << _N << "." << std::endl;
    std::cout << " Density of the Baeriswyl-Shadow NNQS ansatz = " << this -> density() << "." << std::endl << std::endl;

  }
  input_wf.close();

}


BS_NNQS :: ~BS_NNQS() {

  _rnd.SaveSeed();

}


double BS_NNQS :: I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  /*######################################################################*/
  //  Computes the value of the angle
  //
  //        ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽') = Σₖ (𝕆ₖ(𝒗,𝒉) - 𝕆ₖ(𝒗,𝐡ˈ)) • 𝛼ₖᴵ
  //
  //  on the given sampled configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  This angle enters in the determination of the Monte Carlo averages
  //  estimation for the quantum observable during the stochastic
  //  optimization.
  //
  //  N̲O̲T̲E̲: the contribution of the variational parameter 𝜙
  //        is not to be included in the sum defining ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽').
  /*######################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉ˈ⟩
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh = 0.0;  //Storage variable for the hidden-hidden terms
  double II_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      II_hh += double(hidden_ket(0, j) * hidden_ket(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // (𝒽_𝖭•𝒽_𝟢 - 𝒽ˈ_𝖭•𝒽ˈ_𝟢)
      II_vh += double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // 𝓋_𝖭•(𝒽_𝖭 - 𝒽ˈ_𝖭)

    }
    else{

      II_hh += double(hidden_ket(0, j) * hidden_ket(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // (𝒽𝒿•𝒽𝒿+𝟣 - 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣)
      II_vh += double(visible_config(0, j) * (hidden_ket(0, j) - hidden_bra(0, j)));  // 𝓋𝒿•(𝒽𝒿 - 𝒽ˈ𝒿)

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
  //  Computes 𝑙𝑜𝑔(Φ(𝒗,𝒉,𝛂)) with
  //
  //        Φ(𝒗,𝒉,𝛂) = ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //
  //  Φ is that part of variational Shadow ansatz that appears in the 𝐐𝐌𝐂 calculation
  //  of a local quantum observables, i.e.
  //
  //        𝒜(𝜙,𝛂) = ⟨Ψ(𝜙,𝛂)| 𝔸 |Ψ(𝜙,𝛂)⟩
  //                = Σ𝑣 Ψ⋆(𝒗,𝜙,𝛂) • ⟨𝒗| 𝔸 |Ψ(𝜙,𝛂)⟩
  //                = Σ𝑣 ℯ𝓍𝓅(𝜙) • Σₕ Φ⋆(𝒗,𝒉,𝛂) • ⟨𝒗| 𝔸 |Ψ(𝜙,𝛂)⟩
  //                = Σ𝑣ΣₕΣₕˈ ℯ𝓍𝓅(2𝜙ᴿ) • Φ⋆(𝒗,𝒉,𝛂) • Φ(𝒗,𝒉ˈ,𝛂) • Σ𝑣ˈ ⟨𝒗| 𝔸 |𝒗ˈ⟩ • Φ(𝒗ˈ,𝒉ˈ,𝛂) / Φ(𝒗,𝒉ˈ,𝛂)
  //                = Σ𝑣ΣₕΣₕˈ 𝓆(𝑣, 𝒽, 𝒽ˈ) • 𝒜(𝑣,𝒽ˈ)
  //
  //  with 𝔸 a generic quantum observable operator, and plays the same role as, for example, the entire wave
  //  function in the 𝐑𝐁𝐌 case, appearing as the ratio
  //
  //        Φ(𝒗ˈ,𝒉ˈ,𝛂) / Φ(𝒗,𝒉ˈ,𝛂)
  //
  //  in the above calculation.
  //
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                Φ(𝒗,𝒉,𝛂)
  //                   or
  //                Φ(𝒗,𝒉ˈ,𝛂).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }

  //Function variables
  std::complex <double> log_vv = 0.0;  //Storage variable for the visible-visible terms
  std::complex <double> log_hh = 0.0;  //Storage variable for the hidden-hidden terms
  std::complex <double> log_vh = 0.0;  //Storage variable for the visible-hidden terms

  for(unsigned int j = 0; j < _N; j++){

    //Imposing PBC
    if(j == _N-1){

      log_vv += double(visible_config(0, j) * visible_config(0, 0));  // 𝓋_𝖭•𝓋_𝟢
      log_hh += double(hidden_config(0, j) * hidden_config(0, 0));  // 𝒽_𝖭•𝒽_𝟢 or 𝒽ˈ_𝖭•𝒽ˈ_𝟢
      log_vh += double(visible_config(0, j) * hidden_config(0, j));  // 𝓋_𝖭•𝒽_𝖭 or 𝓋_𝖭•𝒽ˈ_𝖭

    }
    else{

      log_vv += double(visible_config(0, j) * visible_config(0, j+1));  // 𝓋𝒿•𝓋𝒿+𝟣
      log_hh += double(hidden_config(0, j) * hidden_config(0, j+1));  // 𝒽𝒿•𝒽𝒿+𝟣 or 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
      log_vh += double(visible_config(0, j) * hidden_config(0, j));  // 𝓋𝒿•𝒽𝒿 or 𝓋𝒿•𝒽ˈ𝒿

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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉))." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(hidden_config.n_rows != 1 || hidden_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the hidden configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉))." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new visible configuration does not match with the number of visible variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |𝒗ⁿᵉʷ⟩
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms
    double log_vv = 0.0;  //Storage variable for the visible-visible terms

    //Computes the visible-hidden terms: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿 with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐬𝐢𝐭𝐞
    for(unsigned int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++)
      log_vh += double(visible_config(0, flipped_visible_site(j_row, 0)) * hidden_config(0, flipped_visible_site(j_row, 0)));

    //Computes the visible-visible terms: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
      else
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)

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
  //  Computes the value of the real natural logarithm of the 'classical' part 𝓆 of the total
  //  probability distribution
  //
  //        𝒫(𝒗,𝒉,𝒉ˈ) = 𝓆(𝒗,𝒉,𝒉ˈ) • [𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ)) + 𝑖𝑠𝑖𝑛(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))]
  //                  = 𝖢(𝒗,𝒉,𝒉ˈ) +  𝑖•𝖲(𝒗,𝒉,𝒉ˈ)
  //
  //  of the enlarged sampling space, i.e. 𝓆(𝒗,𝒉,𝒉ˈ).
  //  The total probability distribution is defined through the sum
  //
  //        Σ𝑣Σ𝒽Σ𝒽ˈ 𝒫(𝒗,𝒉,𝒉ˈ) = Σ𝑣 |Ψ(𝒗,𝜙,𝛂)|^2 = 𝟏
  //
  //  where
  //
  //        Ψ(𝒗,𝜙,𝛂) = ℯ𝓍𝓅(𝜙) • Σₕ ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅( η • Σ𝒿 𝓋𝒿•𝓋𝒿+𝟣 ) •
  //                           • Σ𝒽 ℯ𝓍𝓅( ρ • Σ𝒿 (𝒽𝒿•𝒽𝒿+𝟣) + ω • Σ𝒿 (𝓋𝒿•𝒽𝒿) )
  //
  //  is the variational Baeriswyl-Shadow wave function characterized by the variational
  //  parameters {𝜙, 𝛂} = {𝜙, η, ρ, ω}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        Σ𝑣Σ𝒽Σ𝒽' 𝓆(𝒗,𝒉,𝒉ˈ) 𝒻(𝒗,𝒉,𝒉ˈ) = ⟨𝒻(𝒗,𝒉,𝒉ˈ)⟩𝓆 / ⟨𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))⟩𝓆
  //
  //  So it is clear that the classical probability part 𝓆(𝒗,𝒉,𝒉ˈ) plays the role of
  //  square modulus of the wave function with which to sample the shadow configurations |𝒗, 𝒉, 𝒉ˈ⟩
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        𝓆(𝒗,𝒉,𝒉ˈ) = ℯ𝓍𝓅(2𝜙ᴿ) • ℯ𝓍𝓅(ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ))
  //
  //  where
  //
  //        ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ) = Σₖ (𝕆ₖ(𝒗,𝒉) + 𝕆ₖ(𝒗,𝒉ˈ)) • 𝛼ᴿₖ
  //
  //  and it has to be calculated on the current configuration |𝒗 𝒉 𝒉ˈ⟩.
  /*################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈ))." << std::endl;
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

      log_vv += double(visible_config(0, j) * visible_config(0, 0));  // 𝓋_𝖭•𝓋_𝟢
      log_hh += double(hidden_ket(0, j) * hidden_ket(0, 0) + hidden_bra(0, j) * hidden_bra(0, 0));  // 𝒽_𝖭•𝒽_𝟢 + 𝒽ˈ_𝖭•𝒽ˈ_𝟢
      log_vh += double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // 𝓋_𝖭•(𝒽_𝖭 + 𝒽ˈ_𝖭)

    }
    else{

      log_vv += double(visible_config(0, j) * visible_config(0, j+1));  // Σ𝒿 𝓋𝒿•𝓋𝒿+𝟣
      log_hh += double(hidden_ket(0, j) * hidden_ket(0, j+1) + hidden_bra(0, j) * hidden_bra(0, j+1));  // Σ𝒿 𝒽𝒿•𝒽𝒿+𝟣 + 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
      log_vh += double(visible_config(0, j) * (hidden_ket(0, j) + hidden_bra(0, j)));  // Σ𝒿 𝓋𝒿•(𝒽𝒿 + 𝒽ˈ𝒿)

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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
  if(flipped_visible_site.n_elem==0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ = |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
      else
        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)

    }

    //Computes the visible-hidden term: Σ𝒿 𝓋𝒿ᵒˡᵈ•(𝒽𝒿 + 𝒽ˈ𝒿) with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐯𝐢𝐬𝐢𝐛𝐥𝐞_𝐬𝐢𝐭𝐞
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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
  if(flipped_ket_site.n_elem==0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |𝒉ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: Σ𝒿 (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // (𝒽_𝖭ⁿᵉʷ•𝒽_𝟢ⁿᵉʷ - 𝒽_𝖭ᵒˡᵈ•𝒽_𝟢ᵒˡᵈ)
      else
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)

    }

    //Computes the visible-hidden term: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐤𝐞𝐭_𝐬𝐢𝐭𝐞
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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))
  //  evaluated in a new proposed configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥),
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
  if(flipped_bra_site.n_elem==0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ))." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms
    double log_vh = 0.0;  //Storage variable for the visible-hidden terms

    //Computes the visible-visible term: Σ𝒿 (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1)
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // (𝒽ˈ_𝖭ⁿᵉʷ•𝒽ˈ_𝟢ⁿᵉʷ - 𝒽ˈ_𝖭ᵒˡᵈ•𝒽ˈ_𝟢ᵒˡᵈ)
      else
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)

    }

    //Computes the visible-hidden term: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽ˈ𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐛𝐫𝐚_𝐬𝐢𝐭𝐞
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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥).
  //  In this case we decide to move the spins located at the same (randomly
  //  choosen) lattice sites for all the three variables 𝒗, 𝒉, 𝒉ˈ.
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_equal_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_equal_site);  // |𝒗ⁿᵉʷ⟩
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_equal_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_equal_site);  // |𝒉ˈⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the visible-visible terms
    double log_hh = 0.0;  //Storage variable for the hidden-hidden terms

    for(unsigned int j = 0; j < _N; j++){

      //Imposing PBC
      if(j == _N-1){

        log_vv += double(new_visible_config(0, j) * new_visible_config(0, 0) - visible_config(0, j) * visible_config(0, 0));  // (𝓋_𝖭ⁿᵉʷ•𝓋_𝟢ⁿᵉʷ - 𝓋_𝖭ᵒˡᵈ•𝓋_𝟢ᵒˡᵈ)
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));  // (𝒽_𝖭ⁿᵉʷ•𝒽_𝟢ⁿᵉʷ - 𝒽_𝖭ᵒˡᵈ•𝒽_𝟢ᵒˡᵈ)
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));  // (𝒽ˈ_𝖭ⁿᵉʷ•𝒽ˈ_𝟢ⁿᵉʷ - 𝒽ˈ_𝖭ᵒˡᵈ•𝒽ˈ_𝟢ᵒˡᵈ)

      }
      else{

        log_vv += double(new_visible_config(0, j) * new_visible_config(0, j+1) - visible_config(0, j) * visible_config(0, j+1));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
        log_hh += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
        log_hh += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)

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
  //  Computes 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥).
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0)
    return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)) with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_hidden_ket = generate_config(hidden_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_hidden_bra = generate_config(hidden_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_ket = 0.0;  //Storage variable for the hidden-hidden ket terms
    double log_bra = 0.0;  //Storage variable for the hidden-hidden bra terms
    double log_vk = 0.0;  //Storage variable for the visible-hidden ket terms
    double log_vb = 0.0;  //Storage variable for the visible-hidden bra terms

    //𝟣𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| = ⟨𝒉ˈᵒˡᵈ|
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the hidden-hidden terms only for the ket: Σ𝒿 (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_ket += double(new_hidden_ket(0, j) * new_hidden_ket(0, 0) - hidden_ket(0, j) * hidden_ket(0, 0));
        else
          log_ket += double(new_hidden_ket(0, j) * new_hidden_ket(0, j+1) - hidden_ket(0, j) * hidden_ket(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(unsigned int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(visible_config(0, flipped_ket_site(j_row, 0)) * hidden_ket(0, flipped_ket_site(j_row, 0)));

    }

    //𝟤𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ = |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| ≠ ⟨𝒉ˈᵒˡᵈ|
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the hidden-hidden terms only for the ket: Σ𝒿 (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
      for(unsigned int j = 0; j < _N; j++){

        //Imposing PBC
        if(j == _N-1)
          log_bra += double(new_hidden_bra(0, j) * new_hidden_bra(0, 0) - hidden_bra(0, j) * hidden_bra(0, 0));
        else
          log_bra += double(new_hidden_bra(0, j) * new_hidden_bra(0, j+1) - hidden_bra(0, j) * hidden_bra(0, j+1));

      }

      //Computes the visible-hidden terms only for the ket: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(unsigned int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(visible_config(0, flipped_bra_site(j_row, 0)) * hidden_bra(0, flipped_bra_site(j_row, 0)));

    }

    //𝟥𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| ≠ ⟨𝒉ˈᵒˡᵈ|
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

      std::cerr << " ##OptionError: something went wrong in the determination of 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))." << std::endl;
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
    std::cerr << "   Failed to compute 𝑙𝑜𝑔(𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ))." << std::endl;
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
  //  N̲O̲T̲E̲: in the Shadow ansatz the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
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
  //  𝛂 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the Baeriswyl-Shadow NNQS ansatz the local parameters 𝕆(𝒗,𝒉) are
  //
  //        • η ‹--› 𝕆(𝒗,𝒉) = 𝕆(𝒗) = Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣
  //        • ρ ‹--› 𝕆(𝒗,𝒉) = 𝕆(𝒉) = Σ𝒿 𝒽𝒿•𝒽𝒿+𝟣       𝕆(𝒗,𝒉ˈ) = 𝕆(𝒉ˈ) = Σ𝒿 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
  //        • ω ‹--› 𝕆(𝒗,𝒉) = Σ𝒿 𝒽𝒿•𝑣𝒿                 𝕆(𝒗,𝒉ˈ) = Σ𝒿 𝑣𝒿•𝒽ˈ𝒿
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
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the visible configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(hidden_ket.n_rows != 1 || hidden_ket.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(hidden_bra.n_rows != 1 || hidden_bra.n_cols != _N){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of visible variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this ansatz (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝕆(𝒗,𝒉) and 𝕆(𝒗,𝒉ˈ)." << std::endl;
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

  _LocalOperators(0, 0) = O_vv;  // 𝕆_η(𝒗)
  _LocalOperators(0, 1) = O_vv;  // 𝕆_η(𝒗)
  _LocalOperators(1, 0) = O_hh_ket;  // 𝕆_ρ(𝒉)
  _LocalOperators(1, 1) = O_hh_bra;  // 𝕆_ρ(𝒉ˈ)
  _LocalOperators(2, 0) = O_vh_ket;  // 𝕆_ω(𝒗,𝒉)
  _LocalOperators(2, 1) = O_vh_bra;  // 𝕆_ω(𝒗,𝒉ˈ)

}


#endif
