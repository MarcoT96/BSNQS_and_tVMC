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


#ifndef __ANSATZ__
#define __ANSATZ__


/*********************************************************************************************************/
/*****************************  𝑹𝒆𝒑𝒓𝒆𝒔𝒆𝒏𝒕𝒂𝒕𝒊𝒐𝒏 𝒐𝒇 𝒕𝒉𝒆 𝑴𝒂𝒏𝒚-𝑩𝒐𝒅𝒚 𝑸𝒖𝒂𝒏𝒕𝒖𝒎 𝑺𝒕𝒂𝒕𝒆  ******************************/
/********************************************************************************************************/
/*

  We create several 𝒜𝓃𝓈𝒶𝓉𝓏ℯ in order to represent the quantum state of a many-body system defined
  in the discrete, on a 𝟏 and 𝟐 dimensional lattice 𝚲 ϵ ℤᵈ.
  The structure of the wave function is designed in a Variational Monte Carlo frameworks,
  that is, all the knowledge about the quantum state is encoded by a set of variational parameters
  that characterizes the generic interface of our classes (in the C++ language this is obtained
  through the use of an Abstract Class).
  These parameters should be optimized via a Variational Monte Carlo algorithm (𝐬𝐚𝐦𝐩𝐥𝐞𝐫.𝐜𝐩𝐩).
  Moreover, we are interested in building variational quantum states that are Artificial Intelligence
  (𝑨𝑰) inspired, so we always consider the presence of a certain number of 𝐑𝐄𝐀𝐋 variables,
  also said 𝓋𝒾𝓈𝒾𝒷𝑙ℯ in the context of 𝑨𝑰, (i.e. the actual quantum degrees of freedom of the systems),
  supported by a certain number of 𝐒𝐇𝐀𝐃𝐎𝐖 variables, also said 𝒽𝒾𝒹𝒹ℯ𝓃, (auxiliary quantum degrees of
  freedom); the different types (𝓇ℯ𝒶𝑙 or 𝓈ℎ𝒶𝒹ℴ𝓌) of variables are organized into distinct layers,
  according to a neural-inspired 𝒜𝓃𝓈𝒶𝓉𝓏.
  Depending on the chosen architecture, there may be intra-layer interactions between variables
  of the same type and/or interactions between different variables that live in different layers.
  Even more, in some variational wave function the 𝓈ℎ𝒶𝒹ℴ𝓌 variables will be traceable, and therefore
  we will have to worry only about the 𝓇ℯ𝒶𝑙 particles (as in the 𝐑𝐁𝐌 neural network); in the generic
  case, however, the fictitious quantum variables will not be analytically integrable, and we should
  use more sophisticated sampling techniques (the 𝓈ℎ𝒶𝒹ℴ𝓌 case).
  However, we will consider complex variational parameters, and a generic form of the type

            Ψ(𝒗,𝓥) = ℯ𝓍𝓅(𝜙)•Σₕ ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉)αₖ) = ℯ𝓍𝓅(𝜙)•Σₕ Φ(𝒗,𝒉,𝛂)

  with 𝓞ₖ(𝒗,𝒉) the so-called local operators, 𝜙 a global complex phase, and 𝓥 = {𝜙,𝛂} ϵ ℂⁿ-ᵖᵃʳᵃᵐˢ.

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
#include "random.cpp"  // <-- Random


using namespace arma;


  /*###########################################*/
 /*  𝐕𝐀𝐑𝐈𝐀𝐓𝐈𝐎𝐍𝐀𝐋 𝐖𝐀𝐕𝐄 𝐅𝐔𝐍𝐂𝐓𝐈𝐎𝐍 𝐈𝐍𝐓𝐄𝐑𝐅𝐀𝐂𝐄  */
/*###########################################*/
class WaveFunction {

  protected:

    //Geometric structure
    cx_double _phi;  //The global phase variational parameter 𝜙
    cx_vec _alpha;  //The variational parameters 𝛂 = {α𝟣, α𝟤, …, αⁿ-ᵖᵃʳᵃᵐˢ}
    cx_mat _LocalOperators;  //The local operators 𝓞(𝒗,𝒉)

    //Architecture
    int _L;  //Number of 𝓇ℯ𝒶𝑙 variables 𝒗 = {𝑣𝟣, 𝑣𝟤, …, 𝑣𝖫}
    std::string _type;  //Type of 𝒜𝓃𝓈𝒶𝓉𝓏
    bool _if_PHI;  //Chooses 𝜙 ≠ 𝟢 (true) or 𝜙 = 𝟢 (false)
    bool _if_ZERO_IMAGINARY_PART;  //Chooses whether to initialize the imaginary parts of 𝛂 to zero

    //Random device
    Random _rnd;

  public:

    //Constructor and Destructor
    WaveFunction(int n_real, bool phi_option, bool imaginary_part_option) : _L(n_real), _if_PHI(phi_option), _if_ZERO_IMAGINARY_PART(imaginary_part_option) {}  //Base constructor of a spin wave function
    virtual ~WaveFunction() = default;  //Necessary for dynamic allocation


    /****************************/
    /*  𝒩ℴ𝓃-𝓋𝒾𝓇𝓉𝓊𝒶𝑙 𝒻𝓊𝓃𝒸𝓉𝒾ℴ𝓃  */
    /***************************/
    //Access functions
    int n_real() const {return _L;}  //Returns the number of 𝓇ℯ𝒶𝑙 variables 𝒗 = {𝑣𝟣, 𝑣𝟤, …, 𝑣𝖫}
    std::string type_of_ansatz() const {return _type;}  //Returns the type of the chosen 𝒜𝓃𝓈𝒶𝓉𝓏 architecture
    bool if_phi_neq_zero() const {return _if_PHI;}  //Returns whether or not to use the global phase 𝜙 in the 𝒜𝓃𝓈𝒶𝓉𝓏
    int n_alpha() const {return _alpha.n_elem;}  //Returns the number of variational parameters 𝛂 = {α𝟣, α𝟤, …, αⁿ-ᵖᵃʳᵃᵐˢ}
    cx_double phi() const {return _phi;}  //Returns the global phase variational parameter 𝜙
    cx_vec alpha() const {return _alpha;}  //Returns the set of 𝛂 = {α𝟣, α𝟤, …, αⁿ-ᵖᵃʳᵃᵐˢ}
    cx_mat O() const {return _LocalOperators;}  //Returns the local operators 𝓞(𝒗,𝒉)
    cx_double alpha_at(int) const;  //Returns a selected variational parameter α𝒿

    //Modifier functions
    void set_phi(cx_double new_phi) {_phi = new_phi;}  //Changes the value of the global phase variational parameter 𝜙
    void set_alpha(const cx_vec& new_alpha) {_alpha = new_alpha;}  //Changes the value of the variational parameters 𝛂 = {α𝟣, α𝟤, …, αⁿ-ᵖᵃʳᵃᵐˢ}
    void set_alpha_at(int, cx_double);  //Changes the value of a selected variational parameter α𝒿
    void set_phi_real(double new_phi_real) {_phi.real(new_phi_real);}  //Changes the value of the real part of the global phase 𝜙ᴿ
    void set_phi_imag(double new_phi_imag) {_phi.imag(new_phi_imag);}  //Changes the value of the imaginary part of the global phase 𝜙ᴵ
    void set_alpha_real_at(int, double);  //Changes the value of the real part of a selected variational parameter αᴿ𝒿
    void set_alpha_imag_at(int, double);  //Changes the value of the imaginary part of a selected variational parameter αᴵ𝒿

    //Functional form of the 𝒜𝓃𝓈𝒶𝓉𝓏
    Mat <int> generate_config(const Mat <int>&, const Mat <int>&) const;  //Reconstructs a system configuration |𝒗⟩, |𝒉⟩ or |𝒉ˈ⟩ from its vector representation


    /***********************/
    /*  𝒱𝒾𝓇𝓉𝓊𝒶𝑙 𝒻𝓊𝓃𝒸𝓉𝒾ℴ𝓃  */
    /***********************/
    //Access function
    virtual int shadow_density() const = 0;  //Returns the density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ν = 𝖬 / 𝖫

    //Modifier functions
    virtual void Init_on_Config(const Mat <int>&) = 0;  //Initializes properly the 𝒜𝓃𝓈𝒶𝓉𝓏 on a given quantum configuration
    virtual void Update_on_Config(const Mat <int>&, const Mat <int>&) = 0;  //Updates properly the 𝒜𝓃𝓈𝒶𝓉𝓏 on a given new sampled quantum configuration

    //Functional form of the 𝒜𝓃𝓈𝒶𝓉𝓏
    virtual double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    virtual double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes cos[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    virtual double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes sin[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    virtual cx_double logPhi(const Mat <int>&, const Mat <int>&) const = 0;  //Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)]
    virtual cx_double Phi(const Mat <int>&, const Mat <int>&) const = 0;  //Computes Φ(𝒗,𝒉,𝛂)
    virtual cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]
    virtual cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
    virtual cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the natural logarithm of the ”Metropolis part” of the 𝒜𝓃𝓈𝒶𝓉𝓏
    virtual cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const = 0;  //Computes the ”Metropolis part” of the 𝒜𝓃𝓈𝒶𝓉𝓏
    virtual cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes the natural logarithm of the sqrt of the Matropolis acceptance probability
                                            const Mat <int>&, const Mat <int>&,
                                            const Mat <int>&, const Mat <int>&,
                                            std::string option="") const = 0;
    virtual cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes the sqrt of the Matropolis acceptance probability
                                         const Mat <int>&, const Mat <int>&,
                                         const Mat <int>&, const Mat <int>&,
                                         std::string option="") const = 0;
    virtual double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance
                                            const Mat <int>&, const Mat <int>&,  //probability || Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                            const Mat <int>&, const Mat <int>&,
                                            std::string option="") const = 0;
    virtual void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&) = 0;  //Computes the local operators 𝓞(𝒗,𝒉)

};


  /*####################################*/
 /*  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐰𝐢𝐭𝐡 𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒  */
/*####################################*/
class JWF : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    JWF(int, bool, bool, int);
    JWF(std::string, bool, int);
    ~JWF() {};

    //Access functions
    int shadow_density() const {return 0;}
    cx_double eta() const {return _alpha[0];}  //Returns the nearest-neighbors coupling parameter

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}
    double cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝜂)] on a given 𝓇ℯ𝒶𝑙 configuration
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝜂) on a given 𝓇ℯ𝒶𝑙 configuration
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝜂) / Ψ(𝒗ᵒˡᵈ,𝜙,𝜂)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝜂) / Ψ(𝒗ᵒˡᵈ,𝜙,𝜂)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝜂)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝜂)
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝜂) / Ψ(𝒗ᵒˡᵈ,𝜙,𝜂)]
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝜂) / Ψ(𝒗ᵒˡᵈ,𝜙,𝜂)
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝜙,𝜂) / Ψ(𝒗ᵒˡᵈ,𝜙,𝜂) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗) = ∂𝑙𝑜𝑔[Ψ(𝒗,𝛂)] / ∂𝛂

};


  /*#####################################################*/
 /*  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐰𝐢𝐭𝐡 𝐈𝐍𝐇𝐎𝐌𝐎𝐆𝐄𝐍𝐄𝐎𝐔𝐒 𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒  */
/*#####################################################*/
class JWF_inhom : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    JWF_inhom(int, bool, bool, int);
    JWF_inhom(std::string, bool, int);
    ~JWF_inhom() {};

    //Access functions
    int shadow_density() const {return 0;}
    cx_double eta_j(int) const;  //Returns the selected 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 Jastrow interaction strength

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}
    double cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)] on a given 𝓇ℯ𝒶𝑙 configuration
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝛈) on a given 𝓇ℯ𝒶𝑙 configuration
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝛈)
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)]
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗) = ∂𝑙𝑜𝑔[Ψ(𝒗,𝛂)] / ∂𝛂

};


  /*########################################*/
 /*  𝐋𝐎𝐍𝐆-𝐑𝐀𝐍𝐆𝐄 𝐇𝐎𝐌𝐎𝐆𝐄𝐍𝐄𝐎𝐔𝐒 𝐉𝐀𝐒𝐓𝐑𝐎𝐖  */
/*########################################*/
class LRHJas : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    LRHJas(int, bool, bool, int);
    LRHJas(std::string, bool, int);
    ~LRHJas() {};

    //Access functions
    int shadow_density() const {return 0;}
    cx_double eta_j(int) const;  //Returns the selected 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 Jastrow interaction strength

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}
    double cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)] on a given 𝓇ℯ𝒶𝑙 configuration
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝛈) on a given 𝓇ℯ𝒶𝑙 configuration
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝛈)
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)]
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗) = ∂𝑙𝑜𝑔[Ψ(𝒗,𝛂)] / ∂𝛂

};


  /*##############################################*/
 /*  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐍𝐄𝐔𝐑𝐀𝐋 𝐍𝐄𝐓𝐖𝐎𝐑𝐊 𝐐𝐔𝐀𝐍𝐓𝐔𝐌 𝐒𝐓𝐀𝐓𝐄  */
/*##############################################*/
class JasNQS : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    JasNQS(int, bool, bool, int);
    JasNQS(std::string, bool, int);
    ~JasNQS() {};

    //Access functions
    int shadow_density() const {return 0;}
    cx_double w_jk(int, int) const;  //Returns the selected 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ interaction strength
    void print_W() const;  //Prints on standard output the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ interaction strength matrix 𝕎

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}
    double cosII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)] on a given 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝛈) on a given 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝜙,𝛈)
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)]
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈)
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝜙,𝛈) / Ψ(𝒗ᵒˡᵈ,𝜙,𝛈) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗) = ∂𝑙𝑜𝑔[Ψ(𝒗,𝛂)] / ∂𝛂

};


  /*####################################*/
 /*  𝐑𝐄𝐒𝐓𝐑𝐈𝐂𝐓𝐄𝐃 𝐁𝐎𝐋𝐓𝐙𝐌𝐀𝐍𝐍 𝐌𝐀𝐂𝐇𝐈𝐍𝐄  */
/*####################################*/
class RBM : public WaveFunction {

  private:

    //RBM Neural Network architecture
    int _M;  //Number of 𝒽𝒾𝒹𝒹ℯ𝓃 neurons 𝐡 = {𝒽𝟣, 𝒽𝟤, …, 𝒽𝖬}

    //Look-up table for the effective angles 𝛳(𝒗,𝛂)
    cx_vec _Theta;

    //Fast computation of the wave function
    const double _ln2;  //𝑙𝑜𝑔𝟤

  public:

    //Constructor and Destructor
    RBM(int, int, bool, bool, int);
    RBM(std::string, bool, int);
    ~RBM() {};

    //Access functions
    int shadow_density() const {return _M / _L;}
    int n_hidden() const {return _M;}  //Returns the number of 𝒽𝒾𝒹𝒹ℯ𝓃 neurons 𝐡 = {𝒽𝟣, 𝒽𝟤, …, 𝒽𝖬}
    cx_double a_j(int) const;  //Returns the bias of the 𝒿-th 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neuron
    cx_double b_k(int) const;  //Returns the bias of the 𝓀-th 𝒽𝒾𝒹𝒹ℯ𝓃 neuron
    cx_double W_jk(int, int) const;  //Returns the selected 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interaction strength
    cx_double Theta_k(int) const;  //Returns the effective angles associated to the 𝓀-th 𝒽𝒾𝒹𝒹ℯ𝓃 neuron
    cx_vec effective_angle() const {return _Theta;}  //Returns the set of 𝛳(𝒗,𝛂)
    void print_a() const;  //Prints on standard output the set of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ bias 𝐚
    void print_b() const;  //Prints on standard output the set of 𝒽𝒾𝒹𝒹ℯ𝓃 bias 𝐛
    void print_W() const;  //Prints on standard output the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interaction strength matrix 𝕎
    void print_Theta() const;  //Prints on standard output the set of effective angles 𝛳(𝒗,𝛂)

    //Modifier functions
    void Init_on_Config(const Mat <int>&);
    void Update_on_Config(const Mat <int>&, const Mat <int>&);

    //Wavefunction evaluation
    double lncosh(double) const;  //Computes 𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓍) of a real number 𝓍 ϵ ℝ
    cx_double lncosh(cx_double) const;  //Computes 𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓏) of a complex number 𝓏 ϵ ℂ
    void Init_Theta(const Mat <int>&);  //Initializes the effective angles 𝛳(𝒗,𝛂) on the given 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration |𝒗⟩
    void Update_Theta(const Mat <int>&, const Mat <int>&);  //Updates the effective angles 𝛳(𝒗,𝛂) on a new sampled 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration |𝒗ⁿᵉʷ⟩
    double I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}
    double cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 1.0;}  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {return 0.0;}  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝓥)] on a given 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝓥) on a given 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗,𝓥)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Ψ(𝒗,𝓥)
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥)]
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&, std::string option="") const;  //Computes Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥)
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗) = ∂𝑙𝑜𝑔[Ψ(𝒗,𝛂)] / ∂𝛂

};


  /*############################################*/
 /*  𝐒𝐇𝐀𝐃𝐎𝐖 𝐑𝐄𝐒𝐓𝐑𝐈𝐂𝐓𝐄𝐃 𝐁𝐎𝐋𝐓𝐙𝐌𝐀𝐍𝐍 𝐌𝐀𝐂𝐇𝐈𝐍𝐄  */
/*#############################################*/
class SRBM : public WaveFunction {};


  /*#####################################*/
 /*  𝐁𝐀𝐄𝐑𝐈𝐒𝐖𝐘𝐋-𝐒𝐇𝐀𝐃𝐎𝐖 𝐍𝐍𝐐𝐒 in 𝗱 = 𝟏  */
/*#####################################*/
class BSWF : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    BSWF(int, bool, bool, int);
    BSWF(std::string, bool, int);
    ~BSWF() {};

    //Access functions
    int shadow_density() const {return 1;}
    cx_double eta() const {return _alpha[0];}  //Returns the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength η
    cx_double rho() const {return _alpha[1];}  //Returns the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength ρ
    cx_double omega() const {return _alpha[2];}  //Returns the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength ω

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)]
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗,𝒉,𝛂)
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ,𝓥)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_real(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)]
    double q_over_q_real(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)]
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥)  / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)]
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)] on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)]
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)]
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                                 const Mat <int>&, const Mat <int>&,
                                 const Mat <int>&, const Mat <int>&,
                                 std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗,𝒉) = ∂𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)] / ∂𝛂

};


  /*#########################################################################*/
 /*  𝐍𝐄𝐗𝐓-𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒 𝐁𝐀𝐄𝐑𝐈𝐒𝐖𝐘𝐋-𝐒𝐇𝐀𝐃𝐎𝐖 𝐖𝐀𝐕𝐄 𝐅𝐔𝐍𝐂𝐓𝐈𝐎𝐍 in 𝗱 = 𝟏  */
/*#########################################################################*/
class NNN_BSWF : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    NNN_BSWF(int, bool, bool, int);
    NNN_BSWF(std::string, bool, int);
    ~NNN_BSWF() {};

    //Access functions
    int shadow_density() const {return 1;}
    cx_double eta() const {return _alpha[0];}  //Returns the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength η
    cx_double rho1() const {return _alpha[1];}  //Returns the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 interaction strength ρ𝟣
    cx_double rho2() const {return _alpha[2];}  //Returns the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 𝓃ℯ𝓍𝓉-𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 interaction strength ρ𝟤
    cx_double omega() const {return _alpha[3];}  //Returns the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength ω

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)]
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗,𝒉,𝛂)
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ,𝓥)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_real(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)]
    double q_over_q_real(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)]
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥)  / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)]
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)] on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)]
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)]
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                                 const Mat <int>&, const Mat <int>&,
                                 const Mat <int>&, const Mat <int>&,
                                 std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&,
                                    std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗,𝒉) = ∂𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)] / ∂𝛂

};


  /*#############################################*/
 /*  𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒 (quasi)-𝐮𝐑𝐁𝐌 in 𝗱 = 𝟏  */
/*#############################################*/
class quasi_uRBM : public WaveFunction {

  private:

    /*
      ......
      ......
      ......
    */

  public:

    //Constructor and Destructor
    quasi_uRBM(int, bool, bool, int);
    quasi_uRBM(std::string, bool, int);
    ~quasi_uRBM() {};

    //Access functions
    int shadow_density() const {return 1;}
    cx_double eta_j(int) const;  //Returns the selected 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength
    cx_double rho_j(int) const;  //Returns the selected 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength
    cx_double omega_j(int) const;  //Returns the selected 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength
    void print_eta() const;  //Prints on standard output the set of 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength 𝛈
    void print_rho() const;  //Prints on standard output the set of 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength 𝛒
    void print_omega() const;  //Prints on standard output the set of 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength 𝒘

    //Modifier functions
    void Init_on_Config(const Mat <int>&) {}
    void Update_on_Config(const Mat <int>&, const Mat <int>&) {}

    //Wavefunction evaluation
    double I_minus_I(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')
    double cosII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑐𝑜𝑠[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    double sinII(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑠𝑖𝑛[ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')]
    cx_double logPhi(const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)]
    cx_double Phi(const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗,𝒉,𝛂)
    cx_double logPhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]
    cx_double PhiNew_over_PhiOld(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)
    cx_double logPsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ,𝓥)]
    cx_double PsiMetro(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_real(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)]
    double q_over_q_real(const Mat <int>&, const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ,𝓥)
    double logq_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)]
    double q_over_q_ket(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ,𝓥)
    double logq_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)]
    double q_over_q_bra(const Mat <int>&, const Mat <int>&, const Mat <int>&) const;  //Computes 𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ,𝓥)
    double logq_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)] on the same flipped site
                                  const Mat <int>&, const Mat <int>&) const;
    double q_over_q_equal_site(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥) on the same flipped site
                               const Mat <int>&, const Mat <int>&) const;
    double logq_over_q_braket(const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)]
                              const Mat <int>&, const Mat <int>&,
                              const Mat <int>&, const Mat <int>&) const;
    double q_over_q_braket(const Mat <int>&,  //Computes 𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                           const Mat <int>&, const Mat <int>&,
                           const Mat <int>&, const Mat <int>&) const;
    cx_double logPsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝑙𝑜𝑔[𝓆( 𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)]
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&, std::string option="") const;
    cx_double PsiNew_over_PsiOld(const Mat <int>&, const Mat <int>&,  //Computes 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
                                 const Mat <int>&, const Mat <int>&,
                                 const Mat <int>&, const Mat <int>&, std::string option="") const;
    double PMetroNew_over_PMetroOld(const Mat <int>&, const Mat <int>&,  //Computes the Metropolis-Hastings acceptance probability || Ψ(𝒗ⁿᵉʷ,𝓥) / Ψ(𝒗ᵒˡᵈ,𝓥) ||
                                    const Mat <int>&, const Mat <int>&,
                                    const Mat <int>&, const Mat <int>&, std::string option="") const;
    void LocalOperators(const Mat <int>&, const Mat <int>&, const Mat <int>&);  //Computes the local operators 𝓞(𝒗,𝒉) = ∂𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)] / ∂𝛂

};




/*******************************************************************************************************************************/
/******************************************  𝐕𝐀𝐑𝐈𝐀𝐓𝐈𝐎𝐍𝐀𝐋 𝐖𝐀𝐕𝐄 𝐅𝐔𝐍𝐂𝐓𝐈𝐎𝐍 𝐈𝐍𝐓𝐄𝐑𝐅𝐀𝐂𝐄  ******************************************/
/*******************************************************************************************************************************/
cx_double WaveFunction :: alpha_at(int j) const {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to access the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter α𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[j];

}


void WaveFunction :: set_alpha_at(int j, cx_double new_param) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter α𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    std::abort();

  }

  //Check passed
  else _alpha[j] = new_param;

}


void WaveFunction :: set_alpha_real_at(int j, double new_param_real) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the real part of the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter αᴿ𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    std::abort();

  }

  //Check passed
  else _alpha[j].real(new_param_real);

}


void WaveFunction :: set_alpha_imag_at(int j, double new_param_imag) {

  //Check on the selected index
  if(j < 0 || j >= _alpha.n_elem){

    std::cerr << " ##IndexError: failed to modify the imaginary part of the variational parameter set 𝛂." << std::endl;
    std::cerr << "   The variational parameter αᴵ𝒿 with 𝒿 = " << j << " does not exist." << std::endl;
    std::abort();

  }

  //Check passed
  else _alpha[j].imag(new_param_imag);

}


Mat <int> WaveFunction :: generate_config(const Mat <int>& old_config, const Mat <int>& flipped_site) const {

  //Function variables
  Mat <int> new_config;

  //Reconstruction
  if(flipped_site.n_elem != 0){

    new_config = old_config;
    for(int j_row = 0; j_row < flipped_site.n_rows; j_row++) new_config.at(0, flipped_site.at(j_row, 0)) *= -1;
    return new_config;

  }
  else return old_config;

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐰𝐢𝐭𝐡 𝐍𝐄𝐀𝐑𝐄𝐒𝐓 𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒  *********************************************/
/*******************************************************************************************************************************/
JWF :: JWF(int n_real, bool phi_option, bool imaginary_part_option, int rank)
     : WaveFunction(n_real, phi_option, imaginary_part_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the nearest-neighbors entangling Jastrow variational parameters
  //  𝓥 = {𝜙, 𝜂} = {𝜙, 𝛂} to some small random numbers.
  //
  //  In this case we have only 𝟭 parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝟏 nearest-neighbors 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength 𝜂.
  //
  //  Note that in this case the number of variational parameters remains equal to 𝟏 for any system size 𝖫.
  /*########################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a nearest-neighbors Jastrow wave function with randomly initialized variational parameters 𝓥 = {𝜙, 𝜂}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Jastrow";
  _alpha.set_size(1);
  _LocalOperators.zeros(1, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  _alpha[0].real(_rnd.Gauss(0.0, 0.001));
  if(_if_ZERO_IMAGINARY_PART) _alpha[0].imag(0.0);
  else _alpha[0].imag(_rnd.Gauss(0.0, 0.001));

  //Ends construction
  if(rank == 0){

    std::cout << " Nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
    std::cout << " Nearest-neighbors entangling parameter at initial time → 𝜂(𝟢) = " << _alpha[0] << "." << std::endl << std::endl;

  }

}


JWF :: JWF(std::string file_wf, bool phi_option, int rank)
     : WaveFunction(0, phi_option, 0) {

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
  if(rank == 0) std::cout << "#Create a nearest-neighbors Jastrow wave function from an existing quantum state." << std::endl;

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
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Jastrow";
  input_wf >> _L;
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  _alpha.set_size(1);
  _LocalOperators.zeros(1, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  input_wf >> _alpha[0];

  //Ends construction
  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
      std::cout << " Nearest-neighbors entangling parameter at initial time → 𝜂(𝟢) = " << _alpha[0] << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


cx_double JWF :: logPhi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  /*####################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝜂)] with
  //
  //        Ψ(𝒗,𝜙,𝜂) = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(𝜂 Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣).
  //
  //  Obviously, this 𝒜𝓃𝓈𝒶𝓉𝓏 is not of the 𝓈ℎ𝒶𝒹ℴ𝓌 type,
  //  and no auxiliary variables are introduced here.
  /*####################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝜂)]." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_psi = 0.0;  //Storage variable for the nearest-neighbors interaction

  for(int j = 0; j < _L; j++) log_psi += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // 𝓋𝒿 • 𝓋𝒿+𝟣 in PBCs

  return this -> phi() + this -> eta() * log_psi;

}


cx_double JWF :: Phi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhi(real_config, shadow_config));

}


cx_double JWF :: logPhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                       const Mat <int>& shadow_config) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)] at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the old 𝓇ℯ𝒶𝑙 configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the 𝐔𝐩𝐝𝐚𝐭𝐞_𝐓𝐡𝐞𝐭𝐚() function
  //  defined below in the 𝐑𝐁𝐌 class.
  //  Note that the ratio between the two evaluated wave function, which is the
  //  quantity related to the acceptance kernel of the Metropolis algorithm,
  //  is recovered by taking the exponential of the output of this function.
  //
  //  N̲O̲T̲E̲: once again we emphasize that in the specific case of the Jastrow
  //        𝒜𝓃𝓈𝒶𝓉𝓏 the quantities calculated with the functions inherent to
  //        Φ(𝒗,𝒉,𝛂) correspond to those calculated in the functions related
  //        to the Metropolis algorithm, since we have never introduced any
  //        auxiliary variable.
  //  N̲O̲T̲E̲: the 𝒔𝒉𝒂𝒅𝒐𝒘_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏,
  //        which does not depend upon any 𝓈ℎ𝒶𝒹ℴ𝓌 variables.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled 𝓇ℯ𝒶𝑙 configuration |𝒗ⁿᵉʷ⟩
  if(flipped_real_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    return this -> eta() * log_vv;

  }

}


cx_double JWF :: PhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                    const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_config));

}


cx_double JWF :: logPsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return this -> logPhi(real_config, shadow_ket);

}


cx_double JWF :: PsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return this -> Phi(real_config, shadow_ket);

}


cx_double JWF :: logPsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                       const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                       const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                       std::string option) const {

  return this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_ket);

}


cx_double JWF :: PsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                    const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                    const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                    std::string option) const {

  return this -> PhiNew_over_PhiOld(real_config, flipped_real_site, shadow_ket);

}


double JWF :: PMetroNew_over_PMetroOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                       const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                       const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                       std::string option) const {

  return std::norm(this -> PsiNew_over_PsiOld(real_config, flipped_real_site,
                                              shadow_ket, flipped_ket_site,
                                              shadow_bra, flipped_bra_site));

}


void JWF :: LocalOperators(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameter
  //  𝜂 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 the single local operator 𝓞(𝒗,𝒉) is
  //
  //        • 𝜂 ←→ 𝓞(𝒗,𝒉) = 𝓞(𝒗) = Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣.
  //
  //  and represents the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 correlations.
  //  This operator is necessary to compute the Quantum Geometric Tensor
  //  and the Gradient during the stochastic optimization procedure.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉)." << std::endl;
    std::abort();

  }

  //Function variables
  double O_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

  //Computes the local operator assiociated to the only parameter 𝜂
  for(int j = 0; j < _L; j++) O_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // 𝓋𝒿 • 𝓋𝒿+𝟣 in PBCs

  _LocalOperators.at(0, 0) = O_vv;  // 𝓞_η(𝒗)
  _LocalOperators.at(0, 1) = O_vv;  // 𝓞_η(𝒗)

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/*************************************  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐰𝐢𝐭𝐡 𝐈𝐍𝐇𝐎𝐌𝐎𝐆𝐄𝐍𝐄𝐎𝐔𝐒 𝐍𝐄𝐀𝐑𝐄𝐒𝐓 𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒  *************************************/
/*******************************************************************************************************************************/
JWF_inhom :: JWF_inhom(int n_real, bool phi_option, bool imaginary_part_option, int rank)
           : WaveFunction(n_real, phi_option, imaginary_part_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the nearest-neighbors entangling Jastrow variational parameters
  //  𝓥 = {𝜙, 𝛈} = {𝜙, 𝛂} to some small random numbers.
  //
  //  In this case we have 𝖫 parameters, which depend on the lattice site of the nearest
  //  neighbors they refer to.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝖫 nearest-neighbors 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength η𝒿.
  /*########################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a nearest-neighbors Jastrow wave function with inhomogeneous randomly initialized variational parameters 𝓥 = {𝜙, 𝛈}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Jastrow";
  _alpha.set_size(_L);
  _LocalOperators.zeros(_L, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  for(int p = 0; p < _alpha.n_elem; p++){

    _alpha[p].real(_rnd.Gauss(0.0, 0.001));
    if(_if_ZERO_IMAGINARY_PART) _alpha[p].imag(0.0);
    else _alpha[p].imag(_rnd.Gauss(0.0, 0.001));

  }

  //Ends construction
  if(rank == 0){

    std::cout << " Inhomogeneous nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl << std::endl;

  }

}


JWF_inhom :: JWF_inhom(std::string file_wf, bool phi_option, int rank)
           : WaveFunction(0, phi_option, 0) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the inhomogeneous nearest-neighbors Jastrow variational parameters
  //  𝓥 = {𝜙, 𝛈} = {𝜙, 𝛂} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create an inhomogeneous nearest-neighbors Jastrow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the inhomogeneous nearest-neighbors Jastrow variational parameters 𝓥 = {𝜙, 𝛈} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Jastrow";
  input_wf >> _L;
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the inhomogeneous nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  _alpha.set_size(_L);
  _LocalOperators.zeros(_L, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  for(int p = 0; p < _alpha.n_elem; p++) input_wf >> _alpha[p];

  //Ends construction
  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Inhomogeneous nearest-neighbors Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


cx_double JWF_inhom :: eta_j(int j) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || j < 0){

    std::cerr << " ##IndexError: failed to access the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength vector 𝛈." << std::endl;
    std::cerr << "   Element ηⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[j];

}


cx_double JWF_inhom :: logPhi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  /*####################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)] with
  //
  //        Ψ(𝒗,𝜙,𝛈) = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σ𝒿 𝜂𝒿•𝑣𝒿•𝑣𝒿+𝟣).
  //
  //  Obviously, this 𝒜𝓃𝓈𝒶𝓉𝓏 is not of the 𝓈ℎ𝒶𝒹ℴ𝓌 type,
  //  and no auxiliary variables are introduced here.
  /*####################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_psi = 0.0;  //Storage variable for the nearest-neighbors interaction

  for(int j = 0; j < _L; j++) log_psi += _alpha[j] * double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // 𝜂𝒿 • 𝓋𝒿 • 𝓋𝒿+𝟣 in PBCs

  return this -> phi() + log_psi;

}


cx_double JWF_inhom :: Phi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhi(real_config, shadow_config));

}


cx_double JWF_inhom :: logPhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                             const Mat <int>& shadow_config) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)] at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the old 𝓇ℯ𝒶𝑙 configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the 𝐔𝐩𝐝𝐚𝐭𝐞_𝐓𝐡𝐞𝐭𝐚() function
  //  defined below in the 𝐑𝐁𝐌 class.
  //  Note that the ratio between the two evaluated wave function, which is the
  //  quantity related to the acceptance kernel of the Metropolis algorithm,
  //  is recovered by taking the exponential of the output of this function.
  //
  //  N̲O̲T̲E̲: once again we emphasize that in the specific case of the Jastrow
  //        𝒜𝓃𝓈𝒶𝓉𝓏 the quantities calculated with the functions inherent to
  //        Φ(𝒗,𝒉,𝛂) correspond to those calculated in the functions related
  //        to the Metropolis algorithm, since we have never introduced any
  //        auxiliary variable.
  //  N̲O̲T̲E̲: the 𝒔𝒉𝒂𝒅𝒐𝒘_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏,
  //        which does not depend upon any 𝓈ℎ𝒶𝒹ℴ𝓌 variables.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled 𝓇ℯ𝒶𝑙 configuration |𝒗ⁿᵉʷ⟩
  if(flipped_real_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    cx_double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += _alpha[j] * double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // η𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    return log_vv;

  }

}


cx_double JWF_inhom :: PhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                          const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_config));

}


cx_double JWF_inhom :: logPsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return this -> logPhi(real_config, shadow_ket);

}


cx_double JWF_inhom :: PsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return this -> Phi(real_config, shadow_ket);

}


cx_double JWF_inhom :: logPsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                             const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                             const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                             std::string option) const {

  return this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_ket);

}


cx_double JWF_inhom :: PsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                          const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                          const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                          std::string option) const {

  return this -> PhiNew_over_PhiOld(real_config, flipped_real_site, shadow_ket);

}


double JWF_inhom :: PMetroNew_over_PMetroOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                             const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                             const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                             std::string option) const {

  return std::norm(this -> PsiNew_over_PsiOld(real_config, flipped_real_site,
                                              shadow_ket, flipped_ket_site,
                                              shadow_bra, flipped_bra_site));

}


void JWF_inhom :: LocalOperators(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameter
  //  𝛈 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the inhomogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 the local operators 𝓞(𝒗,𝒉) are
  //
  //        • η𝒿 ←→ 𝓞(𝒗,𝒉) = 𝓞(𝒗) = 𝑣𝒿•𝑣𝒿+𝟣
  //
  //  and represent the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 correlations.
  //  This operator is necessary to compute the Quantum Geometric Tensor
  //  and the Gradient during the stochastic optimization procedure.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉)." << std::endl;
    std::abort();

  }

  //Computes the local operators assiociated to each parameter η𝒿
  for(int j = 0; j < _L; j++){

    _LocalOperators.at(j, 0) = double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));
    _LocalOperators.at(j, 1) = _LocalOperators.at(j, 0);

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/********************************************  𝐋𝐎𝐍𝐆-𝐑𝐀𝐍𝐆𝐄 𝐇𝐎𝐌𝐎𝐆𝐄𝐍𝐄𝐎𝐔𝐒 𝐉𝐀𝐒𝐓𝐑𝐎𝐖  *******************************************/
/*******************************************************************************************************************************/
LRHJas :: LRHJas(int n_real, bool phi_option, bool imaginary_part_option, int rank)
        : WaveFunction(n_real, phi_option, imaginary_part_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the long-range homogeneous Jastrow variational parameters
  //  𝓥 = {𝜙, 𝛈} = {𝜙, 𝛂} to some small random numbers.
  //
  //  In this case we have ⌊𝖫/𝟤⌋ parameters, where ⌊◦⌋ is the greatest integer smaller than ◦,
  //  which do not depend on the lattice site and represent the nearest-neighbors,
  //  next-to nearest-neighbors, next-to next-to nearest-neighbors, etc..., interaction between
  //  the 𝓇ℯ𝒶𝑙 degrees of freedom. In other words, all nearest-neighbor sites interact with a
  //  strength given by η𝟣, all next-to nearest-neighbor sites interact with a strength given
  //  by η𝟤, and so on.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        ⌊𝖫/𝟤⌋ 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength η𝒿 at different range of action.
  /*########################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a long-range homogeneous Jastrow wave function with randomly initialized variational parameters 𝓥 = {𝜙, 𝛈}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Jastrow";
  int P = std::floor(_L / 2.0);  // 𝒑 = ⌊𝖫/𝟤⌋ variational parameters αₖ
  _alpha.set_size(P);
  _LocalOperators.zeros(P, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  for(int p = 0; p < _alpha.n_elem; p++){

    _alpha[p].real(_rnd.Gauss(0.0, 0.001));
    if(_if_ZERO_IMAGINARY_PART) _alpha[p].imag(0.0);
    else _alpha[p].imag(_rnd.Gauss(0.0, 0.001));

  }

  //Ends construction
  if(rank == 0){

    std::cout << " Long-range homogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl << std::endl;

  }

}


LRHJas :: LRHJas(std::string file_wf, bool phi_option, int rank)
        : WaveFunction(0, phi_option, 0) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the long-range homogeneous Jastrow variational parameters
  //  𝓥 = {𝜙, 𝛈} = {𝜙, 𝛂} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a long-range homogeneous Jastrow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the long-range homogeneous Jastrow variational parameters 𝓥 = {𝜙, 𝛈} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Jastrow";
  input_wf >> _L;
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the long-range homogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  int P = std::floor(_L / 2.0);
  _alpha.set_size(P);
  _LocalOperators.zeros(P, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  for(int p = 0; p < _alpha.n_elem; p++) input_wf >> _alpha[p];

  //Ends construction
  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Long-range homogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


cx_double LRHJas :: eta_j(int j) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _alpha.n_elem || j < 0){

    std::cerr << " ##IndexError: failed to access the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength vector 𝛈." << std::endl;
    std::cerr << "   Element ηⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[j];

}


cx_double LRHJas :: logPhi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  /*######################################################################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)] in P̲B̲C̲s̲ with
  //
  //        Ψ(𝒗,𝜙,𝛈) = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σⱼₖ η_𝑙 𝑣𝒿•𝑣𝓀)
  //
  //  where 𝒿 < 𝓀, 𝒿 = 𝟢, 𝟣, 𝟤, …, 𝖫 - 𝟤, 𝓀 = 𝟢, 𝟣, 𝟤, …, 𝖫 - 𝟣 and
  //  the index l has a different definition depending on
  //  whether 𝖫 is 𝒆𝒗𝒆𝒏 or 𝒐𝒅𝒅; in particular, if 𝖫 is 𝒆𝒗𝒆𝒏 we have
  //
  //          l = ⌊𝖫 / 𝟤⌋ - | |𝒿 - 𝓀| - ⌊𝖫 / 𝟤⌋ |
  //
  //  while if 𝖫 is 𝒐𝒅𝒅 the right entangling parameters of the pair
  //  𝑣𝒿•𝑣𝓀 is given by
  //
  //        l =   𝛘(|𝒿 - 𝓀| < ⌊𝖫 / 𝟤⌋ + 1) • |𝒿 - 𝓀| +
  //            + 𝛘(|𝒿 - 𝓀| = ⌊𝖫 / 𝟤⌋ + 1) • [|𝒿 - 𝓀| - 𝟣] +
  //            + 𝛘(|𝒿 - 𝓀| > ⌊𝖫 / 𝟤⌋ + 1) • [⌊𝖫 / 𝟤⌋ + 1 - (|𝒿 - 𝓀| - ⌊𝖫 / 𝟤⌋)]
  //
  //  where 𝛘 is the characteristic function.
  //  Note that the sum is over 𝒿 < 𝓀 in order to count interacting spin pairs at
  //  a certain distance only once, without repetitions.
  //  Obviously, this 𝒜𝓃𝓈𝒶𝓉𝓏 is not of the 𝓈ℎ𝒶𝒹ℴ𝓌 type, and no auxiliary variables are introduced here.
  /*######################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]." << std::endl;
    std::abort();

  }

  //Function variables
  int l_max = _alpha.n_elem;  // The maximum distance in PBCs, i.e. ⌊𝖫 / 𝟤⌋
  int l = 0;  // The correct index of pairwise interaction
  double d = 0;
  cx_double log_psi = 0.0;  //Storage variable

  //Computes all the interactions in PBCs
  for(int j = 0; j < _L - 1; j++){

    for(int k = j + 1; k < _L; k++){

      //Compute l
      d = std::abs(double(j - k));
      if(_L % 2 == 0) l = l_max - std::abs(d - 1.0*l_max);  // 𝖫 𝒆𝒗𝒆𝒏
      else{  // 𝖫 𝒐𝒅𝒅

        if(d < l_max + 1) l = d;
        else if(d == l_max + 1) l = d - 1;
        else if(d > l_max + 1) l = (l_max + 1) - (d - l_max);
        else{

          std::cerr << " ##IndexError: something went wrong in selecting the correct Jastrow interaction parameter." << std::endl;
          std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]." << std::endl;
          std::abort();

        }

      }

      //Adds the correlation
      log_psi += _alpha[l - 1] * double(real_config.at(0, j) * real_config.at(0, k));

    }

  }

  return this -> phi() + log_psi;

}


cx_double LRHJas :: Phi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhi(real_config, shadow_config));

}


cx_double LRHJas :: logPhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                          const Mat <int>& shadow_config) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)] at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the old 𝓇ℯ𝒶𝑙 configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the 𝐔𝐩𝐝𝐚𝐭𝐞_𝐓𝐡𝐞𝐭𝐚() function
  //  defined below in the 𝐑𝐁𝐌 class.
  //  Note that the ratio between the two evaluated wave function, which is the
  //  quantity related to the acceptance kernel of the Metropolis algorithm,
  //  is recovered by taking the exponential of the output of this function.
  //
  //  N̲O̲T̲E̲: once again we emphasize that in the specific case of the Jastrow
  //        𝒜𝓃𝓈𝒶𝓉𝓏 the quantities calculated with the functions inherent to
  //        Φ(𝒗,𝒉,𝛂) correspond to those calculated in the functions related
  //        to the Metropolis algorithm, since we have never introduced any
  //        auxiliary variable.
  //  N̲O̲T̲E̲: the 𝒔𝒉𝒂𝒅𝒐𝒘_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏,
  //        which does not depend upon any 𝓈ℎ𝒶𝒹ℴ𝓌 variables.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled 𝓇ℯ𝒶𝑙 configuration |𝒗ⁿᵉʷ⟩
  if(flipped_real_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    int l_max = _alpha.n_elem;  // The maximum distance in PBCs, i.e. ⌊𝖫 / 𝟤⌋
    int l = 0;  // The correct index of pairwise interaction
    double d = 0.0;
    cx_double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms: Σⱼₖ η_𝑙 • (𝑣𝒿ⁿᵉʷ•𝑣𝓀ⁿᵉʷ - 𝑣𝒿ᵒˡᵈ•𝑣𝓀ᵒˡᵈ) in P̲B̲C̲s̲
    for(int j = 0; j < _L - 1; j++){

      for(int k = j + 1; k < _L; k++){

        //Compute l
        d = std::abs(double(j - k));
        if(_L % 2 == 0) l = l_max - std::abs(d - 1.0*l_max);  // 𝖫 𝒆𝒗𝒆𝒏
        else{  // 𝖫 𝒐𝒅𝒅

          if(d < l_max + 1) l = d;
          else if(d == l_max + 1) l = d - 1;
          else if(d > l_max + 1) l = (l_max + 1) - (d - l_max);
          else{

            std::cerr << " ##IndexError: something went wrong in selecting the correct Jastrow interaction parameter." << std::endl;
            std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]." << std::endl;
            std::abort();

          }

        }

        //Adds the correlation
        log_vv += _alpha[l - 1] * double(new_real_config.at(0, j) * new_real_config.at(0, k) - real_config.at(0, j) * real_config.at(0, k));

      }

    }

    return log_vv;

  }

}


cx_double LRHJas :: PhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                       const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_config));

}


cx_double LRHJas :: logPsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return this -> logPhi(real_config, shadow_ket);

}


cx_double LRHJas :: PsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return this -> Phi(real_config, shadow_ket);

}


cx_double LRHJas :: logPsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                          const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                          const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                          std::string option) const {

  return this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_ket);

}


cx_double LRHJas :: PsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                       const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                       const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                       std::string option) const {

  return this -> PhiNew_over_PhiOld(real_config, flipped_real_site, shadow_ket);

}


double LRHJas :: PMetroNew_over_PMetroOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                          const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                          const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                          std::string option) const {

  return std::norm(this -> PsiNew_over_PsiOld(real_config, flipped_real_site,
                                              shadow_ket, flipped_ket_site,
                                              shadow_bra, flipped_bra_site));

}


void LRHJas :: LocalOperators(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameter
  //  𝛈 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the long-range homogeneous Jastrow 𝒜𝓃𝓈𝒶𝓉𝓏 the local operators
  //  𝓞(𝒗,𝒉) are
  //
  //        • η_𝑙 ←→ 𝓞(𝒗,𝒉) = 𝓞(𝒗) = Σⱼₖ 𝑣𝒿•𝑣𝓀
  //
  //  and represent the sum of all the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 correlations at distance 𝑙.
  //  This operator is necessary to compute the Quantum Geometric Tensor
  //  and the Gradient during the stochastic optimization procedure.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉)." << std::endl;
    std::abort();

  }

  //Function variables
  int l_max = _alpha.n_elem;  // The maximum distance in PBCs, i.e. ⌊𝖫 / 𝟤⌋
  int l = 0;  // The correct index of pairwise interaction
  double d = 0.0;

  //Computes the local operators assiociated to each parameter η_𝑙 in PBCs
  _LocalOperators.zeros();
  for(int j = 0; j < _L - 1; j++){

    for(int k = j + 1; k < _L; k++){

      //Compute l
      d = std::abs(double(j - k));
      if(_L % 2 == 0) l = l_max - std::abs(d - 1.0*l_max);  // 𝖫 𝒆𝒗𝒆𝒏
      else{  // 𝖫 𝒐𝒅𝒅

        if(d < l_max + 1) l = d;
        else if(d == l_max + 1) l = d - 1;
        else if(d > l_max + 1) l = (l_max + 1) - (d - l_max);
        else{

          std::cerr << " ##IndexError: something went wrong in selecting the correct Jastrow interaction parameter." << std::endl;
          std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]." << std::endl;
          std::abort();

        }

      }

      //Adds the correlation
      _LocalOperators.at(l - 1, 0) += double(real_config.at(0, j) * real_config.at(0, k));
      _LocalOperators.at(l - 1, 1) += double(real_config.at(0, j) * real_config.at(0, k));

    }

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/*****************************************  𝐉𝐀𝐒𝐓𝐑𝐎𝐖 𝐍𝐄𝐔𝐑𝐀𝐋 𝐍𝐄𝐓𝐖𝐎𝐑𝐊 𝐐𝐔𝐀𝐍𝐓𝐔𝐌 𝐒𝐓𝐀𝐓𝐄  ****************************************/
/*******************************************************************************************************************************/
JasNQS :: JasNQS(int n_visible, bool phi_option, bool imaginary_part_option, int rank)
        : WaveFunction(n_visible, phi_option, imaginary_part_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the Jastrow neural network quantum state variational parameters
  //  𝓥 = {𝜙, 𝕎} = {𝜙, 𝛂} to some small random numbers.
  //
  //  In this case we have 𝖫 • 𝖫 pairs (i.e., Jastrow) interactions ωⱼₖ
  //  which depend on the lattice site and represent a fully connected architecture
  //  among the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ degrees of freedom, in which we also allow the self-interaction
  //  between them, i.e. ωⱼⱼ ≠ 𝟢 in general.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝖫 • 𝖫 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ pairs interaction strength ωⱼₖ
  //
  //  organized sequentially in the parameter vector data-member.
  //  Note that being 𝕎 = [ωⱼₖ] a matrix, we 'unrolled' it row by row saving
  //  it in _𝐚𝐥𝐩𝐡𝐚 as a vector of 𝖫 • 𝖫 elements.
  //  We remember that the 𝒿-th row of 𝕎 represents the list of the interactions
  //  strength between the 𝒿-th 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variable and each of the 𝖫 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons.
  /*########################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a Jastrow neural network quantum state with randomly initialized variational parameters 𝓥 = {𝜙, 𝕎}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;
  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Neural Network";
  _alpha.set_size(_L * _L);
  _LocalOperators.zeros(_L * _L, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  for(int p = 0; p < _alpha.n_elem; p++){

    _alpha[p].real(_rnd.Gauss(0.0, 0.01));
    if(_if_ZERO_IMAGINARY_PART) _alpha[p].imag(0.0);
    else _alpha[p].imag(_rnd.Gauss(0.0, 0.01));

  }

  //Ends construction
  if(rank == 0){

    std::cout << " Jastrow neural network quantum state correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl << std::endl;

  }

}


JasNQS :: JasNQS(std::string file_wf, bool phi_option, int rank)
        : WaveFunction(0, phi_option, 0) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the Jastrow neural network quantum state variational parameters
  //  𝓥 = {𝜙, 𝕎} = {𝜙, 𝛂} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a Jastrow neural network quantum state from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the Jastrow neural network quantum state variational parameters 𝓥 = {𝜙, 𝕎} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Neural Network";
  int density = 0;
  input_wf >> _L;
  input_wf >> density;
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the Jastrow neural network quantum state." << std::endl;
    std::abort();

  }
  _alpha.set_size(_L * _L);
  _LocalOperators.zeros(_L * _L, 2);  //N̲O̲T̲E̲: 𝕆_𝜙 = 𝟙, so we do not save it in memory
  for(int p = 0; p < _alpha.n_elem; p++) input_wf >> _alpha[p];

  //Ends construction
  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Jastrow neural network quantum state correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


cx_double JasNQS :: w_jk(int j, int k) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || k >= _L || j < 0 || k < 0){

    std::cerr << " ##IndexError: failed to access the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ interaction strength matrix 𝕎." << std::endl;
    std::cerr << "   Element ωⱼₖ with j = " << j << " and k = " << k << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[j * _L + k];

}


void JasNQS :: print_W() const {  //Useful for debugging

  std::cout << "\n=========================================================" << std::endl;
  std::cout << "Jastrow NQS 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ interaction strength matrix 𝕎" << std::endl;
  std::cout << "=========================================================" << std::endl;
  for(int j = 0; j < _L; j++){

    for(int k = 0; k < _L; k++){

      std::cout << _alpha[j * _L + k].real();
      if(_alpha[j * _L + k].imag() >= 0) std::cout << " + i" << _alpha[j * _L + k].imag() << "  ";
      else std::cout << " - i" << -1.0 * _alpha[j * _L + k].imag() << "  ";

    }
    std::cout << std::endl;

  }

}


cx_double JasNQS :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*#########################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)] with
  //
  //        Ψ(𝒗,𝜙,𝕎) = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σₖ 𝕆ₖ(𝒗,𝒉)αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅(Σ𝒿 Σ𝓀 𝑣𝒿 • ω𝒿𝓀 • 𝑣𝓀).
  //
  //  𝒿 = 𝟢, 𝟣, 𝟤, …, 𝖫-𝟣 and 𝓀 = 𝟢, 𝟣, 𝟤, …, 𝖫-𝟣.
  //  Obviously, this 𝒜𝓃𝓈𝒶𝓉𝓏 is not of the 𝓈ℎ𝒶𝒹ℴ𝓌 type,
  //  and no auxiliary variables are introduced here.
  /*#########################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝜙,𝛈)]." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_psi = 0.0;  //Storage variable

  for(int j = 0; j < _L; j++)
    for(int k = 0; k < _L; k++) log_psi += _alpha[j * _L + k] * double(visible_config.at(0, j) * visible_config.at(0, k));

  return this -> phi() + log_psi;

}


cx_double JasNQS :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


cx_double JasNQS :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                          const Mat <int>& hidden_config) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)] at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the old 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration; in fact the
  //  second argument of the function represents the list of the
  //  site to be flipped, formatted as described in the 𝐔𝐩𝐝𝐚𝐭𝐞_𝐓𝐡𝐞𝐭𝐚() function
  //  defined below in the 𝐑𝐁𝐌 class.
  //  Note that the ratio between the two evaluated wave function, which is the
  //  quantity related to the acceptance kernel of the Metropolis algorithm,
  //  is recovered by taking the exponential of the output of this function.
  //
  //  N̲O̲T̲E̲: once again we emphasize that in the specific case of the Jastrow
  //        neural network quantum state the quantities calculated with the
  //        functions inherent to Φ(𝒗,𝒉,𝛂) correspond to those calculated in the
  //        functions related to the Metropolis algorithm, since we have never
  //        introduced any auxiliary variable.
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the Jastrow NQS,
  //        which does not depend upon any 𝒽𝒾𝒹𝒹ℯ𝓃 variables.
  /*##############################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_visible_config = generate_config(visible_config, flipped_visible_site);  // |𝒗ⁿᵉʷ⟩
    cx_double log_vv = 0.0;  //Storage variable for the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ terms

    //Computes the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ terms: Σ𝒿 Σ𝓀 ω𝒿𝓀 (𝑣𝒿ⁿᵉʷ • 𝑣𝓀ⁿᵉʷ - 𝑣𝒿ᵒˡᵈ • 𝑣𝓀ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      for(int k = 0; k < _L; k++) log_vv += _alpha[j * _L + k] * double(new_visible_config.at(0, j) * new_visible_config.at(0, k) - visible_config.at(0, j) * visible_config.at(0, k));

    return log_vv;

  }

}


cx_double JasNQS :: PhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                       const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_config));

}


cx_double JasNQS :: logPsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> logPhi(visible_config, hidden_ket);

}


cx_double JasNQS :: PsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> Phi(visible_config, hidden_ket);

}


cx_double JasNQS :: logPsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                          const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                          const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                          std::string option) const {

  return this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


cx_double JasNQS :: PsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                       const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                       const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                       std::string option) const {

  return this -> PhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


double JasNQS :: PMetroNew_over_PMetroOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                          const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                          const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                          std::string option) const {

  return std::norm(this -> PsiNew_over_PsiOld(visible_config, flipped_visible_site,
                                              hidden_ket, flipped_ket_site,
                                              hidden_bra, flipped_bra_site));

}


void JasNQS :: LocalOperators(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) {

  /*###############################################################################*/
  //  Calculates the local operators associated to the variational parameter
  //  ωⱼₖ on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the Jastrow neural network quantum state the local operators
  //  𝓞(𝒗,𝒉) are
  //
  //        • ωⱼₖ ←→ 𝓞(𝒗,𝒉) = 𝓞(𝒗) = 𝑣ⱼ • 𝑣ₖ
  //
  //  and represent the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝓋𝒾𝓈𝒾𝒷𝑙ℯ correlations.
  //  This operator is necessary to compute the Quantum Geometric Tensor
  //  and the Gradient during the stochastic optimization procedure.
  /*###############################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(visible_config.n_rows != 1 || visible_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉)." << std::endl;
    std::abort();

  }

  //Computes the local operators assiociated to each parameter ωⱼₖ
  for(int j = 0; j < _L; j++){

    for(int k = 0; k < _L; k++){

      _LocalOperators.at(j * _L + k, 0) = double(visible_config.at(0, j) * visible_config.at(0, k));
      _LocalOperators.at(j * _L + k, 1) = _LocalOperators.at(j * _L + k, 0);

    }

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  𝐑𝐄𝐒𝐓𝐑𝐈𝐂𝐓𝐄𝐃 𝐁𝐎𝐋𝐓𝐙𝐌𝐀𝐍𝐍 𝐌𝐀𝐂𝐇𝐈𝐍𝐄  *********************************************/
/*******************************************************************************************************************************/
RBM :: RBM(int n_visible, int density, bool phi_option, bool imaginary_part_option, int rank)
     : WaveFunction(n_visible, phi_option, imaginary_part_option), _M(density * n_visible), _ln2(std::log(2.0)) {

  /*################################################################################*/
  //  Random-based constructor.
  //  Initializes the RBM variational parameters 𝛂 = {𝐚,𝐛,𝕎} to
  //  some small random numbers [G.Hinton, 2010].
  //  We have
  //
  //        𝖫 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neuron bias 𝐚 = {𝑎𝟢, 𝑎𝟣, …, 𝑎𝖫};
  //        𝖬 𝒽𝒾𝒹𝒹ℯ𝓃 neuron bias 𝐛 = {𝑏𝟢, 𝑏𝟣, …, 𝑏𝖬};
  //        𝖫 • 𝖬 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 neuron interaction strength weights 𝕎 = [𝕎]𝒿𝓀
  //
  //  organized sequentially in the parameter vector data-member.
  //  Note that being 𝕎 a matrix, we 'unrolled' it row by row saving
  //  it in _𝐚𝐥𝐩𝐡𝐚 as a vector of 𝖫 • 𝖬 elements.
  //  We remember that the 𝒿-th row of 𝕎 represents the list of the interactions
  //  strength between the 𝒿-th 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variable and each of the 𝖬 𝒽𝒾𝒹𝒹ℯ𝓃 neurons.
  /*################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a RBM wave function with randomly initialized variational parameters 𝛂 = {𝐚,𝐛,𝕎}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  _type = "Neural Network";
  _alpha.set_size(_L + _M + _L * _M);
  _LocalOperators.zeros(_L + _M + _L * _M, 2);
  _Theta.set_size(_M);

  //𝒱𝒾𝓈𝒾𝒷𝑙ℯ bias
  for(int j = 0; j < _L; j++) _alpha[j] = 0.0;  // αⱼ ≡ 𝑎ⱼ

  //ℋ𝒾𝒹𝒹ℯ𝓃 bias
  for(int k = _L; k < _L + _M; k++) _alpha[k] = 0.0;  // αₖ ≡ 𝑏ₖ

  //𝒱𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interaction weights
  for(int jk = _L + _M; jk < _alpha.n_elem; jk++){

    _alpha[jk].real(_rnd.Gauss(0.0, 0.1));  // αⱼᴿ ≡ [𝕎]ᴿ𝒿𝓀
    if(_if_ZERO_IMAGINARY_PART) _alpha[jk].imag(0.0);
    else _alpha[jk].imag(_rnd.Gauss(0.0, 0.1));  // αⱼᴵ ≡ [𝕎]ᴵ𝒿𝓀

  }

  //Ends construction
  if(rank == 0){

    std::cout << " RBM 𝒜𝓃𝓈𝒶𝓉𝓏 correctly initialized with random weights." << std::endl;
    std::cout << " Number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons = " << _L << "." << std::endl;
    std::cout << " Number of 𝒽𝒾𝒹𝒹ℯ𝓃 neurons = " << _M << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
    std::cout << " Density of the 𝒽𝒾𝒹𝒹ℯ𝓃 neurons = " << this -> shadow_density() << "." << std::endl << std::endl;

  }

}


RBM :: RBM(std::string file_wf, bool phi_option, int rank)
     : WaveFunction(0, phi_option, 0), _ln2(std::log(2.0)) {

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
  if(rank == 0) std::cout << "#Create a RBM wave function from an existing quantum state." << std::endl;

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
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  int density = 0;
  input_wf >> _L;
  input_wf >> density;
  _M = _L * density;
  if(!input_wf.good() || _L < 0 || _M < 0){

    std::cerr << " ##FileError: invalid construction of the RBM 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::cerr << "   Failed to initialize the RBM variational parameters 𝛂 = { 𝐚,𝐛,𝕎 } from file." << std::endl;
    std::abort();

  }
  _type = "Neural Network";
  _alpha.set_size(_L + _M + _L * _M);
  _LocalOperators.zeros(_L + _M + _L * _M, 2);  // 𝕆ₖ(𝒗,𝒉) ≡ 𝕆ₖ(𝒗,𝒉ˈ) ≡ 𝕆ₖ(𝒗)
  _Theta.set_size(_M);
  if(_if_PHI) input_wf >> _phi;
  for(int p = 0; p <_alpha.n_elem; p++) input_wf >> _alpha[p];

  //Ends construction
  if(input_wf.good()){

    if(rank == 0){

      std::cout << " RBM 𝒜𝓃𝓈𝒶𝓉𝓏 correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons = " << _L << "." << std::endl;
      std::cout << " Number of 𝒽𝒾𝒹𝒹ℯ𝓃 neurons = " << _M << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
      std::cout << " Density of the 𝒽𝒾𝒹𝒹ℯ𝓃 neurons = " << this -> shadow_density() << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


cx_double RBM :: a_j(int j) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || j < 0){

    std::cerr << " ##IndexError: failed to access the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ bias vector 𝐚 = {𝑎𝟢, 𝑎𝟣, …, 𝑎𝖫}." << std::endl;
    std::cerr << "   Element 𝑎ⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[j];

}


cx_double RBM :: b_k(int k) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(k >= _M || k < 0){

    std::cerr << " ##IndexError: failed to access the 𝒽𝒾𝒹𝒹ℯ𝓃 bias vector 𝐛 = {𝑏𝟢, 𝑏𝟣, …, 𝑏𝖫}." << std::endl;
    std::cerr << "   Element 𝑏ₖ with k = " << k << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[_L + k];

}


cx_double RBM :: W_jk(int j, int k) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || k >= _M || j < 0 || k < 0){

    std::cerr << " ##IndexError: failed to access the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interaction strength matrix 𝕎." << std::endl;
    std::cerr << "   Element 𝕎ⱼₖ with j = " << j << " and k = " << k << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[_L + _M + j * _M + k];

}


cx_double RBM :: Theta_k(int k) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(k >= _M || k < 0){

    std::cerr << " ##IndexError: failed to access the effective angles 𝛳(𝒗,𝛂)." << std::endl;
    std::cerr << "   Element 𝛳ₖ with k = " << k << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _Theta[k];

}


void RBM :: print_a() const {  //Useful for debugging

  std::cout << "\n=========================================" << std::endl;
  std::cout << "RBM 𝓋𝒾𝓈𝒾𝒷𝑙ℯ bias vector 𝐚 = {𝑎𝟢, 𝑎𝟣, …, 𝑎𝖫}" << std::endl;
  std::cout << "=========================================" << std::endl;
  for(int j = 0; j < _L; j++){

    std::cout << _alpha[j].real();
    if(_alpha[j].imag() >= 0) std::cout << " + i" << _alpha[j].imag() << "  " << std::endl;
    else std::cout << " - i" << -1.0 * _alpha[j].imag() << "  " << std::endl;

  }

}


void RBM :: print_b() const {  //Useful for debugging

  std::cout << "\n=========================================" << std::endl;
  std::cout << "RBM 𝒽𝒾𝒹𝒹ℯ𝓃 bias vector 𝐛 = {𝑏𝟢, 𝑏𝟣, …, 𝑏𝖫}" << std::endl;
  std::cout << "=========================================" << std::endl;
  for(int k = 0; k < _M; k++){

    std::cout << _alpha[_L + k].real();
    if(_alpha[_L + k].imag() >= 0) std::cout << " + i" << _alpha[_L + k].imag() << "  " << std::endl;
    else std::cout << " - i" << -1.0 * _alpha[_L + k].imag() << "  " << std::endl;

  }

}


void RBM :: print_W() const {  //Useful for debugging

  std::cout << "\n=========================================================" << std::endl;
  std::cout << "RBM 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interaction strength matrix 𝕎 = [𝕎]𝒿𝓀" << std::endl;
  std::cout << "=========================================================" << std::endl;
  for(int j = 0; j < _L; j++){

    for(int k = 0; k < _M; k++){

      std::cout << _alpha[_L + _M + j * _M + k].real();
      if(_alpha[_L + _M + j * _M + k].imag() >= 0) std::cout << " + i" << _alpha[_L + _M + j * _M + k].imag() << "  ";
      else std::cout << " - i" << -1.0 * _alpha[_L + _M + j * _M + k].imag() << "  ";

    }
    std::cout << std::endl;

  }

}


void RBM :: print_Theta() const {  //Useful for debugging

  std::cout << "\n==========================" << std::endl;
  std::cout << "RBM effective angles 𝛳(𝒗,𝛂)" << std::endl;
  std::cout << "==========================" << std::endl;
  for(int k = 0; k < _Theta.n_elem; k++){

    std::cout << _Theta[k].real();
    if(_Theta[k].imag() >= 0) std::cout << " + i" << _Theta[k].imag() << std::endl;
    else std::cout << " + i" << -1.0 * _Theta[k].imag() << std::endl;

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
  //        𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓍) ~ 𝓍 - 𝑙𝑜𝑔𝟤.
  /*###########################################################*/

  if(x < 6.0) return std::log(std::cosh(x));
  else return x - _ln2;

}


cx_double RBM :: lncosh(cx_double z) const {

  /*##########################################################################################*/
  //  Computes the complex natural logarithm of
  //  the hyperbolic cosine of a generic complex
  //  number 𝓏 ϵ ℂ
  //
  //        𝓏 = ℜe{𝓏} + i•ℑm{𝓏} = 𝓊 + i𝓋
  //
  //  We can manipulate the complex logarithm in
  //  the following way:
  //
  //        𝑙𝑜𝑔(𝑐𝑜𝑠ℎ𝓏) = 𝑙𝑜𝑔[𝟣/𝟤 • (ℯ𝓍𝓅(𝓏) + ℯ𝓍𝓅(-𝓏))]
  //                  = 𝑙𝑜𝑔[𝟣/𝟤 • (ℯ𝓍𝓅(𝓊)•ℯ𝓍𝓅(i𝓋) + ℯ𝓍𝓅(-𝓊)•ℯ𝓍𝓅(i𝓋))]
  //                  = 𝑙𝑜𝑔{𝟣/𝟤 • [ℯ𝓍𝓅(𝓊)•(cos(𝓋) + i•sin(𝓋)) + ℯ𝓍𝓅(-𝓊)(cos(𝓋) - i•sin(𝓋))]}
  //                  = 𝑙𝑜𝑔{𝟣/𝟤 • [cos(𝓋)•(ℯ𝓍𝓅(𝓊) + ℯ𝓍𝓅(-𝓊)) + i•sin(𝓋)•(ℯ𝓍𝓅(𝓊) - ℯ𝓍𝓅(-𝓊))]}
  //                  = 𝑙𝑜𝑔{cosh(𝓊)•cos(𝓋) + i•sinh(𝓊)•sin(𝓋)}
  //                  = 𝑙𝑜𝑔{cosh(𝓊)•[cos(𝓋) + i•tanh(𝓊)•sin(𝓋)]}
  //                  = lncosh(ℜe{𝓏}) + 𝑙𝑜𝑔{cos(ℑm{𝓏}) + i•tanh(ℜe{𝓏})•sin(ℑm{𝓏})}
  //
  //  where the first term in the final line is calculated through the
  //  function lncosh(x) defined above.
  /*##########################################################################################*/

  double xr = z.real();
  double xi = z.imag();

  cx_double result = this -> lncosh(xr);
  result += std::log(cx_double (std::cos(xi), std::tanh(xr) * std::sin(xi)));
  return result;

}


void RBM :: Init_Theta(const Mat <int>& visible_config) {

  /*#######################################################################*/
  //  𝛳(𝒗,𝛂) Initialization  -->  𝛳ₖ(𝒗,𝛂) = 𝑏ₖ + Σₘ [𝕎]ₖₘ•σₘᶻ
  //  Initializes the effective angles that appear thanks to the fact that
  //  in the particular functional form chosen for the this wave function
  //  the 𝒽𝒾𝒹𝒹ℯ𝓃 degrees of freedom are traced out exactly.
  //
  //  N̲O̲T̲E̲: this analytical integration changes the generic form
  //        introduced above for the 𝒜𝓃𝓈𝒶𝓉𝓏 and consequently will also
  //        change the optimization algorithm (𝐬𝐚𝐦𝐩𝐥𝐞𝐫.𝐜𝐩𝐩).
  //        In fact here the local operators associated with variational
  //        parameters become complex and no longer real, just as the angles
  //        𝛳(𝒗,𝛂).
  //
  //  The angles depend on the parameters {𝐛,𝕎} and
  //  on the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ variables (i.e. the quantum spin) that define the
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
  //        |𝒗𝟣 … 𝒗𝖫⟩ =      :  :      …       σᶻ     \     ℳ
  //                    | :  :  :      …       σᶻ    /   |
  //                    | :  :  :      …       σᶻ   /    |
  //                    | σᶻ σᶻ σᶻ     …       σᶻ  /     |
  //                                                     v
  //
  //  for a total of 𝖫 = 𝒩•ℳ quantum degrees of freedom.
  /*#######################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _L){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons." << std::endl;
    std::cerr << "   Failed to initialize the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
    std::abort();

  }

  //Computes the effective angles
  for(int k = 0; k < _M; k++){

    _Theta[k] = _alpha[_L + k];
    for(int m_row = 0; m_row < visible_config.n_rows; m_row++){

      for(int m_col = 0; m_col < visible_config.n_cols; m_col++)
        _Theta[k] += _alpha[_L + _M + (m_row * visible_config.n_cols + m_col) * _M + k] * double(visible_config.at(m_row, m_col));

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
  if(visible_config.n_elem != _L){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons." << std::endl;
    std::cerr << "   Failed to update the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0) return;
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
    cx_double delta_theta;  //Storage variable

    //Updates 𝛳(𝒗,𝛂)
    for(int k = 0; k < _M; k++){

      delta_theta = 0.0;
      for(int m_row = 0; m_row < flipped_visible_site.n_rows; m_row++){

        if(flipped_visible_site.n_cols == 1)  //𝐝 = 𝟏
          delta_theta += double(visible_config.at(0, flipped_visible_site.at(m_row, 0))) * _alpha[_L + _M + flipped_visible_site.at(m_row, 0) * _M + k];
        else if(flipped_visible_site.n_cols == 2)  //𝐝 = 𝟐
          delta_theta += double(visible_config.at(flipped_visible_site.at(m_row, 0), flipped_visible_site.at(m_row, 1))) *
                                _alpha[_L + _M + (flipped_visible_site.at(m_row, 0) * visible_config.n_cols + flipped_visible_site.at(m_row, 1)) * _M + k];
        else{

          std::cerr << " ##SizeError: bad construction of the new quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
          std::cerr << "   Failed to updatet the effective angles vector 𝛳(𝒗,𝛂)." << std::endl;
          std::abort();

        }

      }
      _Theta[k] -= 2.0 * delta_theta;  //Using the Look-up table for fast computation

    }

  }

}


cx_double RBM :: logPhi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  /*###################################################################*/
  //  Since we have managed to integrate exactly the 𝒽𝒾𝒹𝒹ℯ𝓃 degrees
  //  of freedom for this 𝒜𝓃𝓈𝒶𝓉𝓏, we can here interpret Φ(𝒗,𝒉,𝛂) as
  //  the total wave function, which is defined as (we set 𝜙 = 𝟢):
  //
  //        Ψ(𝒗,𝛂) = Σₕ ℯ𝓍𝓅(Σⱼ𝑎ⱼσⱼᶻ + Σₖ𝑏ₖ𝒽ₖ + Σⱼ[𝕎]ⱼₖ𝒽ⱼσₖᶻ)
  //               = ℯ𝓍𝓅(Σⱼ𝑎ⱼσⱼᶻ) • 𝚷ₖ 2𝑐𝑜𝑠ℎ(𝛳ₖ)
  //
  //  where the effective angles are defined above.
  //
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the RBM 𝒜𝓃𝓈𝒶𝓉𝓏,
  //        which does not depend explicitly on the 𝒽𝒾𝒹𝒹ℯ𝓃 variables.
  /*###################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _L){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗,𝛂)]." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_vv(0.0, 0.0);  //Storage variable for the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ terms
  cx_double log_theta(0.0, 0.0);  //Storage variable for the theta angle terms

  //Computes the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons terms: Σⱼ 𝑎ⱼσⱼᶻ
  for(int j_row = 0; j_row < visible_config.n_rows; j_row++){

    for(int j_col = 0; j_col <visible_config.n_cols; j_col++)
      log_vv += _alpha[j_row * visible_config.n_cols + j_col] * double(visible_config.at(j_row, j_col));

  }

  //Computes the theta angles contribution: Σₖ 𝑙𝑜𝑔(𝑐𝑜𝑠ℎ(𝛳ₖ))
  for(int k = 0; k < _M; k++){

    log_theta = _alpha[_L + k];
    for(int j_row = 0; j_row < visible_config.n_rows; j_row++){

      for(int j_col = 0; j_col < visible_config.n_cols; j_col++)
        log_theta += _alpha[_L + _M + (j_row * visible_config.n_cols + j_col) * _M + k] * double(visible_config.at(j_row, j_col));

    }
    log_vv += this -> lncosh(log_theta);

  }

  return this -> phi() + log_vv + _M * _ln2;

}


cx_double RBM :: Phi(const Mat <int>& visible_config, const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhi(visible_config, hidden_config));

}


cx_double RBM :: logPhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                       const Mat <int>& hidden_config) const {

  /*#############################################################################*/
  //  Computes 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)] at fixed variational parameters.
  //  The new proposed configuration is a configuration with a certain number of
  //  flipped spins wrt the current 𝓋𝒾𝓈𝒾𝒷𝑙ℯ configuration; in fact the
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
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐜𝐨𝐧𝐟𝐢𝐠 argument is useless for the RBM 𝒜𝓃𝓈𝒶𝓉𝓏,
  //        which does not depend explicitly on the 𝒽𝒾𝒹𝒹ℯ𝓃 variables.
  /*#############################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem != _L){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝛂) / Ψ(𝒗ᵒˡᵈ,𝛂)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_visible_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    //𝐝 = 𝟏
    if(visible_config.n_rows == 1 && flipped_visible_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
      std::abort();

    }
    //𝐝 = 𝟐
    if(visible_config.n_rows != 1 && flipped_visible_site.n_cols != 2){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    cx_double log_vv(0.0, 0.0);  //Storage variable for the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ terms
    cx_double log_theta(0.0, 0.0);  //Storage variable for the old theta angles
    cx_double log_theta_prime(0.0, 0.0);  //Storage variable for the new theta angles

    //Change due to the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ layer
    for(int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++){

      if(flipped_visible_site.n_cols == 1)  //𝐝 = 𝟏
        log_vv -= _alpha[flipped_visible_site.at(j_row, 0)] * double(visible_config.at(0, flipped_visible_site.at(j_row, 0)));
      else if(flipped_visible_site.n_cols == 2){  //𝐝 = 𝟐

        log_vv -= _alpha[flipped_visible_site.at(j_row, 0) * visible_config.n_cols + flipped_visible_site.at(j_row, 1)] *
                  double(visible_config.at(flipped_visible_site.at(j_row, 0), flipped_visible_site.at(j_row, 1)));
      }
      else{

        std::cerr << " ##SizeError: bad construction of the new quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
        std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ) / Ψ(𝒗ᵒˡᵈ)]." << std::endl;
        std::abort();

      }

    }
    log_vv *= 2.0;

    //Change due to the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interactions
    for(int k = 0; k < _M; k++){

      log_theta = _Theta[k];  //speed-up the calculation with the Look-up table
      log_theta_prime = log_theta;
      for(int j_row = 0; j_row < flipped_visible_site.n_rows; j_row++){

        if(flipped_visible_site.n_cols == 1)  //𝐝 = 𝟏
          log_theta_prime -= 2.0 * double(visible_config.at(0, flipped_visible_site.at(j_row, 0))) * _alpha[_L + _M + flipped_visible_site.at(j_row, 0) * _M + k];
        else if(flipped_visible_site.n_cols == 2){  //𝐝 = 𝟐

          log_theta_prime -= 2.0 * double(visible_config.at(flipped_visible_site.at(j_row, 0), flipped_visible_site.at(j_row, 1))) *
                             _alpha[_L + _M + (flipped_visible_site.at(j_row, 0) * visible_config.n_cols + flipped_visible_site.at(j_row, 1)) * _M + k];

        }
        else{

          std::cerr << " ##SizeError: bad construction of the new quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
          std::cerr << "   Failed to compute 𝑙𝑜𝑔[Ψ(𝒗ⁿᵉʷ,𝛂) / Ψ(𝒗ᵒˡᵈ,𝛂)]." << std::endl;
          std::abort();

        }

      }
      log_vv += this -> lncosh(log_theta_prime) - this -> lncosh(log_theta);

    }

    return log_vv;

  }

}


cx_double RBM :: PhiNew_over_PhiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                    const Mat <int>& hidden_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_config));

}


cx_double RBM :: logPsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> logPhi(visible_config, hidden_ket);

}


cx_double RBM :: PsiMetro(const Mat <int>& visible_config, const Mat <int>& hidden_ket, const Mat <int>& hidden_bra) const {

  return this -> Phi(visible_config, hidden_ket);

}


cx_double RBM :: logPsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
                                       const Mat <int>& hidden_ket, const Mat <int>& flipped_ket_site,
                                       const Mat <int>& hidden_bra, const Mat <int>& flipped_bra_site,
                                       std::string option) const {

  return this -> logPhiNew_over_PhiOld(visible_config, flipped_visible_site, hidden_ket);

}


cx_double RBM :: PsiNew_over_PsiOld(const Mat <int>& visible_config, const Mat <int>& flipped_visible_site,
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
  //  In the case of the RBM 𝒜𝓃𝓈𝒶𝓉𝓏 the local parameters are 𝓞(𝒗,𝒉) = 𝓞(𝒗)
  //
  //        • 𝑎𝒿 ←→ 𝓞(𝒗) = σ𝒿ᶻ
  //        • 𝑏𝓀 ←→ 𝓞(𝒗) = 𝑡𝑎𝑛ℎ(𝛳𝓀(𝒗))
  //        • [𝕎]𝒿𝓀 ←→ 𝓞(𝒗) = σ𝒿ᶻ•𝑡𝑎𝑛ℎ(𝛳𝓀(𝒗))
  //
  //  and are 𝐜𝐨𝐦𝐩𝐥𝐞𝐱 number once evaluated!
  //  These operators are necessary to compute the Quantum
  //  Geometric Tensor and the Gradient during the
  //  stochastic optimization procedure.
  //
  //  N̲O̲T̲E̲: the 𝒉𝒊𝒅𝒅𝒆𝒏_𝐤𝐞𝐭 and 𝒉𝒊𝒅𝒅𝒆𝒏_𝐛𝐫𝐚 arguments are useless for the
  //        RBM 𝒜𝓃𝓈𝒶𝓉𝓏, which does not depend explicitly on the
  //        𝒽𝒾𝒹𝒹ℯ𝓃 variables.
  /*#######################################################################*/

  //Check on the lattice dimensionality
  if(visible_config.n_elem !=_L ){

    std::cerr << " ##SizeError: the matrix representation of the quantum configuration does not match with the number of 𝓋𝒾𝓈𝒾𝒷𝑙ℯ neurons." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗)." << std::endl;
    std::abort();

  }

  //Local operators for the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ bias 𝐚
  for(int j_row = 0; j_row < visible_config.n_rows; j_row++){

    for(int j_col = 0; j_col < visible_config.n_cols; j_col++){

      _LocalOperators.at(j_row * visible_config.n_cols + j_col, 0) = double(visible_config.at(j_row, j_col));
      _LocalOperators.at(j_row * visible_config.n_cols + j_col, 1) = _LocalOperators.at(j_row * visible_config.n_cols + j_col, 0);

    }

  }

  //Local operators for the 𝒽𝒾𝒹𝒹ℯ𝓃 bias 𝐛
  for(int k = 0; k < _M; k++){

    _LocalOperators.at(_L + k, 0) = std::tanh(_Theta[k]);
    _LocalOperators.at(_L + k, 1) = _LocalOperators.at(_L + k, 0);

  }

  //Local operators for the 𝓋𝒾𝓈𝒾𝒷𝑙ℯ-𝒽𝒾𝒹𝒹ℯ𝓃 interaction strength 𝕎
  for(int m_row = 0; m_row < visible_config.n_rows; m_row++){

    for(int m_col = 0; m_col < visible_config.n_cols; m_col++){

      for(int n = 0; n < _M; n++){

        _LocalOperators.at(_L + _M + (m_row * visible_config.n_cols + m_col) * _M + n, 0) = double(visible_config.at(m_row, m_col)) * std::tanh(_Theta[n]);
        _LocalOperators.at(_L + _M + (m_row * visible_config.n_cols + m_col) * _M + n, 1) = _LocalOperators.at(_L + _M + (m_row * visible_config.n_cols + m_col) * _M + n, 0);

      }

    }

  }

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/**********************************************  𝐁𝐀𝐄𝐑𝐈𝐒𝐖𝐘𝐋-𝐒𝐇𝐀𝐃𝐎𝐖 𝐍𝐍𝐐𝐒 in 𝗱 = 𝟏  ********************************************/
/*******************************************************************************************************************************/
BSWF :: BSWF(int n_real, bool phi_option, bool imaginary_part_option, int rank)
      : WaveFunction(n_real, phi_option, imaginary_part_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the Baeriswyl-Shadow variational parameters 𝓥 = {𝜙, η, ρ, ω} = {𝜙, 𝛂} to
  //  some small random numbers.
  //
  //  In this case we have only 𝟯 parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝟏 nearest-neighbors 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength weights η;
  //        𝟏 nearest-neighbors 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights ρ;
  //        𝟏 local 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights ω.
  //
  //  Note that in this case the number of variational parameters remains equal to 𝟯 for any system size 𝖫.
  /*########################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a 1D Baeriswyl-Shadow wave function with randomly initialized variational parameters 𝓥 = {𝜙,η,ρ,ω}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Shadow";
  _alpha.set_size(3);
  _LocalOperators.zeros(3, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  for(int p = 0; p < _alpha.n_elem; p++){

    _alpha[p].real(_rnd.Gauss(0.1, 0.001));
    if(_if_ZERO_IMAGINARY_PART) _alpha[p].imag(0.0);
    else _alpha[p].imag(_rnd.Gauss(0.0, 0.001));

  }

  if(rank == 0){

    std::cout << " Baeriswyl-Shadow NNQS correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
    std::cout << " Number of 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
    std::cout << " Density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << this -> shadow_density() << "." << std::endl;
    std::cout << " Variational parameters at initial time \t → ϕ(𝟢) = " << this -> phi() << std::endl;
    std::cout << "                                        \t → η(𝟢) = " << this -> eta() << std::endl;
    std::cout << "                                        \t → ρ(𝟢) = " << this -> rho() << std::endl;
    std::cout << "                                        \t → ω(𝟢) = " << this -> omega() << std::endl << std::endl;

  }

}


BSWF :: BSWF(std::string file_wf, bool phi_option, int rank)
      : WaveFunction(0, phi_option, 0) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the Baeriswyl-Shadow variational parameters
  //  𝓥 = {𝜙,η,ρ,ω} = {𝜙, 𝛂} from a given external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a 1D Baeriswyl-Shadow wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the Baeriswyl-Shadow NNQS variational parameters 𝓥 = {𝜙,η,ρ,ω} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  input_wf >> _L;
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the 1D Baeriswyl-Shadow 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  _type = "Shadow";
  _alpha.set_size(3);
  _LocalOperators.zeros(3, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  for(int p = 0; p < _alpha.n_elem; p++) input_wf >> _alpha[p];

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Baeriswyl-Shadow NNQS correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓇ℯ𝒶𝑙 neurons = " << _L << "." << std::endl;
      std::cout << " Number of 𝓈ℎ𝒶𝒹ℴ𝓌 neurons = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
      std::cout << " Density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << this -> shadow_density() << "." << std::endl;
      std::cout << " Variational parameters at initial time \t → ϕ(𝟢) = " << this -> phi() << std::endl;
      std::cout << "                                        \t → η(𝟢) = " << this -> eta() << std::endl;
      std::cout << "                                        \t → ρ(𝟢) = " << this -> rho() << std::endl;
      std::cout << "                                        \t → ω(𝟢) = " << this -> omega() << std::endl << std::endl;

    }

  }
  input_wf.close();

}


double BSWF :: I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*######################################################################*/
  //  Computes the value of the angle
  //
  //        ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽') = Σₖ (𝓞ₖ(𝒗,𝒉) - 𝓞ₖ(𝒗,𝐡ˈ)) • αᴵₖ
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉ˈ⟩
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  double II_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

  for(int j = 0; j < _L; j++){

      II_hh += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // (𝒽𝒿•𝒽𝒿+𝟣 - 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣) in PBCs
      II_vh += double(real_config.at(0, j) * (shadow_ket.at(0, j) - shadow_bra.at(0, j)));  // 𝓋𝒿•(𝒽𝒿 - 𝒽ˈ𝒿)

  }

  return this -> rho().imag() * II_hh + this -> omega().imag() * II_vh;

}


double BSWF :: cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::cos(this -> I_minus_I(real_config, shadow_ket, shadow_bra));

}


double BSWF :: sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::sin(this -> I_minus_I(real_config, shadow_ket, shadow_bra));

}


cx_double BSWF :: logPhi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  /*##########################################################################################################*/
  //  Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)] with
  //
  //        Φ(𝒗,𝒉,𝛂) = ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉) • αₖ)
  //
  //  Φ is that part of variational 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 that appears in the 𝐕𝐌𝐂 calculation
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
  //  N̲O̲T̲E̲: the 𝒔𝒉𝒂𝒅𝒐𝒘_𝐜𝐨𝐧𝐟𝐢𝐠 argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                Φ(𝒗,𝒉,𝛂)
  //                   or
  //                Φ(𝒗,𝒉ˈ,𝛂).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(shadow_config.n_rows != 1 || shadow_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  cx_double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  cx_double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

  for(int j = 0; j < _L; j++){

      log_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // 𝓋𝒿•𝓋𝒿+𝟣 in PBCs
      log_hh += double(shadow_config.at(0, j) * shadow_config.at(0, (j + 1) % _L));  // 𝒽𝒿•𝒽𝒿+𝟣 or 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣 in PBCs
      log_vh += double(real_config.at(0, j) * shadow_config.at(0, j));  // 𝓋𝒿•𝒽𝒿 or 𝓋𝒿•𝒽ˈ𝒿

  }

  return this -> eta() * log_vv + this -> rho() * log_hh + this -> omega() * log_vh;

}


cx_double BSWF :: Phi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhi(real_config, shadow_config));

}


cx_double BSWF :: logPhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                        const Mat <int>& shadow_config) const {

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(shadow_config.n_rows != 1 || shadow_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_real_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿 with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝒓𝒆𝒂𝒍_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_real_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_real_site.at(j_row, 0)) * shadow_config.at(0, flipped_real_site.at(j_row, 0)));

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    return -2.0 * this -> omega() * log_vh + this -> eta() * log_vv;

  }

}


cx_double BSWF :: PhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                     const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_config));

}


cx_double BSWF :: logPsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

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
  //        Ψ(𝒗,𝜙,𝛂) = ℯ𝓍𝓅(𝜙) • Σₕ ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉) • αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅( η • Σ𝒿 𝓋𝒿•𝓋𝒿+𝟣 ) •
  //                           • Σ𝒽 ℯ𝓍𝓅( ρ • Σ𝒿 (𝒽𝒿•𝒽𝒿+𝟣) + ω • Σ𝒿 (𝓋𝒿•𝒽𝒿) )
  //
  //  is the variational Baeriswyl-Shadow wave function characterized by the variational
  //  parameters {𝜙, 𝛂} = {𝜙,η,ρ,ω}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        Σ𝑣Σ𝒽Σ𝒽' 𝓆(𝒗,𝒉,𝒉ˈ) 𝒻(𝒗,𝒉,𝒉ˈ) = ⟨𝒻(𝒗,𝒉,𝒉ˈ)⟩𝓆 / ⟨𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))⟩𝓆.
  //
  //  So it is clear that the classical probability part 𝓆(𝒗,𝒉,𝒉ˈ) plays the role of
  //  square modulus of the wave function with which to sample the 𝓈ℎ𝒶𝒹ℴ𝓌 configurations |𝒗, 𝒉, 𝒉ˈ⟩
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        𝓆(𝒗,𝒉,𝒉ˈ) = ℯ𝓍𝓅(2𝜙ᴿ) • ℯ𝓍𝓅(ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ))
  //
  //  where
  //
  //        ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ) = Σₖ (𝓞ₖ(𝒗,𝒉) + 𝓞ₖ(𝒗,𝒉ˈ)) • αᴿₖ
  //
  //  and it has to be calculated on the current configuration |𝒗 𝒉 𝒉ˈ⟩.
  /*################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Function variables
  double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  cx_double log_q;

  for(int j = 0; j < _L; j++){

    log_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // Σ𝒿 𝓋𝒿•𝓋𝒿+𝟣 in PBCs
    log_hh += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L) + shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // Σ𝒿 𝒽𝒿•𝒽𝒿+𝟣 + 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣 in PBCs
    log_vh += double(real_config.at(0, j) * (shadow_ket.at(0, j) + shadow_bra.at(0, j)));  // Σ𝒿 𝓋𝒿•(𝒽𝒿 + 𝒽ˈ𝒿)

  }

  log_q.imag(0.0);
  log_q.real(2.0 * this -> phi().real() + 2.0 * this -> eta().real() * log_vv + this -> rho().real() * log_hh + this -> omega().real() * log_vh);
  return log_q;

}


cx_double BSWF :: PsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  //Function variables
  cx_double P;
  P.imag(0.0);
  P.real(std::exp(this -> logPsiMetro(real_config, shadow_ket, shadow_bra)).real());

  return P;

}


double BSWF :: logq_over_q_real(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓇ℯ𝒶𝑙 variables have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
  if(flipped_real_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ = |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 term: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 𝓋𝒿ᵒˡᵈ•(𝒽𝒿 + 𝒽ˈ𝒿) with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝒓𝒆𝒂𝒍_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_real_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_real_site.at(j_row, 0)) * (shadow_ket.at(0, flipped_real_site.at(j_row, 0)) + shadow_bra.at(0, flipped_real_site.at(j_row, 0))));

    return 2.0 * this -> eta().real() * log_vv - 2.0 * this -> omega().real() * log_vh;

  }

}


double BSWF :: q_over_q_real(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                             const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::exp(this -> logq_over_q_real(real_config, flipped_real_site, shadow_ket, shadow_bra));

}


double BSWF :: logq_over_q_ket(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site) const {

  /*#################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ket have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
  if(flipped_ket_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |𝒉ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐤𝐞𝐭_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));

    return this -> rho().real() * log_hh - 2.0 * this -> omega().real() * log_vh;

  }

}


double BSWF :: q_over_q_ket(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site) const {

  return std::exp(this -> logq_over_q_ket(real_config, shadow_ket, flipped_ket_site));

}


double BSWF :: logq_over_q_bra(const Mat <int>& real_config, const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  /*#################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]
  //  evaluated in a new proposed configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ket have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
  if(flipped_bra_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽ˈ𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐛𝐫𝐚_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    return this -> rho().real() * log_hh - 2.0 * this -> omega().real() * log_vh;

  }

}


double BSWF :: q_over_q_bra(const Mat <int>& real_config, const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_bra(real_config, shadow_bra, flipped_bra_site));

}


double BSWF :: logq_over_q_equal_site(const Mat <int>& real_config, const Mat <int>& flipped_equal_site,
                                      const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*###################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site-flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site-flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_equal_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site-flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_equal_site);  // |𝒗ⁿᵉʷ⟩
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_equal_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_equal_site);  // |𝒉ˈⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
    double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    for(int j = 0; j < _L; j++){

      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs
      log_hh += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ) in PBCs
      log_hh += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ) in PBCs

    }

    return 2.0 * this -> eta().real() * log_vv + this -> rho().real() * log_hh;

  }

}


double BSWF :: q_over_q_equal_site(const Mat <int>& real_config, const Mat <int>& flipped_equal_site,
                                   const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::exp(this -> logq_over_q_equal_site(real_config, flipped_equal_site, shadow_ket, shadow_bra));

}


double BSWF :: logq_over_q_braket(const Mat <int>& real_config,
                                  const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                  const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  /*################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_ket = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 ket terms
    double log_bra = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 bra terms
    double log_vk = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 ket terms
    double log_vb = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 bra terms

    //𝟣𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| = ⟨𝒉ˈᵒˡᵈ|
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
      for(int j = 0; j < _L; j++)
        log_ket += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));

    }

    //𝟤𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ = |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| ≠ ⟨𝒉ˈᵒˡᵈ|
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the bra: Σ𝒿 (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
      for(int j = 0; j < _L; j++)
        log_bra += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the bra: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    }

    //𝟥𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| ≠ ⟨𝒉ˈᵒˡᵈ|
    else if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem != 0){

      //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j = 0; j < _L; j++){

        log_ket += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));
        log_bra += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

      }

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));
      for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    }

    else{

      std::cerr << " ##OptionError: something went wrong in the determination of 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    return this -> rho().real() * (log_ket + log_bra) - 2.0 * this -> omega().real() * (log_vk + log_vb);

  }

}


double BSWF :: q_over_q_braket(const Mat <int>& real_config,
                               const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                               const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_braket(real_config, shadow_ket, flipped_ket_site, shadow_bra, flipped_bra_site));

}


cx_double BSWF :: logPsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                        const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                        const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                        std::string option) const {

  //Function variables
  cx_double logPoP;
  logPoP.imag(0.0);  //In this case the acceptance is a pure real number

  if(option == "real")
    logPoP.real( this -> logq_over_q_real(real_config, flipped_real_site, shadow_ket, shadow_bra));
  else if(option == "ket")
    logPoP.real(this -> logq_over_q_ket(real_config, shadow_ket, flipped_ket_site));
  else if(option == "bra")
    logPoP.real(this -> logq_over_q_bra(real_config, shadow_bra, flipped_bra_site));
  else if(option == "equal site")
    logPoP.real(this -> logq_over_q_equal_site(real_config, flipped_real_site, shadow_ket, shadow_bra));
  else if(option == "braket")
    logPoP.real(this -> logq_over_q_braket(real_config, shadow_ket, flipped_ket_site, shadow_bra, flipped_bra_site));
  else{

    std::cerr << " ##OptionError: no available option as function argument." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  return logPoP;

}


cx_double BSWF :: PsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                     const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                     const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                     std::string option) const {

  //Function variables
  cx_double PoP;

  PoP = std::exp(this -> logPsiNew_over_PsiOld(real_config, flipped_real_site,
                                               shadow_ket, flipped_ket_site,
                                               shadow_bra, flipped_bra_site,
                                               option));
  PoP.imag(0.0);
  return PoP;

}


double BSWF :: PMetroNew_over_PMetroOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                        const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                        const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                        std::string option) const {

  /*######################################################################*/
  //  N̲O̲T̲E̲: in the Shadow 𝒜𝓃𝓈𝒶𝓉𝓏 the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
  //
  /*######################################################################*/

  cx_double p = this -> PsiNew_over_PsiOld(real_config, flipped_real_site,
                                                       shadow_ket, flipped_ket_site,
                                                       shadow_bra, flipped_bra_site,
                                                       option);
  if(p.imag() != 0.0){

    std::cerr << " ##ValueError: the imaginary part of the Metropolis-Hastings acceptance probability must be zero!" << std::endl;
    std::cerr << "   Failed to compute the Metropolis-Hastings acceptance probability." << std::endl;
    std::abort();

  }

  return p.real();

}


void BSWF :: LocalOperators(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameters
  //  𝛂 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the Baeriswyl-Shadow NNQS the local parameters 𝓞(𝒗,𝒉) are
  //
  //        • η ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒗) = Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣
  //        • ρ ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒉) = Σ𝒿 𝒽𝒿•𝒽𝒿+𝟣       𝓞(𝒗,𝒉ˈ) = 𝓞(𝒉ˈ) = Σ𝒿 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
  //        • ω ‹--› 𝓞(𝒗,𝒉) = Σ𝒿 𝒽𝒿•𝑣𝒿                 𝓞(𝒗,𝒉ˈ) = Σ𝒿 𝑣𝒿•𝒽ˈ𝒿
  //
  //  It is important to note that in the 𝓈ℎ𝒶𝒹ℴ𝓌 wave function the local operators
  //  (which are a geometric properties of the wave function itself) related to
  //  the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interactions and the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction, respectively
  //  depend also on the auxiliary variables, and not only on the actual quantum degrees
  //  of freedom of the system.
  //  These operators are necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  //  We remember that in the 𝓈ℎ𝒶𝒹ℴ𝓌 case the local operators are real.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }

  //Function variables
  double O_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  double O_hh_ket = 0.0;  //Storage variable for the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 terms
  double O_hh_bra = 0.0;  //Storage variable for the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 terms
  double O_vh_ket = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 terms
  double O_vh_bra = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 terms

  for(int j = 0; j < _L; j++){

    O_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));
    O_hh_ket += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));
    O_hh_bra += double(shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));
    O_vh_ket += double(real_config.at(0, j) * shadow_ket.at(0, j));
    O_vh_bra += double(real_config.at(0, j) * shadow_bra.at(0, j));

  }

  _LocalOperators.at(0, 0) = O_vv;  // 𝓞_η(𝒗)
  _LocalOperators.at(0, 1) = O_vv;  // 𝓞_η(𝒗)
  _LocalOperators.at(1, 0) = O_hh_ket;  // 𝓞_ρ(𝒉)
  _LocalOperators.at(1, 1) = O_hh_bra;  // 𝓞_ρ(𝒉ˈ)
  _LocalOperators.at(2, 0) = O_vh_ket;  // 𝓞_ω(𝒗,𝒉)
  _LocalOperators.at(2, 1) = O_vh_bra;  // 𝓞_ω(𝒗,𝒉ˈ)

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/****************************  𝐍𝐄𝐗𝐓-𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐑𝐒 𝐁𝐀𝐄𝐑𝐈𝐒𝐖𝐘𝐋-𝐒𝐇𝐀𝐃𝐎𝐖 𝐖𝐀𝐕𝐄 𝐅𝐔𝐍𝐂𝐓𝐈𝐎𝐍 in 𝗱 = 𝟏  **************************/
/*******************************************************************************************************************************/
NNN_BSWF :: NNN_BSWF(int n_real, bool phi_option, bool imaginary_part_option, int rank)
          : WaveFunction(n_real, phi_option, imaginary_part_option) {

  /*########################################################################################################*/
  //  Random-based constructor.
  //  Initializes the Shadow wave function with n.n.n. correlations variational parameters
  //  𝓥 = {𝜙, η, ρ𝟣, ρ𝟤, ω} = {𝜙, 𝛂} to some small random numbers.
  //
  //  In this case we have only 𝟰 parameters, which do not depend on the lattice site
  //  of the variables to which they refer, regardless of the boundary conditions imposed
  //  on the system.
  //  In particular we have
  //
  //        𝟏 complex phase 𝜙
  //        𝟏 𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength weights η;
  //        𝟏 𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights ρ𝟣;
  //        𝟏 𝓃ℯ𝓍𝓉-𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights ρ𝟤;
  //        𝟏 local 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights ω.
  //
  //  Note that in this case the number of variational parameters remains equal to 𝟰 for any system size 𝖫.
  /*########################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a 1D wave function with next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌 correlations with randomly initialized variational parameters 𝓥 = {𝜙,η,ρ𝟣,ρ𝟤,ω}." << std::endl;

  //Warning: if 𝖫 ≤ 𝟦 it is necessary to revise the next-nearest-neighbor terms due to repetitions
  if(n_real <= 4){

    std::cerr << " ##ValueError: choose more quantum spins in the system." << std::endl;
    std::cerr << "   Failed to construct the variational quantum state." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Shadow";
  _alpha.set_size(4);
  _LocalOperators.zeros(4, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI){

    _phi.real(_rnd.Gauss(0.0, 0.01));
    _phi.imag(_rnd.Gauss(0.0, 0.01));

  }
  else _phi = 0.0;
  for(int p = 0; p < _alpha.n_elem; p++){

    _alpha[p].real(_rnd.Gauss(0.0, 0.01));
    if(_if_ZERO_IMAGINARY_PART) _alpha[p].imag(0.0);
    else _alpha[p].imag(_rnd.Gauss(0.0, 0.01));

  }

  if(rank == 0){

    std::cout << " Next-to nearest-neighbor Baeriswyl-Shadow wave function correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
    std::cout << " Number of 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
    std::cout << " Density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << this -> shadow_density() << "." << std::endl;
    std::cout << " Variational parameters at initial time \t → ϕ(𝟢) = " << this -> phi() << std::endl;
    std::cout << "                                        \t → η(𝟢) = " << this -> eta() << std::endl;
    std::cout << "                                        \t → ρ𝟣(𝟢) = " << this -> rho1() << std::endl;
    std::cout << "                                        \t → ρ𝟤(𝟢) = " << this -> rho2() << std::endl;
    std::cout << "                                        \t → ω(𝟢) = " << this -> omega() << std::endl << std::endl;

  }

}


NNN_BSWF :: NNN_BSWF(std::string file_wf, bool phi_option, int rank)
          : WaveFunction(0, phi_option, 0) {

  /*#################################################################################*/
  //  File-based constructor.
  //  Initializes the 𝓈ℎ𝒶𝒹ℴ𝓌 wave function with n.n.n. correlations
  //  variational parameters 𝓥 = {𝜙, η, ρ𝟣, ρ𝟤, ω} = {𝜙, 𝛂} from a given
  //  external file in '.wf' format;
  //  this can be useful in a second moment during a check phase after the
  //  stochastic optimization or to start a time-dependent variational Monte Carlo
  //  with a previously optimized ground state wave function.
  //  The structure of the input file is easily understandable
  //  from the code lines below.
  /*#################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a 1D wave function with next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌 correlations from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the wave function with next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌 correlations variational parameters 𝓥 = {𝜙,η,ρ𝟣,ρ𝟤,ω} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  input_wf >> _L;
  if(_L <= 4){

    std::cerr << " ##ValueError: choose more quantum spins in the system." << std::endl;
    std::cerr << "   Failed to construct the variational quantum state." << std::endl;
    std::abort();

  }
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the 1D 𝓈ℎ𝒶𝒹ℴ𝓌 n.n.n. 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  _type = "Shadow";
  _alpha.set_size(4);
  _LocalOperators.zeros(4, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  for(int p = 0; p < _alpha.n_elem; p++) input_wf >> _alpha[p];

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " Next-to nearest-neighbor Baeriswyl-Shadow wave function correctly initialized with random interactions." << std::endl;
      std::cout << " Number of 𝓇ℯ𝒶𝑙 neurons = " << _L << "." << std::endl;
      std::cout << " Number of 𝓈ℎ𝒶𝒹ℴ𝓌 neurons = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
      std::cout << " Density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << this -> shadow_density() << "." << std::endl;
      std::cout << " Variational parameters at initial time \t → ϕ(𝟢) = " << this -> phi() << std::endl;
      std::cout << "                                        \t → η(𝟢) = " << this -> eta() << std::endl;
      std::cout << "                                        \t → ρ𝟣(𝟢) = " << this -> rho1() << std::endl;
      std::cout << "                                        \t → ρ𝟤(𝟢) = " << this -> rho2() << std::endl;
      std::cout << "                                        \t → ω(𝟢) = " << this -> omega() << std::endl << std::endl;

    }

  }
  input_wf.close();

}


double NNN_BSWF :: I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*######################################################################*/
  //  Computes the value of the angle
  //
  //        ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽') = Σₖ (𝓞ₖ(𝒗,𝒉) - 𝓞ₖ(𝒗,𝐡ˈ)) • αᴵₖ
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉ˈ⟩
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors terms
  double II_hh_nnn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors terms
  double II_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

  //Nearest-neighbors terms
  for(int j = 0; j < _L; j++){

      II_hh_nnn += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // (𝒽𝒿•𝒽𝒿+𝟣 - 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣) in PBCs
      II_vh += double(real_config.at(0, j) * (shadow_ket.at(0, j) - shadow_bra.at(0, j)));  // 𝓋𝒿•(𝒽𝒿 - 𝒽ˈ𝒿)

  }

  //Next-nearest-neighbors terms
  for(int j = 0; j < _L; j++)
    II_hh_nnn += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));  // (𝒽𝒿•𝒽𝒿+𝟤 - 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟤) in PBCs

  return this -> rho1().imag() * II_hh_nn + this -> rho2().imag() * II_hh_nnn + this -> omega().imag() * II_vh;

}


double NNN_BSWF :: cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::cos(this -> I_minus_I(real_config, shadow_ket, shadow_bra));

}


double NNN_BSWF :: sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::sin(this -> I_minus_I(real_config, shadow_ket, shadow_bra));

}


cx_double NNN_BSWF :: logPhi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  /*##########################################################################################################*/
  //  Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)] with
  //
  //        Φ(𝒗,𝒉,𝛂) = ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉) • αₖ)
  //
  //  Φ is that part of variational Shadow 𝒜𝓃𝓈𝒶𝓉𝓏 that appears in the 𝐕𝐌𝐂 calculation
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
  //  N̲O̲T̲E̲: the 𝒔𝒉𝒂𝒅𝒐𝒘_𝐜𝐨𝐧𝐟𝐢𝐠 argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                Φ(𝒗,𝒉,𝛂)
  //                   or
  //                Φ(𝒗,𝒉ˈ,𝛂).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(shadow_config.n_rows != 1 || shadow_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  cx_double log_hh_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors terms
  cx_double log_hh_nnn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors terms
  cx_double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

  //Nearest-neighbors terms
  for(int j = 0; j < _L; j++){

      log_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // 𝓋𝒿•𝓋𝒿+𝟣 in PBCs
      log_hh_nn += double(shadow_config.at(0, j) * shadow_config.at(0, (j + 1) % _L));  // 𝒽𝒿•𝒽𝒿+𝟣 or 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣 in PBCs
      log_vh += double(real_config.at(0, j) * shadow_config.at(0, j));  // 𝓋𝒿•𝒽𝒿 or 𝓋𝒿•𝒽ˈ𝒿

  }

  //Next-nearest-neighbors terms
  for(int j = 0; j < _L; j++) log_hh_nnn += double(shadow_config.at(0, j) * shadow_config.at(0, (j + 2) % _L));  // 𝒽𝒿•𝒽𝒿+𝟤 or 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟤 in PBCs

  return this -> eta() * log_vv + this -> rho1() * log_hh_nn + this -> rho2() * log_hh_nnn + this -> omega() * log_vh;

}


cx_double NNN_BSWF :: Phi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhi(real_config, shadow_config));

}


cx_double NNN_BSWF :: logPhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                            const Mat <int>& shadow_config) const {

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(shadow_config.n_rows != 1 || shadow_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled 𝓇ℯ𝒶𝑙 configuration |𝒗ⁿᵉʷ⟩
  if(flipped_real_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉) / Φ(𝒗ᵒˡᵈ,𝒉)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿 with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝒓𝒆𝒂𝒍_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_real_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_real_site.at(j_row, 0)) * shadow_config.at(0, flipped_real_site.at(j_row, 0)));

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    return -2.0 * this -> omega() * log_vh + this -> eta() * log_vv;

  }

}


cx_double NNN_BSWF :: PhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                         const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_config));

}


cx_double NNN_BSWF :: logPsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*##################################################################################################*/
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
  //        Ψ(𝒗,𝜙,𝛂) = ℯ𝓍𝓅(𝜙) • Σₕ ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉) • αₖ)
  //                 = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅( η • Σ𝒿 𝓋𝒿•𝓋𝒿+𝟣 ) •
  //                           • Σ𝒽 ℯ𝓍𝓅( ρ𝟣 • Σ𝒿 (𝒽𝒿•𝒽𝒿+𝟣) + ρ𝟤 • Σ𝒿 (𝒽𝒿•𝒽𝒿+𝟤) + ω • Σ𝒿 (𝓋𝒿•𝒽𝒿) )
  //
  //  is the variational next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌 wave function characterized by the variational
  //  parameters {𝜙, 𝛂} = {𝜙,η,ρ𝟣,ρ𝟤,ω}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        Σ𝑣Σ𝒽Σ𝒽' 𝓆(𝒗,𝒉,𝒉ˈ) 𝒻(𝒗,𝒉,𝒉ˈ) = ⟨𝒻(𝒗,𝒉,𝒉ˈ)⟩𝓆 / ⟨𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))⟩𝓆.
  //
  //  So it is clear that the classical probability part 𝓆(𝒗,𝒉,𝒉ˈ) plays the role of
  //  square modulus of the wave function with which to sample the 𝓈ℎ𝒶𝒹ℴ𝓌 configurations |𝒗, 𝒉, 𝒉ˈ⟩
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        𝓆(𝒗,𝒉,𝒉ˈ) = ℯ𝓍𝓅(2𝜙ᴿ) • ℯ𝓍𝓅(ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ))
  //
  //  where
  //
  //        ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ) = Σₖ (𝓞ₖ(𝒗,𝒉) + 𝓞ₖ(𝒗,𝒉ˈ)) • αᴿₖ
  //
  //  and it has to be calculated on the current configuration |𝒗 𝒉 𝒉ˈ⟩.
  /*##################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Function variables
  double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  double log_hh_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors terms
  double log_hh_nnn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors terms
  double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  cx_double log_q;

  //Nearest-neighbors terms
  for(int j = 0; j < _L; j++){

    log_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // Σ𝒿 𝓋𝒿•𝓋𝒿+𝟣 in PBCs
    log_hh_nn += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L) + shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // Σ𝒿 𝒽𝒿•𝒽𝒿+𝟣 + 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣 in PBCs
    log_vh += double(real_config.at(0, j) * (shadow_ket.at(0, j) + shadow_bra.at(0, j)));  // Σ𝒿 𝓋𝒿•(𝒽𝒿 + 𝒽ˈ𝒿)

  }

  //Next-nearest-neighbors terms
  for(int j = 0; j < _L; j++)
    log_hh_nnn += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L) + shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));  // Σ𝒿 𝒽𝒿•𝒽𝒿+𝟤 + 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟤 in PBCs

  log_q.imag(0.0);
  log_q.real(2.0 * this -> phi().real() + 2.0 * this -> eta().real() * log_vv + this -> rho1().real() * log_hh_nn + this -> rho2().real() * log_hh_nnn + this -> omega().real() * log_vh);
  return log_q;

}


cx_double NNN_BSWF :: PsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  //Function variables
  cx_double P;
  P.imag(0.0);
  P.real(std::exp(this -> logPsiMetro(real_config, shadow_ket, shadow_bra)).real());

  return P;

}


double NNN_BSWF :: logq_over_q_real(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                    const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓇ℯ𝒶𝑙 variables have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
  if(flipped_real_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ = |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 term: Σ𝒿 (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 𝓋𝒿ᵒˡᵈ•(𝒽𝒿 + 𝒽ˈ𝒿) with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝒓𝒆𝒂𝒍_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_real_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_real_site.at(j_row, 0)) * (shadow_ket.at(0, flipped_real_site.at(j_row, 0)) + shadow_bra.at(0, flipped_real_site.at(j_row, 0))));

    return 2.0 * this -> eta().real() * log_vv - 2.0 * this -> omega().real() * log_vh;

  }

}


double NNN_BSWF :: q_over_q_real(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                 const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::exp(this -> logq_over_q_real(real_config, flipped_real_site, shadow_ket, shadow_bra));

}


double NNN_BSWF :: logq_over_q_ket(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site) const {

  /*#################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ket have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
  if(flipped_ket_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |𝒉ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    double log_hh_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors terms
    double log_hh_nnn = 0.0; //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the nearest-neighbors term: Σ𝒿 (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh_nn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the next-nearest-neighbors term: Σ𝒿 (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟤ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟤ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh_nnn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 2) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟤ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟤ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐤𝐞𝐭_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));

    return this -> rho1().real() * log_hh_nn + this -> rho2().real() * log_hh_nnn - 2.0 * this -> omega().real() * log_vh;

  }

}


double NNN_BSWF :: q_over_q_ket(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site) const {

  return std::exp(this -> logq_over_q_ket(real_config, shadow_ket, flipped_ket_site));

}


double NNN_BSWF :: logq_over_q_bra(const Mat <int>& real_config, const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  /*#################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]
  //  evaluated in a new proposed configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ket have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
  if(flipped_bra_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_hh_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors terms
    double log_hh_nnn = 0.0; //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the nearest-neighbors term: Σ𝒿 (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh_nn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the next-nearest-neighbors term: Σ𝒿 (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟤ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟤ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh_nnn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 2) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟤ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟤ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽ˈ𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐛𝐫𝐚_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
      log_vh += double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    return this -> rho1().real() * log_hh_nn + this -> rho2().real() * log_hh_nnn - 2.0 * this -> omega().real() * log_vh;

  }

}


double NNN_BSWF :: q_over_q_bra(const Mat <int>& real_config, const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_bra(real_config, shadow_bra, flipped_bra_site));

}


double NNN_BSWF :: logq_over_q_equal_site(const Mat <int>& real_config, const Mat <int>& flipped_equal_site,
                                          const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*###################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_equal_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_equal_site);  // |𝒗ⁿᵉʷ⟩
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_equal_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_equal_site);  // |𝒉ˈⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
    double log_hh_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors terms
    double log_hh_nnn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors terms

    //Nearest-neighbors terms
    for(int j = 0; j < _L; j++){

      log_vv += double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs
      log_hh_nn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ) in PBCs
      log_hh_nn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ) in PBCs

    }

    //Next-nearest-neighbors terms
    for(int j = 0; j < _L; j++){

      log_hh_nnn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 2) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L));  // (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟤ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟤ᵒˡᵈ) in PBCs
      log_hh_nnn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 2) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));  // (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟤ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟤ᵒˡᵈ) in PBCs

    }

    return 2.0 * this -> eta().real() * log_vv + this -> rho1().real() * log_hh_nn + this -> rho2().real() * log_hh_nnn;

  }

}


double NNN_BSWF :: q_over_q_equal_site(const Mat <int>& real_config, const Mat <int>& flipped_equal_site,
                                       const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::exp(this -> logq_over_q_equal_site(real_config, flipped_equal_site, shadow_ket, shadow_bra));

}


double NNN_BSWF :: logq_over_q_braket(const Mat <int>& real_config,
                                      const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                      const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  /*################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_ket_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors ket terms
    double log_ket_nnn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors ket terms
    double log_bra_nn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 nearest-neighbors bra terms
    double log_bra_nnn = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 next-nearest-neighbors bra terms
    double log_vk = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 ket terms
    double log_vb = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 bra terms

    //𝟣𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| = ⟨𝒉ˈᵒˡᵈ|
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket
      for(int j = 0; j < _L; j++)
        log_ket_nn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));

      //Computes next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket
      for(int j = 0; j < _L; j++)
        log_ket_nnn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 2) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L));

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));

    }

    //𝟤𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ = |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| ≠ ⟨𝒉ˈᵒˡᵈ|
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket
      for(int j = 0; j < _L; j++)
        log_bra_nn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

      //Computes next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket
      for(int j = 0; j < _L; j++)
        log_bra_nnn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 2) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the bra: Σ𝒿 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    }

    //𝟥𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & ⟨𝒉ˈⁿᵉʷ| ≠ ⟨𝒉ˈᵒˡᵈ|
    else if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem != 0){

      //Computes the nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j = 0; j < _L; j++){

        log_ket_nn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));
        log_bra_nn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

      }

      //Computes the next-nearest-neighbors 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j = 0; j < _L; j++){

        log_ket_nnn += double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 2) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L));
        log_bra_nnn += double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 2) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));

      }

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));
      for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    }

    else{

      std::cerr << " ##OptionError: something went wrong in the determination of 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    return this -> rho1().real() * (log_ket_nn + log_bra_nn) + this -> rho2().real() * (log_ket_nnn + log_bra_nnn) - 2.0 * this -> omega().real() * (log_vk + log_vb);

  }

}


double NNN_BSWF :: q_over_q_braket(const Mat <int>& real_config,
                                   const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                   const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_braket(real_config, shadow_ket, flipped_ket_site, shadow_bra, flipped_bra_site));

}


cx_double NNN_BSWF :: logPsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                            const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                            const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                            std::string option) const {

  //Function variables
  cx_double logPoP;
  logPoP.imag(0.0);  //In this case the acceptance is a pure real number

  if(option == "real")
    logPoP.real( this -> logq_over_q_real(real_config, flipped_real_site, shadow_ket, shadow_bra));
  else if(option == "ket")
    logPoP.real(this -> logq_over_q_ket(real_config, shadow_ket, flipped_ket_site));
  else if(option == "bra")
    logPoP.real(this -> logq_over_q_bra(real_config, shadow_bra, flipped_bra_site));
  else if(option == "equal site")
    logPoP.real(this -> logq_over_q_equal_site(real_config, flipped_real_site, shadow_ket, shadow_bra));
  else if(option == "braket")
    logPoP.real(this -> logq_over_q_braket(real_config, shadow_ket, flipped_ket_site, shadow_bra, flipped_bra_site));
  else{

    std::cerr << " ##OptionError: no available option as function argument." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  return logPoP;

}


cx_double NNN_BSWF :: PsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                         const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                         const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                         std::string option) const {

  //Function variables
  cx_double PoP;

  PoP = std::exp(this -> logPsiNew_over_PsiOld(real_config, flipped_real_site,
                                               shadow_ket, flipped_ket_site,
                                               shadow_bra, flipped_bra_site,
                                               option));
  PoP.imag(0.0);
  return PoP;

}


double NNN_BSWF :: PMetroNew_over_PMetroOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                            const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                            const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                            std::string option) const {

  /*######################################################################*/
  //  N̲O̲T̲E̲: in the 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ,𝓥)/𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ,𝓥)
  //
  /*######################################################################*/

  cx_double p = this -> PsiNew_over_PsiOld(real_config, flipped_real_site,
                                                       shadow_ket, flipped_ket_site,
                                                       shadow_bra, flipped_bra_site,
                                                       option);
  if(p.imag() != 0.0){

    std::cerr << " ##ValueError: the imaginary part of the Metropolis-Hastings acceptance probability must be zero!" << std::endl;
    std::cerr << "   Failed to compute the Metropolis-Hastings acceptance probability." << std::endl;
    std::abort();

  }

  return p.real();

}


void NNN_BSWF :: LocalOperators(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameters
  //  𝛂 on the sampled enlarged quantum configuration |𝒗 𝒉 𝒉ˈ⟩.
  //  In the case of the 𝓈ℎ𝒶𝒹ℴ𝓌 wave function with n.n.n. 𝓈ℎ𝒶𝒹ℴ𝓌 correlations
  //  the local parameters 𝓞(𝒗,𝒉) are
  //
  //        • η ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒗) = Σ𝒿 𝑣𝒿•𝑣𝒿+𝟣
  //        • ρ𝟣 ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒉) = Σ𝒿 𝒽𝒿•𝒽𝒿+𝟣       𝓞(𝒗,𝒉ˈ) = 𝓞(𝒉ˈ) = Σ𝒿 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣
  //        • ρ𝟤 ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒉) = Σ𝒿 𝒽𝒿•𝒽𝒿+𝟤       𝓞(𝒗,𝒉ˈ) = 𝓞(𝒉ˈ) = Σ𝒿 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟤
  //        • ω ‹--› 𝓞(𝒗,𝒉) = Σ𝒿 𝒽𝒿•𝑣𝒿                 𝓞(𝒗,𝒉ˈ) = Σ𝒿 𝑣𝒿•𝒽ˈ𝒿
  //
  //  It is important to note that in the 𝓈ℎ𝒶𝒹ℴ𝓌 wave function the local operators
  //  (which are a geometric properties of the wave function itself) related to
  //  the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interactions and the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction, respectively
  //  depend also on the auxiliary variables, and not only on the actual quantum degrees
  //  of freedom of the system.
  //  These operators are necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  //  We remember that in the 𝓈ℎ𝒶𝒹ℴ𝓌 case the local operators are real.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }

  //Function variables
  double O_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  double O_hh_ket_nn = 0.0;  //Storage variable for the ket 𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 correlations
  double O_hh_bra_nn = 0.0;  //Storage variable for the bra 𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 correlations
  double O_hh_ket_nnn = 0.0;  //Storage variable for the ket 𝓃ℯ𝓍𝓉-𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 correlations
  double O_hh_bra_nnn = 0.0;  //Storage variable for the bra 𝓃ℯ𝓍𝓉-𝓃ℯ𝒶𝓇ℯ𝓈𝓉-𝓃ℯ𝒾ℊ𝒽𝒷ℴ𝓇𝓈 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 correlations
  double O_vh_ket = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 terms
  double O_vh_bra = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 terms

  //Nearest-neighbors correlations
  for(int j = 0; j < _L; j++){

    O_vv += double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));
    O_hh_ket_nn += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));
    O_hh_bra_nn += double(shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));
    O_vh_ket += double(real_config.at(0, j) * shadow_ket.at(0, j));
    O_vh_bra += double(real_config.at(0, j) * shadow_bra.at(0, j));

  }

  //Next-nearest-neighbors correlations
  for(int j = 0; j < _L; j++){

    O_hh_ket_nnn += double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 2) % _L));
    O_hh_bra_nnn += double(shadow_bra.at(0, j) * shadow_bra.at(0, (j + 2) % _L));

  }

  _LocalOperators.at(0, 0) = O_vv;  // 𝓞_η(𝒗)
  _LocalOperators.at(0, 1) = O_vv;  // 𝓞_η(𝒗)
  _LocalOperators.at(1, 0) = O_hh_ket_nn;  // 𝓞_ρ𝟣(𝒉)
  _LocalOperators.at(1, 1) = O_hh_bra_nn;  // 𝓞_ρ𝟣(𝒉ˈ)
  _LocalOperators.at(2, 0) = O_hh_ket_nnn;  // 𝓞_ρ𝟤(𝒉)
  _LocalOperators.at(2, 1) = O_hh_bra_nnn;  // 𝓞_ρ𝟤(𝒉ˈ)
  _LocalOperators.at(3, 0) = O_vh_ket;  // 𝓞_ω(𝒗,𝒉)
  _LocalOperators.at(3, 1) = O_vh_bra;  // 𝓞_ω(𝒗,𝒉ˈ)

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/*****************************************  𝐍𝐄𝐀𝐑𝐄𝐒𝐓-𝐍𝐄𝐈𝐆𝐇𝐁𝐎𝐔𝐑 (quasi)-𝐮𝐑𝐁𝐌 in 𝗱 = 𝟏  ****************************************/
/*******************************************************************************************************************************/
quasi_uRBM :: quasi_uRBM(int n_real, bool phi_option, bool imaginary_part_option, int rank)
            : WaveFunction(n_real, phi_option, imaginary_part_option) {

  /*######################################################################################################*/
  //  Random-based constructor.
  //  Initializes the (quasi)-uRBM variational parameters 𝓥 = {𝜙,𝛈,𝛒,𝒘} = {𝜙,𝛂} to
  //  some small random numbers.
  //
  //  Imposing periodic boundary conditions we have
  //
  //        𝟏 complex phase 𝜙
  //        𝖫 nearest-neighbors 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength weights 𝛈 = {ηⱼ};
  //        𝖫 nearest-neighbors 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights 𝛒 = {ρⱼ};
  //        𝖫 local 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction strength weights 𝒘 = {𝑤ⱼ};
  //
  //  We remember only in the special case of 𝟏 dimension the size of the sets of intra- and extra-layer
  //  connections is the same, since in 𝟏 dimension the number of nearest-neighbors site is 𝟐.
  /*######################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Create a 1D n.n. (quasi)-uRBM wave function with randomly initialized variational parameters 𝓥 = {𝜙,𝛈,𝛒,𝒘}." << std::endl;

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed1.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  _type = "Shadow";
  _alpha.set_size(3 * _L);
  _LocalOperators.zeros(3 * _L, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  if(_if_PHI) {

    _phi.real(_rnd.Gauss(0.0, 0.001));
    _phi.imag(_rnd.Gauss(0.0, 0.001));

  }
  else _phi = 0.0;
  for(int p = 0; p < _alpha.n_elem; p++){

    _alpha[p].real(_rnd.Gauss(0.0, 0.001));
    if(_if_ZERO_IMAGINARY_PART) _alpha[p].imag(0.0);
    else _alpha[p].imag(_rnd.Gauss(0.0, 0.001));

  }

  if(rank == 0){

    std::cout << " (quasi)-uRBM 𝒜𝓃𝓈𝒶𝓉𝓏 correctly initialized with random interactions." << std::endl;
    std::cout << " Number of 𝓇ℯ𝒶𝑙 variables = " << _L << "." << std::endl;
    std::cout << " Number of 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << _L << "." << std::endl;
    std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
    std::cout << " Density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << this -> shadow_density() << "." << std::endl << std::endl;

  }

}


quasi_uRBM :: quasi_uRBM(std::string file_wf, bool phi_option, int rank)
            : WaveFunction(0, phi_option, 0) {

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
  if(rank == 0) std::cout << "#Create a 1D n.n. (quasi)-uRBM wave function from an existing quantum state." << std::endl;

  std::ifstream input_wf(file_wf.c_str());
  if(!input_wf.good()){

    std::cerr << " ##FileError: failed to open the quantum state file " << file_wf << "." << std::endl;
    std::cerr << "   File not found." << std::endl;
    std::cerr << "   Failed to initialize the (quasi)-uRBM variational parameters 𝓥 = {𝜙,𝛈,𝛒,𝒘} from file." << std::endl;
    std::abort();

  }

  //Creates and initializes the Random Number Generator
  int seed[4];
  int p1, p2;
  std::ifstream Primes("./input_random_device/Primes_32001.in");
  if(Primes.is_open()) Primes >> p1 >> p2;
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
    if(rank == 0) std::cout << " Random device created correctly." << std::endl;

  }
  else{

    std::cerr << " ##FileError: Unable to open seed.in." << std::endl;
    std::cerr << "   Failed to initialize the random device." << std::endl;
    std::abort();

  }

  //Data-members initialization
  input_wf >> _L;
  if(_if_PHI) input_wf >> _phi;
  if(!input_wf.good() || _L < 0){

    std::cerr << " ##FileError: invalid construction of the 1D n.n. (quasi)-uRBM 𝒜𝓃𝓈𝒶𝓉𝓏." << std::endl;
    std::abort();

  }
  _type = "Shadow";
  _alpha.set_size(3 * _L);
  _LocalOperators.zeros(3 * _L, 2);  //N̲O̲T̲E̲: 𝓞_𝜙 = 𝟙, so we do not save it in memory
  for(int p = 0; p < _alpha.n_elem; p++) input_wf >> _alpha[p];

  if(input_wf.good()){

    if(rank == 0){

      std::cout << " (quasi)-uRBM 𝒜𝓃𝓈𝒶𝓉𝓏 correctly loaded from file " << file_wf << "." << std::endl;
      std::cout << " Number of 𝓇ℯ𝒶𝑙 neurons = " << _L << "." << std::endl;
      std::cout << " Number of 𝓈ℎ𝒶𝒹ℴ𝓌 neurons = " << _L << "." << std::endl;
      std::cout << " Number of variational parameters 𝛂 = " << _alpha.n_elem << "." << std::endl;
      std::cout << " Density of the 𝓈ℎ𝒶𝒹ℴ𝓌 variables = " << this -> shadow_density() << "." << std::endl << std::endl;

    }

  }
  input_wf.close();

}


cx_double quasi_uRBM :: eta_j(int j) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || j < 0){

    std::cerr << " ##IndexError: failed to access the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction strength vector 𝛈." << std::endl;
    std::cerr << "   Element ηⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[j];

}


cx_double quasi_uRBM :: rho_j(int j) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || j < 0){

    std::cerr << " ##IndexError: failed to access the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 interaction strength vector 𝛒." << std::endl;
    std::cerr << " Element ρⱼₖ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[_L + j];

}


cx_double quasi_uRBM :: omega_j(int j) const {  //Useful for debugging

  //Check on the choosen index: the first element has index 0 in C++
  if(j >= _L || j < 0){

    std::cerr << " ##IndexError: failed to access the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 interaction strength vector 𝒘." << std::endl;
    std::cerr << " Element 𝑤ⱼ with j = " << j << " does not exist." << std::endl;
    return -1.0;

  }

  //Check passed
  else return _alpha[2 * _L + j];

}


void quasi_uRBM :: print_eta() const {  //Useful for debugging

  std::cout << "\n===========================================" << std::endl;
  std::cout << "quasi_uRBM 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interaction vector 𝛈" << std::endl;
  std::cout << "===========================================" << std::endl;
  for(int j = 0; j < _L; j++){

    std::cout << _alpha[j].real();
    if(_alpha[j].imag() >= 0)
      std::cout << " + i" << _alpha[j].imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha[j].imag() << "  " << std::endl;

  }

}


void quasi_uRBM :: print_rho() const {  //Useful for debugging

  std::cout << "\n===============================================" << std::endl;
  std::cout << "quasi_uRBM 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 interaction vector 𝛒" << std::endl;
  std::cout << "===============================================" << std::endl;
  for(int j = 0; j < _L; j++){

    std::cout << _alpha[_L + j].real();
    if(_alpha[_L + j].imag() >= 0)
      std::cout << " + i" << _alpha[_L + j].imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha[_L + j].imag() << "  " << std::endl;

  }

}


void quasi_uRBM :: print_omega() const {  //Useful for debugging

  std::cout << "\n================================================" << std::endl;
  std::cout << "quasi_uRBM 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 interaction vector 𝒘" << std::endl;
  std::cout << "================================================" << std::endl;
  for(int j = 0; j < _L; j++){

    std::cout << _alpha[2 * _L + j].real();
    if(_alpha[2 * _L + j].imag() >= 0)
      std::cout << " + i" << _alpha[2 * _L + j].imag() << "  " << std::endl;
    else
      std::cout << " - i" << -1.0 * _alpha[2 * _L + j].imag() << "  " << std::endl;

  }

}


double quasi_uRBM :: I_minus_I(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*######################################################################*/
  //  Computes the value of the angle
  //
  //        ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽') = Σₖ (𝓞ₖ(𝒗,𝒉) - 𝓞ₖ(𝒗,𝐡ˈ)) • αᴵₖ
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the angle ℐ(𝑣, 𝒽) - ℐ(𝑣, 𝒽')." << std::endl;
    std::abort();

  }

  //Function variables
  double II_hh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  double II_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

  for(int j = 0; j < _L; j++){

    II_hh += this -> rho_j(j).imag() * double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // ρᴵ𝒿 • (𝒽𝒿•𝒽𝒿+𝟣 - 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣) in PBCs
    II_vh += this -> omega_j(j).imag() * double(real_config.at(0, j) * (shadow_ket.at(0, j) - shadow_bra.at(0, j)));  // ωᴵ𝒿•𝓋𝒿 • (𝒽𝒿 - 𝒽ˈ𝒿)

  }

  return II_hh + II_vh;

}


double quasi_uRBM :: cosII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::cos(this -> I_minus_I(real_config, shadow_ket, shadow_bra));

}


double quasi_uRBM :: sinII(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::sin(this -> I_minus_I(real_config, shadow_ket, shadow_bra));

}


cx_double quasi_uRBM :: logPhi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  /*##########################################################################################################*/
  //  Computes 𝑙𝑜𝑔[Φ(𝒗,𝒉,𝛂)] with
  //
  //        Φ(𝒗,𝒉,𝛂) = ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉) • αₖ)
  //
  //  Φ is that part of variational 𝓈ℎ𝒶𝒹ℴ𝓌 𝒜𝓃𝓈𝒶𝓉𝓏 that appears in the 𝐕𝐌𝐂 calculation
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
  //  N̲O̲T̲E̲: the 𝒔𝒉𝒂𝒅𝒐𝒘_𝐜𝐨𝐧𝐟𝐢𝐠 argument can be both a ket and a bra system sampled configuration
  //        i.e.
  //
  //                Φ(𝒗,𝒉,𝛂)
  //                   or
  //                Φ(𝒗,𝒉ˈ,𝛂).
  /*##########################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(shadow_config.n_rows != 1 || shadow_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute Φ(𝒗,𝒉,𝛂)." << std::endl;
    std::abort();

  }

  //Function variables
  cx_double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  cx_double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  cx_double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

  for(int j = 0; j < _L; j++){

    log_vv += this -> eta_j(j) * double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // η𝒿 • 𝓋𝒿•𝓋𝒿+𝟣 in PBCs
    log_hh += this -> rho_j(j) * double(shadow_config.at(0, j) * shadow_config.at(0, (j + 1) % _L));  // ρ𝒿 • 𝒽𝒿•𝒽𝒿+𝟣 or ρ𝒿 • 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣 in PBCs
    log_vh += this -> omega_j(j) * double(real_config.at(0, j) * shadow_config.at(0, j));  // ω𝒿 • 𝓋𝒿•𝒽𝒿 or ω𝒿 • 𝓋𝒿•𝒽ˈ𝒿

  }

  return log_vv + log_hh + log_vh;

}


cx_double quasi_uRBM :: Phi(const Mat <int>& real_config, const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhi(real_config, shadow_config));

}


cx_double quasi_uRBM :: logPhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                                          const Mat <int>& shadow_config) const {

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩ or ⟨𝒉ˈ|
  if(shadow_config.n_rows != 1 || shadow_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓈ℎ𝒶𝒹ℴ𝓌 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled visible configuration |𝒗ⁿᵉʷ⟩
  if(flipped_real_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ⟩ = |𝒗ᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the matrix representation of the new 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
      std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[Φ(𝒗ⁿᵉʷ,𝒉,𝛂) / Φ(𝒗ᵒˡᵈ,𝒉,𝛂)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    cx_double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    cx_double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms: Σ𝒿 ω𝒿•𝓋𝒿ᵒˡᵈ•𝒽𝒿 with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝒓𝒆𝒂𝒍_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_real_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_real_site.at(j_row, 0)) * double(real_config.at(0, flipped_real_site.at(j_row, 0)) * shadow_config.at(0, flipped_real_site.at(j_row, 0)));

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms: Σ𝒿 η𝒿•(𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += this -> eta_j(j) * double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // η𝒿•(𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    return -2.0 * log_vh + log_vv;

  }

}


cx_double quasi_uRBM :: PhiNew_over_PhiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                           const Mat <int>& shadow_config) const {

  return std::exp(this -> logPhiNew_over_PhiOld(real_config, flipped_real_site, shadow_config));

}


cx_double quasi_uRBM :: logPsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

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
  //        Ψ(𝒗,𝓥) = Ψ(𝒗,𝜙,𝛂) = ℯ𝓍𝓅(𝜙)•Σₕ ℯ𝓍𝓅(Σₖ 𝓞ₖ(𝒗,𝒉) • αₖ)
  //                = ℯ𝓍𝓅(𝜙) • ℯ𝓍𝓅{Σⱼₖ ηⱼₖ𝑣ⱼ𝑣ₖ} • Σₕ ℯ𝓍𝓅(Σⱼₖ ρⱼₖ𝒽ⱼ𝒽ₖ + Σⱼ 𝓌ⱼ𝑣ⱼ𝒽ₖ}
  //
  //  is the variational 𝓈ℎ𝒶𝒹ℴ𝓌 wave function characterized by the variational
  //  parameters {𝜙, 𝛂} = {𝜙,𝛈,𝛒,𝒘}.
  //  We are interested in computing, in a Monte Carlo framework, expectation values
  //  of the following kind:
  //
  //        Σ𝑣Σ𝒽Σ𝒽' 𝓆(𝒗,𝒉,𝒉ˈ,𝓥) 𝒻(𝒗,𝒉,𝒉ˈ) = ⟨𝒻(𝒗,𝒉,𝒉ˈ)⟩𝓆 / ⟨𝑐𝑜𝑠(ℐ(𝒗,𝒉)-ℐ(𝒗,𝒉ˈ))⟩𝓆.
  //
  //  So it is clear that the classical probability part 𝓆(𝒗,𝒉,𝒉ˈ,𝓥) plays the role of
  //  square modulus of the wave function with which to sample the 𝓈ℎ𝒶𝒹ℴ𝓌 configurations |𝒗, 𝒉, 𝒉ˈ⟩
  //  with the Metropolis-Hastings algorithm, and for this reason its determination is made within
  //  this virtual function, although it does not represent the whole variational wave function.
  //
  //  However, this is defined as
  //
  //        𝓆(𝒗,𝒉,𝒉ˈ,𝓥) = ℯ𝓍𝓅(2ϕᴿ) • ℯ𝓍𝓅(ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ))
  //
  //  where
  //
  //        ℛ(𝑣, 𝒽) + ℛ(𝑣, 𝒽ˈ) = Σₖ (𝓞ₖ(𝒗,𝒉) + 𝓞ₖ(𝒗,𝒉ˈ)) • αᴿₖ
  //
  //  and it has to be calculated on the current 𝓇ℯ𝒶𝑙 configuration |𝒗⟩ and the sampled
  //  𝓈ℎ𝒶𝒹ℴ𝓌 configuration ket |𝒉⟩ and bra ⟨𝒉ˈ|.
  /*################################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Function variables
  double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
  double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
  cx_double log_psi;

  for(int j = 0; j < _L; j++){

    log_vv += this -> eta_j(j).real() * double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // Σ𝒿 ηᴿ𝒿 • 𝓋𝒿•𝓋𝒿+𝟣 in PBCs
    log_hh += this -> rho_j(j).real() * double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L) + shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // Σ𝒿 ρᴿ𝒿 • 𝒽𝒿•𝒽𝒿+𝟣 + 𝒽ˈ𝒿•𝒽ˈ𝒿+𝟣 in PBCs
    log_vh += this -> omega_j(j).real() * double(real_config.at(0, j) * (shadow_ket.at(0, j) + shadow_bra.at(0, j)));  // Σ𝒿 ωᴿ𝒿 • 𝓋𝒿•(𝒽𝒿 + 𝒽ˈ𝒿)

  }

  log_psi.real(2.0 * this -> phi().real() + 2.0 * log_vv + log_hh + log_vh);
  log_psi.imag(0.0);
  return log_psi;

}


cx_double quasi_uRBM :: PsiMetro(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  //Function variables
  cx_double P;
  P.imag(0.0);
  P.real(std::exp(this -> logPsiMetro(real_config, shadow_ket, shadow_bra)).real());

  return P;

}


double quasi_uRBM :: logq_over_q_real(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                      const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*##############################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]
  //  evaluated in a new proposed configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓇ℯ𝒶𝑙 variables have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩
  if(flipped_real_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉 𝒉ˈ⟩ = |𝒗ᵒˡᵈ 𝒉 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_real_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉,𝒉ˈ) / 𝓆(𝒗ᵒˡᵈ,𝒉,𝒉ˈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_real_site);  // |𝒗ⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 term: Σ𝒿 ηᴿ𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_vv += this -> eta_j(j).real() * double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // ηᴿ𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•(𝒽𝒿 + 𝒽ˈ𝒿) with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝒓𝒆𝒂𝒍_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_real_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_real_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_real_site.at(j_row, 0)) *
                (shadow_ket.at(0, flipped_real_site.at(j_row, 0)) + shadow_bra.at(0, flipped_real_site.at(j_row, 0))));

    return 2.0 * log_vv - 2.0 * log_vh;

  }

}


double quasi_uRBM :: q_over_q_real(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                   const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::exp(this -> logq_over_q_real(real_config, flipped_real_site, shadow_ket, shadow_bra));

}


double quasi_uRBM :: logq_over_q_ket(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site) const {

  /*#################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]
  //  evaluated in a new proposed configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ wrt
  //  the current configuration |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables ket have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩
  if(flipped_ket_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled ket configuration |𝒉ⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈ)/𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 term: Σ𝒿 ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh += this -> rho_j(j).real() * double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐤𝐞𝐭_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_ket_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));

    return log_hh - 2.0 * log_vh;

  }

}


double quasi_uRBM :: q_over_q_ket(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site) const {

  return std::exp(this -> logq_over_q_ket(real_config, shadow_ket, flipped_ket_site));

}


double quasi_uRBM :: logq_over_q_bra(const Mat <int>& real_config, const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  /*#################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]
  //  evaluated in a new proposed configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ wrt
  //  the current configuration |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩ (at fixed variational parameters 𝓥),
  //  where only the 𝓈ℎ𝒶𝒹ℴ𝓌 variables bra have been changed.
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩
  if(flipped_bra_site.n_elem==0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled bra configuration ⟨𝒉ˈⁿᵉʷ|." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
    double log_vh = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    //Computes the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 term: Σ𝒿 ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
    for(int j = 0; j < _L; j++)
      log_hh += this -> rho_j(j).real() * double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ) in PBCs

    //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 term: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽ˈ𝒿ᵒˡᵈ with 𝒿 ϵ 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐛𝐫𝐚_𝐬𝐢𝐭𝐞
    for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
      log_vh += this -> omega_j(flipped_bra_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    return log_hh - 2.0 * log_vh;

  }

}


double quasi_uRBM :: q_over_q_bra(const Mat <int>& real_config, const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_bra(real_config, shadow_bra, flipped_bra_site));

}


double quasi_uRBM :: logq_over_q_equal_site(const Mat <int>& real_config, const Mat <int>& flipped_equal_site,
                                            const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  /*###################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_equal_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗ᵒˡᵈ 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_equal_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with equal-site flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_real_config = generate_config(real_config, flipped_equal_site);  // |𝒗ⁿᵉʷ⟩
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_equal_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_equal_site);  // |𝒉ˈⁿᵉʷ⟩
    double log_vv = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 terms
    double log_hh = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms

    for(int j = 0; j < _L; j++){

      log_vv += this -> eta_j(j).real() * double(new_real_config.at(0, j) * new_real_config.at(0, (j + 1) % _L) - real_config.at(0, j) * real_config.at(0, (j + 1) % _L));  // ηᴿ𝒿 • (𝓋𝒿ⁿᵉʷ•𝓋𝒿+𝟣ⁿᵉʷ - 𝓋𝒿ᵒˡᵈ•𝓋𝒿+𝟣ᵒˡᵈ) in PBCs
      log_hh += this -> rho_j(j).real() * double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));  // ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ) in PBCs
      log_hh += this -> rho_j(j).real() * double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));  // ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ) in PBCs

    }

    return 2.0 * log_vv + log_hh;

  }

}


double quasi_uRBM :: q_over_q_equal_site(const Mat <int>& real_config, const Mat <int>& flipped_equal_site,
                                         const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) const {

  return std::exp(this -> logq_over_q_equal_site(real_config, flipped_equal_site, shadow_ket, shadow_bra));

}


double quasi_uRBM :: logq_over_q_braket(const Mat <int>& real_config,
                                        const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                        const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  /*################################################################################*/
  //  Computes 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]
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
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
    std::abort();

  }

  //Check on the new sampled enlarged configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩
  if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem == 0) return 0.0;  //𝑙𝑜𝑔(1) = 0, the case |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩ = |𝒗 𝒉ᵒˡᵈ 𝒉ˈᵒˡᵈ⟩
  else{

    //Check on the lattice dimensionality
    if(flipped_ket_site.n_cols != 1 || flipped_bra_site.n_cols != 1){

      std::cerr << " ##SizeError: the dimensionality of the lattice does not match the size of the new sampled quantum configuration |𝒗 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)] with 𝓈ℎ𝒶𝒹ℴ𝓌 equal-site flipped-spin." << std::endl;
      std::abort();

    }

    //Function variables
    Mat <int> new_shadow_ket = generate_config(shadow_ket, flipped_ket_site);  // |𝒉ⁿᵉʷ⟩
    Mat <int> new_shadow_bra = generate_config(shadow_bra, flipped_bra_site);  // ⟨𝒉ˈⁿᵉʷ|
    double log_ket = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 ket terms
    double log_bra = 0.0;  //Storage variable for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 bra terms
    double log_vk = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 ket terms
    double log_vb = 0.0;  //Storage variable for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 bra terms

    //𝟣𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & |𝒉ˈⁿᵉʷ⟩ = |𝒉ˈᵒˡᵈ⟩
    if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem == 0){

      //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 ρᴿ𝒿 • (𝒽𝒿ⁿᵉʷ•𝒽𝒿+𝟣ⁿᵉʷ - 𝒽𝒿ᵒˡᵈ•𝒽𝒿+𝟣ᵒˡᵈ)
      for(int j = 0; j < _L; j++)
        log_ket += this -> rho_j(j).real() * double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += this -> omega_j(flipped_ket_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));

    }

    //𝟤𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ = |𝒉ᵒˡᵈ⟩ & |𝒉ˈⁿᵉʷ⟩ ≠ |𝒉ˈᵒˡᵈ⟩
    else if(flipped_ket_site.n_elem == 0 && flipped_bra_site.n_elem != 0){

      //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 ρᴿ𝒿 • (𝒽ˈ𝒿ⁿᵉʷ•𝒽ˈ𝒿+𝟣ⁿᵉʷ - 𝒽ˈ𝒿ᵒˡᵈ•𝒽ˈ𝒿+𝟣ᵒˡᵈ)
      for(int j = 0; j < _L; j++)
        log_bra += this -> rho_j(j).real() * double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms only for the ket: Σ𝒿 ωᴿ𝒿 • 𝓋𝒿ᵒˡᵈ•𝒽𝒿ᵒˡᵈ
      for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += this -> omega_j(flipped_bra_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    }

    //𝟥𝓈𝓉 𝒸𝒶𝓈ℯ:  |𝒉ⁿᵉʷ⟩ ≠ |𝒉ᵒˡᵈ⟩ & |𝒉ˈⁿᵉʷ⟩ ≠ |𝒉ˈᵒˡᵈ⟩
    else if(flipped_ket_site.n_elem != 0 && flipped_bra_site.n_elem != 0){

      //Computes the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j = 0; j < _L; j++){

        log_ket += this -> rho_j(j).real() * double(new_shadow_ket.at(0, j) * new_shadow_ket.at(0, (j + 1) % _L) - shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));
        log_bra += this -> rho_j(j).real() * double(new_shadow_bra.at(0, j) * new_shadow_bra.at(0, (j + 1) % _L) - shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

      }

      //Computes the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 terms
      for(int j_row = 0; j_row < flipped_ket_site.n_rows; j_row++)
        log_vk += this -> omega_j(flipped_ket_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_ket_site.at(j_row, 0)) * shadow_ket.at(0, flipped_ket_site.at(j_row, 0)));
      for(int j_row = 0; j_row < flipped_bra_site.n_rows; j_row++)
        log_vb += this -> omega_j(flipped_bra_site.at(j_row, 0)).real() * double(real_config.at(0, flipped_bra_site.at(j_row, 0)) * shadow_bra.at(0, flipped_bra_site.at(j_row, 0)));

    }

    else{

      std::cerr << " ##OptionError: something went wrong in the determination of 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
      std::abort();

    }

    return log_ket + log_bra - 2.0 * (log_vk + log_vb);

  }

}


double quasi_uRBM :: q_over_q_braket(const Mat <int>& real_config,
                                     const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                     const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site) const {

  return std::exp(this -> logq_over_q_braket(real_config,
                                             shadow_ket, flipped_ket_site,
                                             shadow_bra, flipped_bra_site));

}


cx_double quasi_uRBM :: logPsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                              const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                              const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                              std::string option) const {

  //Function variables
  cx_double logPoP;
  logPoP.imag(0.0);  //In this case the acceptance is a pure real number

  if(option == "real")
    logPoP.real(this -> logq_over_q_real(real_config, flipped_real_site, shadow_ket, shadow_bra));
  else if(option == "ket")
    logPoP.real(this -> logq_over_q_ket(real_config, shadow_ket, flipped_ket_site));
  else if(option == "bra")
    logPoP.real(this -> logq_over_q_bra(real_config, shadow_bra, flipped_bra_site));
  else if(option == "equal site")
    logPoP.real(this -> logq_over_q_equal_site(real_config, flipped_real_site, shadow_ket, shadow_bra));
  else if(option == "braket")
    logPoP.real(this -> logq_over_q_braket(real_config, shadow_ket, flipped_ket_site, shadow_bra, flipped_bra_site));
  else{

    std::cerr << " ##OptionError: no available option as function argument." << std::endl;
    std::cerr << "   Failed to compute 𝑙𝑜𝑔[𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)]." << std::endl;
    std::abort();

  }

  return logPoP;

}


cx_double quasi_uRBM :: PsiNew_over_PsiOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                           const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                           const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                           std::string option) const {

  //Function variables
  cx_double PoP;
  PoP.imag(0.0);

  PoP.real(std::exp(this -> logPsiNew_over_PsiOld(real_config, flipped_real_site,
                                                  shadow_ket, flipped_ket_site,
                                                  shadow_bra, flipped_bra_site,
                                                  option).real()));
  return PoP;

}


double quasi_uRBM :: PMetroNew_over_PMetroOld(const Mat <int>& real_config, const Mat <int>& flipped_real_site,
                                              const Mat <int>& shadow_ket, const Mat <int>& flipped_ket_site,
                                              const Mat <int>& shadow_bra, const Mat <int>& flipped_bra_site,
                                              std::string option) const {

  /*######################################################################*/
  //  N̲O̲T̲E̲: in the Shadow 𝒜𝓃𝓈𝒶𝓉𝓏 the acceptance probability
  //        which enters the Metropolis-Hastings test is
  //        precisely 𝓆(𝒗ⁿᵉʷ,𝒉ⁿᵉʷ,𝒉ˈⁿᵉʷ) / 𝓆(𝒗ᵒˡᵈ,𝒉ᵒˡᵈ,𝒉ˈᵒˡᵈ)
  //
  /*######################################################################*/

  cx_double p = this -> PsiNew_over_PsiOld(real_config, flipped_real_site,
                                                       shadow_ket, flipped_ket_site,
                                                       shadow_bra, flipped_bra_site,
                                                       option);
  if(p.imag() != 0.0){

    std::cerr << " ##ValueError: the imaginary part of the Metropolis-Hastings acceptance probability must be zero!" << std::endl;
    std::cerr << "   Failed to compute the Metropolis-Hastings acceptance probability." << std::endl;
    std::abort();

  }

  return p.real();

}


void quasi_uRBM :: LocalOperators(const Mat <int>& real_config, const Mat <int>& shadow_ket, const Mat <int>& shadow_bra) {

  /*#########################################################################################*/
  //  Calculates the local operators associated to the variational parameters
  //  𝛂 on the sampled enlarged quantum configuration |𝒗ⁿᵉʷ 𝒉ⁿᵉʷ 𝒉ˈⁿᵉʷ⟩.
  //  In the case of the (quasi)-uRBM 𝒜𝓃𝓈𝒶𝓉𝓏 the local parameters are 𝓞(𝒗,𝒉)
  //
  //        • η𝒿 ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒗) = 𝑣𝒿 • 𝑣𝒿+𝟣
  //        • ρ𝒿 ‹--› 𝓞(𝒗,𝒉) = 𝓞(𝒉) = 𝒽𝒿 • 𝒽𝒿+𝟣      𝓞(𝒗,𝒉ˈ) = 𝓞(𝒉ˈ) = 𝒽ˈ𝒿 • 𝒽ˈ𝒿+𝟣
  //        • ω𝒿 ‹--› 𝓞(𝒗,𝒉) = 𝒽𝒿 • 𝑣𝒿                𝓞(𝒗,𝒉ˈ) = 𝒽ˈ𝒿 • 𝑣𝒿
  //
  //  It is important to note that in the 𝓈ℎ𝒶𝒹ℴ𝓌 wave function the local operators
  //  (which are a geometric properties of the wave function itself) related to
  //  the 𝓈𝒽𝒶𝒹ℴ𝓌-𝓈𝒽𝒶𝒹ℴ𝓌 interactions and the 𝓇ℯ𝒶𝑙-𝓈𝒽𝒶𝒹ℴ𝓌 interaction, respectively
  //  depend also on the auxiliary variables, and not only on the actual quantum degrees
  //  of freedom of the system.
  //  These operators are necessary to compute the Quantum Geometric Tensor and the Gradient
  //  during the stochastic optimization procedure.
  //  We remember that in the 𝓈ℎ𝒶𝒹ℴ𝓌 case the local operators are real.
  /*#########################################################################################*/

  //Check on the lattice dimensionality
  // |𝒗⟩
  if(real_config.n_rows != 1 || real_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the 𝓇ℯ𝒶𝑙 configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // |𝒉⟩
  if(shadow_ket.n_rows != 1 || shadow_ket.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the ket configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }
  // ⟨𝒉ˈ|
  if(shadow_bra.n_rows != 1 || shadow_bra.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the bra configuration does not match with the number of 𝓇ℯ𝒶𝑙 variables" << std::endl;
    std::cerr << "              or with the lattice dimensionality for this 𝒜𝓃𝓈𝒶𝓉𝓏 (𝖽 = 𝟏)." << std::endl;
    std::cerr << "   Failed to compute the local operators 𝓞(𝒗,𝒉) and 𝓞(𝒗,𝒉ˈ)." << std::endl;
    std::abort();

  }

  //Local operators for the 𝓇ℯ𝒶𝑙-𝓇ℯ𝒶𝑙 interactions strength 𝛈
  for(int j = 0; j < _L; j++){

    _LocalOperators.at(j, 0) = double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));
    _LocalOperators.at(j, 1) = double(real_config.at(0, j) * real_config.at(0, (j + 1) % _L));

  }

  //Local operators for the 𝓈ℎ𝒶𝒹ℴ𝓌-𝓈ℎ𝒶𝒹ℴ𝓌 interactions strength 𝛒
  for(int j = 0; j < _L; j++){

    _LocalOperators.at(_L + j, 0) = double(shadow_ket.at(0, j) * shadow_ket.at(0, (j + 1) % _L));
    _LocalOperators.at(_L + j, 1) = double(shadow_bra.at(0, j) * shadow_bra.at(0, (j + 1) % _L));

  }

  //Local operators for the 𝓇ℯ𝒶𝑙-𝓈ℎ𝒶𝒹ℴ𝓌 interactions strength 𝒘
  for(int j = 0; j < _L; j++){

    _LocalOperators.at(2 + _L + j, 0) = double(real_config.at(0, j) * shadow_ket.at(0, j));
    _LocalOperators.at(2 + _L + j, 1) = double(real_config.at(0, j) * shadow_bra.at(0, j));

  }

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
