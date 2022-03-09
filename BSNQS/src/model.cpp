#ifndef __MODEL__
#define __MODEL__


/***************************************************************************************************************/
/******************************************  Quantum Hamiltonians  *********************************************/
/***************************************************************************************************************/
/*

  We create several models in order to represent the Many-Body Hamiltonians we want to study.
  In particular, we define the Hamiltonians for discrete Strongly-Correlated quantum systems,
  on both 𝟏 and 𝟐 dimensional lattice 𝚲 ϵ ℤᵈ.
  These classes describe the Hamiltonian operator, in particular its matrix elements, which
  are calculated and stored in a clever way, by searching which configurations of the total
  Hilbert space are connected by the local operators making up the Hamiltonian.
  More to the point, during a Monte Carlo procedure the quantum averages of the system
  properties are computed as

            ⟨𝓞⟩ = ⟨Ψ(𝓥)| Ô |Ψ(𝓥)⟩ = Σ𝒮 𝒪ˡᵒᶜ(𝒮) |Ψ(𝒮,𝓥)|^2
            𝒪ˡᵒᶜ(𝒮) = Σ𝒮ˈ ⟨𝒮| 𝒪 |𝒮ˈ⟩ • (Φ(𝒮ˈ,𝒉,𝛂) / Φ(𝒮,𝒉,𝛂))

  It is obviously not convenient to calculate all the matrix elements present in the sum defined above,
  but rather to consider only those for which

            ⟨𝒮| Ô |𝒮'⟩ ≠ 0

  In this case, we will say that the configuration |𝒮'⟩ is connected to the configuration |𝒮⟩;
  In the models we want to study the number of non-zero matrix elements (the connections) related to
  the various quantum observables of interest is typically small, and the task of the C++ classes
  defined below is precisely to manage the connections and the associated set of |𝒮'⟩ that are
  needed in the calculation of the instantaneous properies during the optimization via Variational
  Monte Carlo, once a system configuration |𝒮⟩ is sampled.
  The characteristics of the connections and of the | 𝒮'⟩ depend on the observable that we want to
  measure and the particular quantum model that we are studying.

  N̲O̲T̲E̲: so far, we’ve only implemented one class of Hamiltonians related to Lattice Spin Systems (𝐋𝐒𝐒).
        Other types of Discrete Strongly Correlated systems can be implemented in the future.

*/
/***************************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etc…
#include <cstdlib>  // <-- std::abort()
#include <complex>  // <-- std::complex<>, .real(), .imag()
#include <armadillo>  // <-- arma::Mat, arma::Col, arma::field


using namespace arma;


/*################################*/
/*  𝐒𝐏𝐈𝐍 𝐇𝐀𝐌𝐈𝐋𝐓𝐎𝐍𝐈𝐀𝐍 𝐈𝐍𝐓𝐄𝐑𝐅𝐀𝐂𝐄  */
/*################################*/
class SpinHamiltonian {

  protected:

    //System features
    const unsigned int _Nspin;  //Number of quantum spins
    unsigned int _LatticeDim;  //Dimensionality of the lattice 𝚲 ϵ ℤᵈ
    bool _pbc;  //Periodic Boundary Conditions
    const std::complex <double> _i;  //The imaginary unit 𝑖

    //Matrix representation of the spin observable operators
    field <Row <std::complex <double>>> _Connections;  //Non-zero Matrix Elements (i.e. the connections) of the spin observable operators
    field <field <Mat <int>>> _StatePrime;  //List of the flipped-spin lattice sites associated to each observables connections

  public:

    //Constructor and Destructor
    SpinHamiltonian(unsigned int n_spin, bool pbc=true) : _Nspin(n_spin), _pbc(pbc), _i(0.0, 1.0) {}  //Base constructor of a spin system
    virtual ~SpinHamiltonian() = default;  //Necessary for dynamic allocation

    //Virtual functions
    virtual void Quench(double) = 0;  //Quantum quench of a Hamiltonian coupling for real time dynamics
    virtual void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <Row <std::complex <double>>>&) = 0;  //Finds the connections for a given spin configuration |𝒮⟩
    virtual unsigned int MinFlips() const = 0;  //Returns the minimum number of spins to try to move during the MCMC

    //Access functions
    inline unsigned int n_spin() const {return _Nspin;}  //Returns the number of quantum degrees of freedom
    inline unsigned int dimensionality() const {return _LatticeDim;}  //Returns the lattice dimensionality 𝖽
    inline bool if_pbc() const {return _pbc;}  //Returns true if periodic boundary conditions are used on the system
    inline std::complex <double> i() const {return _i;}  //Returns the value of the imaginary unit
    inline field <Row <std::complex <double>>> get_connections() const {return _Connections;}  //Returns the list of connections
    inline field <field <Mat <int>>> all_state_prime() const {return _StatePrime;}  //Returns all the |𝒮'⟩ connected to the current configuration |𝒮⟩ of the system
    inline Row <std::complex <double>> EnergyConn() const {return _Connections(0);}  //Returns the list of connections related to ⟨Ĥ⟩
    inline Row <std::complex <double>> SxConn() const {return _Connections(1);}  //Returns the list of connections related to ⟨σ̂ˣ⟩
    inline Row <std::complex <double>> SyConn() const {return _Connections(2);}  //Returns the list of connections related to ⟨σ̂ʸ⟩
    inline Row <std::complex <double>> SzConn() const {return _Connections(3);}  //Returns the list of connections related to ⟨σ̂ᶻ⟩
    /*
    inline Row <std::complex <double>> SxSxConn() const {return _Connections(4);}  //Returns the list of connections related to ⟨σ̂ˣσ̂ˣ⟩
    inline Row <std::complex <double>> SySyConn() const {return _Connections(5);}  //Returns the list of connections related to ⟨σ̂ʸσ̂ʸ⟩
    inline Row <std::complex <double>> SzSzConn() const {return _Connections(6);}  //Returns the list of connections related to ⟨σ̂ᶻσ̂ᶻ⟩
    inline Row <std::complex <double>> SxSyConn() const {return _Connections(7);}  //Returns the list of connections related to ⟨σ̂ˣσ̂ʸ⟩
    inline Row <std::complex <double>> SxSzConn() const {return _Connections(8);}  //Returns the list of connections related to ⟨σ̂ˣσ̂ᶻ⟩
    inline Row <std::complex <double>> SySzConn() const {return _Connections(9);}  //Returns the list of connections related to ⟨σ̂ʸσ̂ᶻ⟩
    */

};


/*###########################################*/
/*  𝐓𝐑𝐀𝐍𝐒𝐕𝐄𝐑𝐒𝐄 𝐅𝐈𝐄𝐋𝐃 𝐈𝐒𝐈𝐍𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  */
/*##########################################*/
class Ising1d : public SpinHamiltonian {

  private:

    //Coupling constants (real valued)
    const double _J;  //σ̂ᶻ-σ̂ᶻ exchange interaction
    double _h;  //Transverse magnetic field

  public:

    //Constructor and Destructor
    Ising1d(unsigned int, double, double j=1.0, bool pbc=true);
    ~Ising1d(){}

    //Access functions
    inline double J() const {return _J;}  //Returns the σ̂ᶻ-σ̂ᶻ exchange interaction
    inline double h() const {return _h;}  //Returns the transverse magnetic field

    //Modifier functions
    void Quench(double);
    void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <Row <std::complex <double>>>&);
    unsigned int MinFlips() const {return 1;}

};


/*###############################*/
/*  𝐇𝐄𝐈𝐒𝐄𝐍𝐁𝐄𝐑𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  */
/*##############################*/
class Heisenberg1d : public SpinHamiltonian {

  private:

    //Coupling constants (real valued)
    const double _h;  //External magnetic field
    const double _Jx;  //σ̂ˣ-σ̂ˣ exchange interaction
    const double _Jy;  //σ̂ʸ-σ̂ʸ exchange interaction
    double _Jz;  //σ̂ᶻ-σ̂ᶻ exchange interaction

  public:

    //Constructor and Destructor
    Heisenberg1d(unsigned int, double hfield=0.0, double jx=-1.0, double jy=-1.0, double jz=-1.0, bool pbc=true);
    ~Heisenberg1d(){}

    //Access functions
    inline double h() const {return _h;}  //Returns the external magnetic field
    inline double Jx() const {return _Jx;}  //Returns the σ̂ˣ-σ̂ˣ exchange interaction
    inline double Jy() const {return _Jy;}  //Returns the σ̂ʸ-σ̂ʸ exchange interaction
    inline double Jz() const {return _Jz;}  //Returns the σ̂ᶻ-σ̂ᶻ exchange interaction

    //Modifier functions
    void Quench(double);
    void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <Row <std::complex <double>>>&);
    unsigned int MinFlips() const {return 2;}

};


/*******************************************************************************************************************************/
/******************************************  𝐓𝐑𝐀𝐍𝐒𝐕𝐄𝐑𝐒𝐄 𝐅𝐈𝐄𝐋𝐃 𝐈𝐒𝐈𝐍𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  *******************************************/
/*******************************************************************************************************************************/
Ising1d :: Ising1d(unsigned int n_spin, double h_field, double j, bool pbc)
         : SpinHamiltonian(n_spin, pbc), _J(j), _h(h_field) {

  /*############################################################################################################################*/
  //  Creates the Hamiltonian operator for the 𝐓𝐅𝐈 model on a 𝐝 = 𝟏 lattice (𝟏𝐝 Quantum Chain)
  //
  //        Ĥ = -h•Σⱼσ̂ⱼˣ - J•Σⱼₖσ̂ⱼᶻσ̂ₖᶻ
  //
  //  with j,k n.n., i.e. k = j + 1, on the computational basis
  //
  //        |𝒮⟩ = |σᶻ𝟣 σᶻ𝟤 … σᶻ𝖭⟩
  //
  //  The observables connections we want to measure in our stochastic
  //  framework are
  //
  //        _Connections(0) ‹--›  ⟨𝒮|  Ĥ  |𝒮'⟩
  //        _Connections(1) ‹--›  ⟨𝒮|  σ̂ˣ |𝒮'⟩
  //        _Connections(2) ‹--›  ⟨𝒮|  σ̂ʸ |𝒮'⟩
  //        _Connections(3) ‹--›  ⟨𝒮|  σ̂ᶻ |𝒮'⟩
  //        _Connections(4) ‹--›  ⟨𝒮| σ̂ˣσ̂ˣ |𝒮'⟩
  //        _Connections(5) ‹--›  ⟨𝒮| σ̂ʸσ̂ʸ |𝒮'⟩
  //        _Connections(6) ‹--›  ⟨𝒮| σ̂ᶻσ̂ᶻ |𝒮'⟩
  //        _Connections(7) ‹--›  ⟨𝒮| σ̂ˣσ̂ʸ |𝒮'⟩
  //        _Connections(8) ‹--›  ⟨𝒮| σ̂ˣσ̂ᶻ |𝒮'⟩
  //        _Connections(9) ‹--›  ⟨𝒮| σ̂ʸσ̂ᶻ |𝒮'⟩
  //
  //
  //  N̲O̲T̲E̲: we find instructive to explain the procedure that leads to the determination
  //        of the connections and of the configurations set |𝒮'⟩ for the quantum average of
  //        the energy of this model; the following arguments are easily extended to the
  //        other observable and to the other lattice models.
  //
  //  In the evaluation of the Hamiltonian operator matrix elements we have the
  //  following situation that greatly simplifies the calculation of the connections:
  //  the terms related to the σ̂ᶻ-σ̂ᶻ interaction connects only the same configuration,
  //  i.e. |𝒮⟩ = |𝒮'⟩; therefore there is only one non-zero-matrix element for
  //  this piece of Hamiltonian, which we save in the variable 𝐞𝐧𝐞𝐫𝐠𝐲_𝐜𝐨𝐧𝐧(𝟎) below and
  //  which we have to recalculate every time a new configuration |𝒮⟩ is sampled in the MCMC.
  //  When |𝒮'⟩ ≠ |𝒮⟩ a non-zero matrix element is obtained only when |𝒮'⟩ is
  //  identical to |𝒮⟩, less than a single flipped spin in a certain position 𝜈;
  //  In this case the term of the Hamiltonian related to the
  //  transverse field σ̂𝜈ˣ flips that spin on the right, giving
  //
  //        -h•⟨𝒮| σ̂𝜈ˣ |𝒮'⟩ = -h
  //
  //  while all the other terms are zero.
  //  In order to manage more easily the calculation of these non-zero matrix elements,
  //  rather than keeping in memory the list of all possible configurations in which the system
  //  can be found (which would be 2ᴺ), only the position of the flipped spin
  //  in the various configurations |𝒮'⟩ of the sum in the local energy is kept in memory in _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞(𝟎).
  //  In practice _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞(𝟎) represents all the possible configurations
  //  |𝒮'⟩ in the Σ𝒮' ⟨𝒮| Ĥ |𝒮'⟩ for which the matrix elements
  //
  //        ⟨𝒮| Ĥ |𝒮'⟩ ≠ 𝟢
  //
  //  including the case |𝒮'⟩ = |𝒮⟩ which is saved in position 𝟢 of _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞(𝟎,𝟎),
  //  respect of which no spin is flipped.
  //  These considerations extend easily to all other non-zero matrix elements related
  //  to each local observable we want to measure; therefore, to summarize, each element
  //  _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞(𝒋) will be the representation, as explained above, of all the primate
  //  configurations |𝒮'⟩ related to the various non-zero matrix elements of the appropriate
  //  observable among those listed above, collected in each _𝐂𝐨𝐧𝐧𝐞𝐜𝐭𝐢𝐨𝐧𝐬(𝒋).
  //
  //  In other words, each element of _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞 is nothing more than the list of |𝒮'⟩
  //  (each of them represented by a matrix of 𝐟𝐥𝐢𝐩𝐩𝐞𝐝_𝐬𝐢𝐭𝐞 indices as explained in 𝐚𝐧𝐬𝐚𝐭𝐳.𝐜𝐩𝐩)
  //  that identify the locations of flipped spin lattice sites in |𝒮'⟩ compared to the current configuration
  //  of the system |𝒮⟩, and such that ⟨𝒮| Ô |𝒮'⟩ ≠ 𝟢.
  //  Except in 𝐝 = 𝟏, these indices will be multidimensional indices, e.g. in the 𝐝 = 𝟐 case
  //  a generic element |𝒮'⟩ in the row _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞(𝒋) will be of the type
  //
  //        ⌈  0    0  ⌉   -->  Flip the 1st spin of the 2d lattice
  //        |  1    5  |   -->  ....
  //        |  ......  |   -->  ....
  //        |  ......  |   -->  ....
  //        |  ......  |   -->  ....
  //        ⌊  4    4  ⌋   -->  ....
  //
  //  For example, assuming that we are in the spin configuration
  //
  //               | +1 -1 +1     …      +1  \
  //               | +1 -1 -1 +1 -1 +1 … -1   \
  //        |𝒮⟩ =    :  :  :      …      -1    \
  //               | :  :  :      …      -1    /
  //               | -1 +1 +1 +1 -1 …    -1   /
  //               | :  :  :      …      +1  /
  //
  //  the newly particular defined |𝒮'⟩ in the row _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞(𝒋) would represent the configuration
  //
  //               | -1 -1 +1 … … … …  … +1  \
  //               | +1 -1 -1 +1 -1 -1 … -1   \
  //        |𝒮'⟩ =   :  :  :      …      -1    \
  //               | :  :  :      …      -1    /
  //               | -1 +1 +1 +1 +1 …    -1   /
  //               | :  :  :      …      +1  /
  //
  //  As mentioned above, _𝐒𝐭𝐚𝐭𝐞𝐏𝐫𝐢𝐦𝐞 will be, from a linear algebra point of view, a kind of tensor,
  //  with as many rows as there are observable that we want to measure during the Monte Carlo optimization;
  //  each row will have a certain number of matrices (columns) representing each of the |𝒮'⟩ configurations
  //  for which the connection is not zero; we remember once again that these matrices are nothing more
  //  than the list of lattice sites where the spins have been flipped compared to the configuration |𝒮⟩ in which the
  //  system is located.
  //  At this point the structure of _𝐂𝐨𝐧𝐧𝐞𝐜𝐭𝐢𝐨𝐧𝐬 is obvious: each row (Row) will contain the list of non-zero
  //  connections, and will have as many elements as the number of |𝒮'⟩ described above.
  //  Schematically:
  //
  //                                         ⌈                                                           ⌉
  //        _Connections(0) = EnergyConn ‹-› |  ⟨𝒮|Ĥ|𝒮'1⟩,  ⟨𝒮|Ĥ|𝒮'2⟩,  ⟨𝒮|Ĥ|𝒮'3⟩,  ……………,  ⟨𝒮|Ĥ|𝒮'ℕ⟩  |
  //                                         ⌊                                                           ⌋
  //
  //                                                               ⇕
  //
  //                                         ⌈                                                                               ⌉
  //        _StatePrime(0) =  {|𝒮'⟩}  ‹-›    |  |𝒮'1⟩ = Mat( • | • ),  |𝒮'2⟩ = Mat( • | • ),  ……………,  |𝒮'ℕ⟩ = Mat( • | • )  |
  //                                         ⌊                                                                               ⌋
  //
  //  and so on for the other observable connections.
  /*############################################################################################################################*/

  //Information
  std::cout << "#Creation of the model Hamiltonian" << std::endl;

  //Data-members initialization
  _LatticeDim = 1;  //𝟏𝐝 Quantum Chain
  _Connections.set_size(4, 1);
  _StatePrime.set_size(4, 1);
  _StatePrime(0, 0).set_size(1, _Nspin+1);  //List of |𝒮'⟩ for the energy
  _StatePrime(1, 0).set_size(1, _Nspin);  //List of |𝒮'⟩ for σ̂ˣ
  _StatePrime(2, 0).set_size(1, _Nspin);  //List of |𝒮'⟩ for σ̂ʸ
  _StatePrime(3, 0).set_size(1, 1);  //List of |𝒮'⟩ for σ̂ᶻ

  //Function variables
  Row <std::complex <double>> energy_conn(_Nspin+1, fill::zeros);  //The first element corresponds to the case |𝒮'⟩ = |𝒮⟩
  Row <std::complex <double>> sigmax_conn(_Nspin, fill::zeros);  //Storage variable

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| Ĥ |𝒮'⟩
  _StatePrime(0, 0)(0, 0).reset();  //empty flipped_site matrix, i.e. |𝒮'⟩ = |𝒮⟩, diagonal term
  for(unsigned int j_flipped=1; j_flipped<_Nspin+1; j_flipped++){

    _StatePrime(0, 0)(0, j_flipped).set_size(1, 1);  //|𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped-1
    _StatePrime(0, 0)(0, j_flipped)(0, 0) = j_flipped-1;
    energy_conn(j_flipped) = -_h;  //non-diagonal term

  }
  _Connections(0, 0) = energy_conn;

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| σ̂ⱼ |𝒮'⟩
  for(unsigned int j_flipped=0; j_flipped<_Nspin; j_flipped++){

    _StatePrime(1, 0)(0, j_flipped).set_size(1, 1);  //σ̂ˣ, |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped
    _StatePrime(1, 0)(0, j_flipped)(0, 0) = j_flipped;
    sigmax_conn(j_flipped) = 1.0;  //only non-diagonal term

    _StatePrime(2, 0)(0, j_flipped).set_size(1, 1);  //σ̂ʸ, |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped
    _StatePrime(2, 0)(0, j_flipped)(0, 0) = j_flipped;

  }
  _StatePrime(3, 0)(0, 0).reset();  //σ̂ᶻ, empty flipped_site matrix, i.e. |𝒮'⟩ = |𝒮⟩, only diagonal term
  _Connections(1, 0) = sigmax_conn;

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| σ̂ⱼσ̂ₖ |𝒮'⟩
  /*
    ..........
    ..........
    ..........
  */

  //Indicates the created model
  std::cout << " Transverse Field Ising model in 𝐝 = 𝟏 with " << _Nspin << " Quantum spins in h = " << _h << " magnetic field." << std::endl;
  std::cout << " Coupling constant of the TFI model:" << std::endl;
  std::cout << " \tJ = " << _J << std::endl << std::endl;

}


void Ising1d :: Quench(double hf) {

  /*##################################################*/
  //  Introduces nontrivial quantum dynamics
  //  by means of an instantaneous change in the
  //  transverse field from _h to hf.
  //  Due to this quantum quench certain observable
  //  needs to be modified, such as the local energy.
  /*##################################################*/

  _h = hf;

  //Recalculate necessary pre-computed connections
  for(unsigned int mel=0; mel<_Nspin; mel++)
    _Connections(0, 0)(mel+1) = -_h;

}


void Ising1d :: FindConn(const Mat <int>& current_config, field <field <Mat <int>>>& state_prime, field <Row <std::complex <double>>>& connections) {

  /*###################################################################################*/
  //  Finds the non-zero matrix elements of the spin observables
  //  on a given sampled spin configuration |𝒮⟩ named 𝐜𝐮𝐫𝐫𝐞𝐧𝐭_𝐬𝐭𝐚𝐭𝐞.
  //  In particular it searches all the |𝒮'⟩ such that
  //
  //        ⟨𝒮| Ô |𝒮'⟩ ≠ 𝟢
  //
  //  The configuration |𝒮'⟩ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |𝒮⟩ as abundantly described above.
  //
  //  Note that not all the observable connections change: for example they should
  //  not be recalculated for the transverse polarization σ̂ₓ, for which the constructor
  //  calculations are sufficient!
  /*###################################################################################*/

  //Check on the lattice dimensionality
  if(current_config.n_rows != 1 || current_config.n_cols != _Nspin){

    std::cerr << " ##SizeError: the matrix representation of the quantum spin configuration does not match with the dimensionality of the system lattice." << std::endl;
    std::cerr << "   The system lives on a " << _LatticeDim << " dimensional lattice and is composed of " << _Nspin << " quantum spins." << std::endl;
    std::cerr << "   Failed to find the observable connections." << std::endl;
    std::abort();

  }

  //Assign pre-computed connections and |𝒮'⟩
  connections = _Connections;
  state_prime = _StatePrime;

  //Computing σ̂ᶻ-σ̂ᶻ interaction part for the local energy
  connections(0, 0)(0) = 0.0;
  for(unsigned int j = 0; j < (_Nspin-1); j++)
    connections(0, 0)(0) += double(current_config(0, j) * current_config(0, j+1));
  if(_pbc)
    connections(0, 0)(0) += double(current_config(0, _Nspin-1) * current_config(0, 0));
  connections(0, 0)(0) *= -_J;

  //Computing the other connections
  Row <std::complex <double>> sigmay_conn(_Nspin, fill::zeros);
  Row <std::complex <double>> sigmaz_conn(1, fill::zeros);
  for(unsigned int j = 0; j < _Nspin; j++){

    sigmay_conn(j) = _i * double(current_config(0, j));
    sigmaz_conn(0) += double(current_config(0, j));

  }
  connections(2, 0) = sigmay_conn;
  connections(3, 0) = sigmaz_conn;

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/*************************************************  𝐇𝐄𝐈𝐒𝐄𝐍𝐁𝐄𝐑𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  ************************************************/
/*******************************************************************************************************************************/
Heisenberg1d :: Heisenberg1d(unsigned int n_spin, double h_field, double jx, double jy, double jz, bool pbc)
              : SpinHamiltonian(n_spin, pbc), _h(h_field),  _Jx(jx), _Jy(jy), _Jz(jz) {

  /*#################################################################################################*/
  //  Creates the Hamiltonian operator for the Heisenberg model on a 𝐝 = 𝟏 lattice (𝟏𝐝 Quantum Chain)
  //
  //        Ĥ = -Σⱼₖ(Jˣ•σ̂ˣⱼσ̂ˣₖ + Jʸ•σ̂ʸⱼσ̂ʸₖ + Jᶻ•σ̂ᶻⱼσ̂ᶻₖ) - h•Σⱼσ̂ᶻⱼ
  //        k = j + 1 (and pbc)
  //
  //  on the computational basis
  //
  //        |𝒮⟩ = |σᶻ𝟣 σᶻ𝟤 … σᶻ𝖭⟩
  //
  //  For this model we can have various combinations regarding the value of the
  //  coupling constants {h, Jˣ, Jʸ, Jᶻ}, catalogued with the following models:
  //
  //        • Jˣ ≠ Jʸ ≠ Jᶻ ≠ 𝟢 & h ≠ 𝟢      ‹--›  𝐗𝐘𝐙 Model
  //        • Jˣ = Jʸ = J & Jᶻ = Δ & h ≠ 𝟢  ‹--›  𝐗𝐗𝐙 Model
  //        • Jˣ = Jʸ = Jᶻ = J & h ≠ 𝟢      ‹--›  𝐗𝐗𝐗 Model
  //        • Jˣ = Jʸ = Jᶻ = -1 & h = 𝟢     ‹--›  𝐀𝐅𝐇 Model (AntiFerromagnetic Heisenberg Model)
  //
  //  where the 𝐀𝐅𝐇 is the default in this implementation.
  //  The local spin observables we want to measure in our stochastic
  //  framework are
  //
  //        _Connections(0) ‹--›  ⟨𝒮|  Ĥ  |𝒮'⟩
  //        _Connections(1) ‹--›  ⟨𝒮 |  σ̂ˣ |𝒮'⟩
  //        _Connections(2) ‹--›  ⟨𝒮 |  σ̂ʸ |𝒮'⟩
  //        _Connections(3) ‹--›  ⟨𝒮 |  σ̂ᶻ |𝒮'⟩
  //        _Connections(4) ‹--›  ⟨𝒮 | σ̂ˣσ̂ˣ |𝒮'⟩
  //        _Connections(5) ‹--›  ⟨𝒮 | σ̂ʸσ̂ʸ |𝒮'⟩
  //        _Connections(6) ‹--›  ⟨𝒮 | σ̂ᶻσ̂ᶻ |𝒮'⟩
  //        _Connections(7) ‹--›  ⟨𝒮 | σ̂ˣσ̂ʸ |𝒮'⟩
  //        _Connections(8) ‹--›  ⟨𝒮 | σ̂ˣσ̂ᶻ |𝒮'⟩
  //        _Connections(9) ‹--›  ⟨𝒮 | σ̂ʸσ̂ᶻ |𝒮'⟩
  //
  //  The considerations relating to the non-zero matrix elements for these observables
  //  are analogous to the previous model, but in this case even the primate configurations
  //  where two adjacent spins are flipped contribute to the non-zero connections.
  /*#################################################################################################*/

  //Information
  std::cout << "#Creation of the model Hamiltonian" << std::endl;

  //Data-members initialization
  _LatticeDim = 1;  //𝟏𝐝 Quantum Chain
  _Connections.set_size(4, 1);
  _StatePrime.set_size(4, 1);

  //Function variables
  Row <std::complex <double>> energy_conn;  //Storage variable

  if(_pbc){

    _StatePrime(0, 0).set_size(1, _Nspin+1);  //List of |𝒮'⟩ for the energy
    energy_conn.set_size(_Nspin+1);  //The first element corresponds to the case |𝒮'⟩ = |𝒮⟩

  }
  else{

    _StatePrime(0, 0).set_size(1, _Nspin);  //List of |𝒮'⟩ for the energy
    energy_conn.set_size(_Nspin);  //The first element corresponds to the case |𝒮'⟩ = |𝒮⟩

  }
  _StatePrime(1, 0).set_size(1, _Nspin);  //List of |𝒮'⟩ for σ̂ˣ
  _StatePrime(2, 0).set_size(1, _Nspin);  //List of |𝒮'⟩ for σ̂ʸ
  _StatePrime(3, 0).set_size(1, 1);  //List of |𝒮'⟩ for σ̂ᶻ
  Row <std::complex <double>> sigmax_conn(_Nspin, fill::zeros);

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| Ĥ |𝒮'⟩
  _StatePrime(0, 0)(0, 0).reset();  //empty flipped_site matrix, i.e. |𝒮'⟩ = |𝒮⟩, diagonal term
  for(unsigned int j_flipped=1; j_flipped<_Nspin; j_flipped++){

    _StatePrime(0, 0)(0, j_flipped).set_size(2, 1);  //|𝒮'⟩ ≠ |𝒮⟩ due to two adjacent flipped spin at lattice site j_flipped-1 & j_flipped
    _StatePrime(0, 0)(0, j_flipped)(0, 0) = j_flipped-1;
    _StatePrime(0, 0)(0, j_flipped)(1, 0) = j_flipped;
    energy_conn(j_flipped) = -_Jx;  //non-diagonal term related to the σ̂ˣ-σ̂ˣ exchange interaction

  }
  if(_pbc){

    _StatePrime(0, 0)(0, _Nspin).set_size(2, 1);  //|𝒮'⟩ ≠ |𝒮⟩ due to two adjacent flipped spin at the edge of the lattice site
    _StatePrime(0, 0)(0, _Nspin)(0, 0) = _Nspin-1;
    _StatePrime(0, 0)(0, _Nspin)(1, 0) = 0;
    energy_conn(_Nspin) = -_Jx;

  }
  _Connections(0, 0) = energy_conn;

  //Pre-computed connections and associated |S'⟩ definitions for ⟨𝒮| σ̂ⱼ |𝒮'⟩
  for(unsigned int j_flipped=0; j_flipped<_Nspin; j_flipped++){

    _StatePrime(1, 0)(0, j_flipped).set_size(1, 1);  //σ̂ˣ, |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped
    _StatePrime(1, 0)(0, j_flipped)(0, 0) = j_flipped;
    sigmax_conn(j_flipped) = 1.0;  //only non-diagonal term

    _StatePrime(2, 0)(0, j_flipped).set_size(1, 1);  //σ̂ʸ, |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped
    _StatePrime(2, 0)(0, j_flipped)(0, 0) = j_flipped;

  }
  _StatePrime(3, 0)(0, 0).reset();  //σ̂ᶻ, empty flipped_site matrix, i.e. |𝒮'⟩ = |𝒮⟩, only diagonal term
  _Connections(1, 0) = sigmax_conn;

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| σ̂ⱼσ̂ₖ |𝒮'⟩
  /*
    ..........
    ..........
    ..........
  */

  //Indicates the created model
  if(_Jx!=_Jy &&  _Jy!=_Jz && _Jz!=_Jx)
    std::cout << " XYZ model in 𝐝 = 𝟏 with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;
  else if(_Jx==_Jy &&  _Jy!=_Jz && _Jz!=_Jx)
    std::cout << " XXZ model in 𝐝 = 𝟏 with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;
  else if(_Jx==_Jy &&  _Jy==_Jz && _Jz==_Jx && _Jx!=-1.0)
    std::cout << " XXX model in 𝐝 = 𝟏 with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;
  else if(_Jx==-1.0 && _Jy==-1.0 && _Jz==-1.0)
    std::cout << " AntiFerromagnetic Heisenberg model in 𝐝 = 𝟏 with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;

  std::cout << " Coupling constants of the Heisenberg model:" << std::endl;
  std::cout << " \tJˣ = " << _Jx << std::endl;
  std::cout << " \tJʸ = " << _Jy << std::endl;
  std::cout << " \tJᶻ = " << _Jz << std::endl << std::endl;

}


void Heisenberg1d :: Quench(double jf) {

  /*##################################################*/
  //  Introduces nontrivial quantum dynamics
  //  by means of an instantaneous change in the
  //  σ̂ᶻ-σ̂ᶻ exchange interaction from _Jz to jf.
  //  Due to this quantum quench certain observable
  //  needs to be modified, such as the local energy.
  /*##################################################*/

  _Jz = jf;

}


void Heisenberg1d :: FindConn(const Mat <int>& current_config, field <field <Mat <int>>>& state_prime, field <Row <std::complex <double>>>& connections) {

  /*########################################################################################*/
  //  Finds the non-zero matrix elements of the spin observables
  //  on a given sampled configuration passed as the first argument of this function.
  //  In particular it searches all the |𝒮'⟩ such that
  //
  //        ⟨𝒮| Ô |𝒮'⟩ ≠ 𝟢
  //
  //  The configuration |𝒮'⟩ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |𝒮⟩.
  //  For example, in the evaluation of the Hamiltonian operator we have
  //  the following situation:
  //
  //        • |𝒮'⟩ = |𝒮⟩
  //          When no spin flips are performed we have
  //                    ⟨𝒮| σ̂ˣ[j]σ̂ˣ[j+1] |𝒮'⟩ = 𝟢    for all j
  //                    ⟨𝒮| σ̂ʸ[j]σ̂ʸ[j+1] |𝒮'⟩ = 𝟢    for all j
  //                    ⟨𝒮| σ̂ᶻ[j]σ̂ᶻ[j+1] |𝒮'⟩ = σᶻ[j]σᶻ[j+1]
  //                    ⟨𝒮| σ̂ᶻ[j] |𝒮'⟩ = σᶻ[j]
  //        • |𝒮'⟩ ≠ |𝒮⟩
  //          When only one spin is flipped in position 𝒿 we have
  //                    ⟨𝒮| σ̂ˣ[𝒿]σ̂ˣ[𝒿+1] |𝒮'⟩ = 𝟢    for all 𝒿
  //                    ⟨𝒮| σ̂ʸ[𝒿]σ̂ʸ[𝒿+1] |𝒮'⟩ = 𝟢    for all 𝒿
  //                    ⟨𝒮| σ̂ᶻ[𝒿]σ̂ᶻ[𝒿+1] |𝒮'⟩ = 𝟢    for all 𝒿
  //                    ⟨𝒮| σ̂ᶻ[𝒿] |𝒮'⟩ = 𝟢           for all 𝒿
  //        • |𝒮'⟩ ≠ |𝒮⟩
  //          When two spins are flipped in position 𝒿, 𝒿+1 we have
  //                    ⟨𝒮| σ̂ˣ[𝒿]σ̂ˣ[𝒿+1] |𝒮'⟩ = 1    for all 𝒿
  //                    ⟨𝒮| σ̂ʸ[𝒿]σ̂ʸ[𝒿+1] |𝒮'⟩ = -σᶻ[𝒿]σᶻ[j+𝒿]
  //                    ⟨𝒮| σ̂ᶻ[𝒿]σ̂ᶻ[𝒿+1] |𝒮'⟩ = 𝟢    for all 𝒿
  //                    ⟨𝒮| σ̂ᶻ[𝒿] |𝒮'⟩ = 𝟢           for all 𝒿
  //        • |𝒮'⟩ ≠ |𝒮⟩
  //          When more than two spins are flipped in position we have
  //                    ⟨𝒮| σ̂ˣ[j]σ̂ˣ[j+1] |𝒮'⟩ = 𝟢    for all j
  //                    ⟨𝒮| σ̂ʸ[j]σ̂ʸ[j+1] |𝒮'⟩ = 𝟢    for all j
  //                    ⟨𝒮| σ̂ᶻ[j]σ̂ᶻ[j+1] |𝒮'⟩ = 𝟢    for all j
  //                    ⟨𝒮| σ̂ᶻ[j] |𝒮'⟩ = 𝟢           for all j
  //
  //  The configuration |𝒮'⟩ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |𝒮⟩ as abundantly described above.
  //
  //  Note that not all the observable connections change: for example they should
  //  not be recalculated for the transverse polarization σ̂ₓ, for which the constructor
  //  calculations are sufficient!
  /*########################################################################################*/

  //Check on the lattice dimensionality
  if(current_config.n_rows != 1 || current_config.n_cols != _Nspin){

    std::cerr << " ##SizeError: the matrix representation of the quantum spin configuration does not match with the dimensionality of the system lattice." << std::endl;
    std::cerr << "   The system lives on a " << _LatticeDim << " dimensional lattice and is composed of " << _Nspin << " quantum spins." << std::endl;
    std::cerr << "   Failed to find the observable connections." << std::endl;
    std::abort();

  }

  //Assign pre-computed non-zero matrix elements and spin flips
  connections = _Connections;
  state_prime = _StatePrime;

  //|𝒮'⟩ = |𝒮⟩, diagonal term
  connections(0, 0)(0) = 0.0;
  std::complex <double> acc_J = 0.0;  //σ̂ᶻ-σ̂ᶻ interaction part of the Hamiltonian
  std::complex <double> acc_h = 0.0;  //interaction with the external magnetic field part of the Hamiltonian
  for(unsigned int j=0; j<=(_Nspin-2); j++){

    acc_J += double(current_config(0, j)*current_config(0, j+1));  //Computing the interaction part σ̂ᶻ-σ̂ᶻ
    acc_h += double(current_config(0, j));  //Computing the magnetic field interaction part

  }
  if(_pbc)
    connections(0, 0)(0) = -_Jz * (acc_J + double(current_config(0, _Nspin-1)*current_config(0, 0))) - _h * (acc_h + double(current_config(0, _Nspin-1)));
  else
    connections(0, 0)(0) = -_Jz * acc_J - _h * (acc_h + double(current_config(0, _Nspin-1)));

  //|𝒮'⟩ ≠ |𝒮⟩, non diagonal terms
  //Computing the interaction part σ̂ʸ-σ̂ʸ
  for(unsigned int j_flipped=1; j_flipped<_Nspin; j_flipped++)
    connections(0, 0)(j_flipped) += _Jy * double(current_config(0, state_prime(0, 0)(0, j_flipped)(0, 0))*current_config(0, state_prime(0, 0)(0, j_flipped)(1, 0)));
  if(_pbc)
    connections(0, 0)(_Nspin) += _Jy * double(current_config(0, state_prime(0, 0)(0, _Nspin)(0, 0))*current_config(0, state_prime(0, 0)(0, _Nspin)(1, 0)));

  //Computing the other connections
  Row <std::complex <double>> sigmay_conn(_Nspin, fill::zeros);
  Row <std::complex <double>> sigmaz_conn(1, fill::zeros);
  for(unsigned int j=0; j<_Nspin; j++){

    sigmay_conn(j) = _i * double(current_config(0, j));
    sigmaz_conn(0) += double(current_config(0, j));

  }
  connections(2, 0) = sigmay_conn;
  connections(3, 0) = sigmaz_conn;

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


#endif
