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


#ifndef __MODEL__
#define __MODEL__


/****************************************************************************************************************/
/*******************************************  𝑸𝒖𝒂𝒏𝒕𝒖𝒎 𝑯𝒂𝒎𝒊𝒍𝒕𝒐𝒏𝒊𝒂𝒏𝒔  **********************************************/
/***************************************************************************************************************/
/*

  We create several models in order to represent the Many-Body Hamiltonians we want to study.
  In particular, we define the Hamiltonians for discrete strongly-correlated quantum systems,
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
  the various quantum observables of interest is typically small (of 𝑂(𝖫), where 𝖫 is the size of the
  system), and the task of the C++ classes defined below is precisely to manage the connections and
  the associated set of |𝒮'⟩ that needed in the calculation of the instantaneous properies during the
  evolution via time-dependent Variational Monte Carlo, once a system configuration |𝒮⟩ is sampled.
  The characteristics of the connections and of the | 𝒮'⟩ depend on the observable that we want to
  measure and the particular quantum model that we are studying.

  N̲O̲T̲E̲: so far, we’ve only implemented one class of Hamiltonians related to lattice spin systems (𝐋𝐒𝐒).
        Other types of discrete strongly-correlated systems can be implemented in the future.

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
    const int _L;  //Number of quantum spins
    int _d;  //Dimensionality of the lattice 𝚲 ϵ ℤᵈ
    bool _PBCs;  //Periodic Boundary Conditions
    const cx_double _i;  //The imaginary unit 𝑖

    //Matrix representation of the non-diagonal observable operators
    field <cx_rowvec> _Connections;  //Non-zero matrix elements (i.e. the connections) of the spin non-diagonal observable operators
    field <field <Mat <int>>> _StatePrime;  //List of the flipped-spin lattice sites associated to each observables connections

  public:

    //Constructor and Destructor
    SpinHamiltonian(int n_spin, bool pbc=true) : _L(n_spin), _PBCs(pbc), _i(0.0, 1.0) {}  //Base constructor of a spin Hamiltonian
    virtual ~SpinHamiltonian() = default;  //Necessary for dynamic allocation

    //Virtual functions
    virtual void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <cx_rowvec>&) = 0;  //Finds the connections for a given configuration |𝒮⟩
    virtual int MinFlips() const = 0;  //Returns the minimum number of spins to try to move in the single bunch along the MCMC sweeps

    //Access functions
    inline int n_spin() const {return _L;}  //Returns the number of quantum degrees of freedom
    inline int dimensionality() const {return _d;}  //Returns the lattice dimensionality 𝖽
    inline bool if_PBCs() const {return _PBCs;}  //Returns true if periodic boundary conditions are imposed
    inline cx_double i() const {return _i;}  //Returns the value of the imaginary unit

    //Helpful in debugging
    inline field <cx_rowvec> get_connections() const {return _Connections;}  //Returns the list of connections
    inline field <field <Mat <int>>> all_state_prime() const {return _StatePrime;}  //Returns all the |𝒮'⟩ connected to the current configuration |𝒮⟩ of the system

};


  /*###########################################*/
 /*  𝐓𝐑𝐀𝐍𝐒𝐕𝐄𝐑𝐒𝐄 𝐅𝐈𝐄𝐋𝐃 𝐈𝐒𝐈𝐍𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  */
/*##########################################*/
class Ising1d : public SpinHamiltonian {

  private:

    //Coupling constants (real valued)
    const double _J;  //σ̂ᶻ-σ̂ᶻ exchange interaction 𝐽
    double _h;  //Transverse magnetic field 𝒉

  public:

    //Constructor and Destructor
    Ising1d(int, double, int, double J=1.0, bool pbc=true);
    ~Ising1d(){}

    //Access functions
    inline double J() const {return _J;}  //Returns the σ̂ᶻ-σ̂ᶻ exchange interaction 𝐽
    inline double h() const {return _h;}  //Returns the transverse magnetic field 𝒉

    //Modifier functions
    void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <cx_rowvec>&);
    int MinFlips() const {return 1;}

};


  /*###############################*/
 /*  𝐇𝐄𝐈𝐒𝐄𝐍𝐁𝐄𝐑𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  */
/*##############################*/
class Heisenberg1d : public SpinHamiltonian {

  private:

    //Coupling constants (real valued)
    const double _h;  //External magnetic field 𝒉
    const double _Jx;  //σ̂ˣ-σ̂ˣ exchange interaction 𝐽ˣ
    const double _Jy;  //σ̂ʸ-σ̂ʸ exchange interaction 𝐽ʸ
    double _Jz;  //σ̂ᶻ-σ̂ᶻ exchange interaction 𝐽ᶻ

  public:

    //Constructor and Destructor
    Heisenberg1d(int, int, double hfield=0.0, double Jx=-1.0, double Jy=-1.0, double Jz=-1.0, bool pbc=true);
    ~Heisenberg1d(){}

    //Access functions
    inline double h() const {return _h;}  //Returns the external magnetic field 𝒉
    inline double Jx() const {return _Jx;}  //Returns the σ̂ˣ-σ̂ˣ exchange interaction 𝐽ˣ
    inline double Jy() const {return _Jy;}  //Returns the σ̂ʸ-σ̂ʸ exchange interaction 𝐽ʸ
    inline double Jz() const {return _Jz;}  //Returns the σ̂ᶻ-σ̂ᶻ exchange interaction 𝐽ᶻ

    //Modifier functions
    void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <cx_rowvec>&);
    int MinFlips() const {return 2;}

};


/*******************************************************************************************************************************/
/******************************************  𝐓𝐑𝐀𝐍𝐒𝐕𝐄𝐑𝐒𝐄 𝐅𝐈𝐄𝐋𝐃 𝐈𝐒𝐈𝐍𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  *******************************************/
/*******************************************************************************************************************************/
Ising1d :: Ising1d(int n_spin, double h_field, int rank, double J, bool pbc)
         : SpinHamiltonian(n_spin, pbc), _J(J), _h(h_field) {

  /*############################################################################################################################*/
  //  Creates the Hamiltonian operator for the 𝐓𝐅𝐈 model on a 𝐝 = 𝟏 lattice (𝟏𝐝 quantum chain)
  //
  //        Ĥ = -h • Σⱼσ̂ⱼˣ - J • Σⱼₖσ̂ⱼᶻσ̂ₖᶻ
  //
  //  with j,k are nearest-neighbors indeces, i.e. k = j + 1, on the computational basis
  //
  //        |𝒮⟩ = |σᶻ𝟣 σᶻ𝟤 … σᶻ𝖭⟩.
  //
  //  The observables connections we want to measure in our stochastic
  //  framework are
  //
  //        _Connections(𝟢) ‹--›  ⟨𝒮|  Ĥ  |𝒮'⟩
  //        _Connections(𝟣) ‹--›  ⟨𝒮|  Σ̂ˣ |𝒮'⟩
  //
  //  with Σ̂ˣ = Σⱼ σ̂ⱼˣ the magnetization along the transverse field direction x.
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
  //  can be found (which would be 𝟤ᴸ), only the position of the flipped spin
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
  //        _Connections(𝟢) = EnergyConn ‹-› |  ⟨𝒮|Ĥ|𝒮'1⟩,  ⟨𝒮|Ĥ|𝒮'2⟩,  ⟨𝒮|Ĥ|𝒮'3⟩,  ……………,  ⟨𝒮|Ĥ|𝒮'ℕ⟩  |
  //                                         ⌊                                                           ⌋
  //
  //                                                               ⇕
  //
  //                                         ⌈                                                                               ⌉
  //        _StatePrime(𝟢) =  {|𝒮'⟩}  ‹-›    |  |𝒮'1⟩ = Mat( • | • ),  |𝒮'2⟩ = Mat( • | • ),  ……………,  |𝒮'ℕ⟩ = Mat( • | • )  |
  //                                         ⌊                                                                               ⌋
  //
  //  and so on for the other observable connections.
  /*############################################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Creation of the model Hamiltonian." << std::endl;

  //Data-members initialization
  _d = 1;  //𝟏𝐝 quantum chain
  _Connections.set_size(2, 1);
  _StatePrime.set_size(2, 1);
  _StatePrime.at(0, 0).set_size(1, _L + 1);  //List of |𝒮'⟩ for the energy
  _StatePrime.at(1, 0).set_size(1, _L);  //List of |𝒮'⟩ for Σ̂ˣ

  //Function variables
  cx_rowvec energy_conn(_L + 1, fill::zeros);  //The first element corresponds to the case |𝒮'⟩ = |𝒮⟩
  cx_rowvec SigmaX_conn(_L, fill::zeros);  //Storage variable

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| Ĥ |𝒮'⟩
  _StatePrime.at(0, 0).at(0, 0).reset();  //empty flipped_site matrix, i.e. |𝒮'⟩ = |𝒮⟩, diagonal term
  for(int j_flipped = 1; j_flipped < _L + 1; j_flipped++){

    _StatePrime.at(0, 0).at(0, j_flipped).set_size(1, 1);  // |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped - 1
    _StatePrime.at(0, 0).at(0, j_flipped).at(0, 0) = j_flipped - 1;
    energy_conn[j_flipped] = -_h;  //non-diagonal term

  }
  _Connections.at(0, 0) = energy_conn;

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| Σ̂ˣ |𝒮'⟩
  for(int j_flipped = 0; j_flipped < _L; j_flipped++){

    _StatePrime.at(1, 0).at(0, j_flipped).set_size(1, 1);  // |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped
    _StatePrime.at(1, 0).at(0, j_flipped).at(0, 0) = j_flipped;
    SigmaX_conn[j_flipped] = 1.0;  //only non-diagonal term

  }
  _Connections.at(1, 0) = SigmaX_conn;

  //Indicates the created model
  if(rank == 0){

    std::cout << " Transverse Field Ising model in 𝐝 = 𝟏 with " << _L << " quantum spins in 𝙝 = " << _h << " magnetic field and ";
    if(_PBCs) std::cout << "periodic boundary conditions." << std::endl;
    else std::cout << "open boundary conditions." << std::endl;
    std::cout << " Coupling constant of the TFI model → 𝐽 = " << _J << std::endl << std::endl;

  }

}


void Ising1d :: FindConn(const Mat <int>& current_config, field <field <Mat <int>>>& state_prime, field <cx_rowvec>& connections) {

  /*###################################################################################*/
  //  Finds the non-zero matrix elements of the spin observables
  //  on a given sampled spin configuration |𝒮⟩ named 𝐜𝐮𝐫𝐫𝐞𝐧𝐭_𝐜𝐨𝐧𝐟𝐢𝐠.
  //  In particular it searches all the |𝒮'⟩ such that
  //
  //        ⟨𝒮| Ô |𝒮'⟩ ≠ 𝟢
  //
  //  The configuration |𝒮'⟩ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |𝒮⟩ as abundantly described above.
  //
  //  Note that not all the observable connections change: for example they should
  //  not be recalculated for the transverse magnetization Σ̂ˣ, for which the constructor
  //  calculations are sufficient!
  /*###################################################################################*/

  //Check on the lattice dimensionality
  if(current_config.n_rows != 1 || current_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the quantum spin configuration does not match with the dimensionality of the system lattice." << std::endl;
    std::cerr << "   The system lives on a " << _d << " dimensional lattice and is composed of " << _L << " quantum spins." << std::endl;
    std::cerr << "   Failed to find the observable connections." << std::endl;
    std::abort();

  }

  //Assign pre-computed connections and |𝒮'⟩
  connections = _Connections;
  state_prime = _StatePrime;

  //Computing σ̂ᶻ-σ̂ᶻ interaction part for the local energy
  connections.at(0, 0)[0] = 0.0;
  if(_PBCs)
    for(int j = 0; j < _L; j++) connections.at(0, 0)[0] += double(current_config.at(0, j) * current_config.at(0, (j + 1) % _L));
  else
    for(int j = 0; j < (_L - 1); j++) connections.at(0, 0)[0] += double(current_config.at(0, j) * current_config.at(0, j + 1));
  connections.at(0, 0)[0] *= -_J;

}
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


/*******************************************************************************************************************************/
/*************************************************  𝐇𝐄𝐈𝐒𝐄𝐍𝐁𝐄𝐑𝐆 𝐌𝐎𝐃𝐄𝐋 𝐢𝐧 𝐝 = 𝟏  ************************************************/
/*******************************************************************************************************************************/
Heisenberg1d :: Heisenberg1d(int n_spin, int rank, double h_field, double jx, double jy, double jz, bool pbc)
              : SpinHamiltonian(n_spin, pbc), _h(h_field),  _Jx(jx), _Jy(jy), _Jz(jz) {

  /*#################################################################################################*/
  //  Creates the Hamiltonian operator for the Heisenberg model on a 𝐝 = 𝟏 lattice (𝟏𝐝 quantum chain)
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
  //        _Connections(𝟢) ‹--›  ⟨𝒮|  Ĥ  |𝒮'⟩
  //        _Connections(𝟣) ‹--›  ⟨𝒮|  Σ̂ˣ |𝒮'⟩
  //
  //  The considerations relating to the non-zero matrix elements for these observables
  //  are analogous to the previous model, but in this case even the primate configurations
  //  where two adjacent spins are flipped contribute to the non-zero connections.
  /*#################################################################################################*/

  //Information
  if(rank == 0) std::cout << "#Creation of the model Hamiltonian." << std::endl;

  //Data-members initialization
  _d = 1;  //𝟏𝐝 quantum chain
  _Connections.set_size(2, 1);
  _StatePrime.set_size(2, 1);

  //Function variables
  cx_rowvec energy_conn;  //Storage variable

  if(_PBCs){

    _StatePrime.at(0, 0).set_size(1, _L + 1);  //List of |𝒮'⟩ for the energy
    energy_conn.set_size(_L + 1);  //The first element corresponds to the case |𝒮'⟩ = |𝒮⟩

  }
  else{

    _StatePrime.at(0, 0).set_size(1, _L);  //List of |𝒮'⟩ for the energy
    energy_conn.set_size(_L);  //The first element corresponds to the case |𝒮'⟩ = |𝒮⟩

  }
  _StatePrime.at(1, 0).set_size(1, _L);  //List of |𝒮'⟩ for Σ̂ˣ
  cx_rowvec SigmaX_conn(_L, fill::zeros);

  //Pre-computed connections and associated |𝒮'⟩ definitions for ⟨𝒮| Ĥ |𝒮'⟩
  _StatePrime.at(0, 0).at(0, 0).reset();  //empty flipped_site matrix, i.e. |𝒮'⟩ = |𝒮⟩, diagonal term
  for(int j_flipped = 1; j_flipped < _L; j_flipped++){

    _StatePrime.at(0, 0).at(0, j_flipped).set_size(2, 1);  //|𝒮'⟩ ≠ |𝒮⟩ due to two adjacent flipped spin at lattice site j_flipped-1 & j_flipped
    _StatePrime.at(0, 0).at(0, j_flipped).at(0, 0) = j_flipped - 1;
    _StatePrime.at(0, 0).at(0, j_flipped).at(1, 0) = j_flipped;
    energy_conn[j_flipped] = -_Jx;  //non-diagonal term related to the σ̂ˣ-σ̂ˣ exchange interaction

  }
  if(_PBCs){

    _StatePrime.at(0, 0).at(0, _L).set_size(2, 1);  //|𝒮'⟩ ≠ |𝒮⟩ due to two adjacent flipped spin at the edge of the lattice site
    _StatePrime.at(0, 0).at(0, _L).at(0, 0) = _L - 1;
    _StatePrime.at(0, 0).at(0, _L).at(1, 0) = 0;
    energy_conn[_L] = -_Jx;

  }
  _Connections.at(0, 0) = energy_conn;

  //Pre-computed connections and associated |S'⟩ definitions for ⟨𝒮| Σ̂ˣ |𝒮'⟩
  for(int j_flipped = 0; j_flipped < _L; j_flipped++){

    _StatePrime.at(1, 0).at(0, j_flipped).set_size(1, 1);  // |𝒮'⟩ ≠ |𝒮⟩ due to a flipped spin at lattice site j_flipped
    _StatePrime.at(1, 0).at(0, j_flipped).at(0, 0) = j_flipped;
    SigmaX_conn[j_flipped] = 1.0;  //only non-diagonal term

  }
  _Connections.at(1, 0) = SigmaX_conn;

  //Indicates the created model
  if(rank == 0){

    if(_Jx != _Jy &&  _Jy != _Jz && _Jz != _Jx) std::cout << " XYZ model in 𝐝 = 𝟏 with " << _L << " quantum spins in 𝙝 = " << _h << " external magnetic field and ";
    else if(_Jx == _Jy &&  _Jy != _Jz && _Jz != _Jx) std::cout << " XXZ model in 𝐝 = 𝟏 with " << _L << " quantum spins in 𝙝 = " << _h << " external magnetic field and";
    else if(_Jx == _Jy &&  _Jy == _Jz && _Jz == _Jx && _Jx != -1.0) std::cout << " XXX model in 𝐝 = 𝟏 with " << _L << " quantum spins in 𝙝 = " << _h << " external magnetic field and ";
    else if(_Jx == -1.0 && _Jy == -1.0 && _Jz == -1.0) std::cout << " AntiFerromagnetic Heisenberg model in 𝐝 = 𝟏 with " << _L << " quantum spins in 𝙝 = " << _h << " external magnetic field and ";

    if(_PBCs) std::cout << "periodic boundary conditions." << std::endl;
    else std::cout << "open boundary conditions." << std::endl;

    std::cout << " Coupling constants of the Heisenberg model → 𝙅ˣ = " << _Jx << std::endl;
    std::cout << "                                            → 𝙅ʸ = " << _Jy << std::endl;
    std::cout << "                                            → 𝙅ᶻ = " << _Jz << std::endl << std::endl;

  }

}


void Heisenberg1d :: FindConn(const Mat <int>& current_config, field <field <Mat <int>>>& state_prime, field <cx_rowvec>& connections) {

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
  //  not be recalculated for the transverse polarization Σ̂ˣ, for which the constructor
  //  calculations are sufficient!
  /*########################################################################################*/

  //Check on the lattice dimensionality
  if(current_config.n_rows != 1 || current_config.n_cols != _L){

    std::cerr << " ##SizeError: the matrix representation of the quantum spin configuration does not match with the dimensionality of the system lattice." << std::endl;
    std::cerr << "   The system lives on a " << _d << " dimensional lattice and is composed of " << _L << " quantum spins." << std::endl;
    std::cerr << "   Failed to find the observable connections." << std::endl;
    std::abort();

  }

  //Assign pre-computed non-zero matrix elements and spin flips
  connections = _Connections;
  state_prime = _StatePrime;

  //|𝒮'⟩ = |𝒮⟩, diagonal term
  connections.at(0, 0)[0] = 0.0;
  cx_double acc_J = 0.0;  //σ̂ᶻ-σ̂ᶻ interaction part of the Hamiltonian
  cx_double acc_h = 0.0;  //interaction with the external magnetic field part of the Hamiltonian
  for(int j = 0; j <= (_L - 2); j++){

    acc_J += double(current_config.at(0, j) * current_config.at(0, j + 1));  //Computing the interaction part σ̂ᶻ-σ̂ᶻ
    acc_h += double(current_config.at(0, j));  //Computing the magnetic field interaction part

  }
  if(_PBCs) connections.at(0, 0)[0] = -_Jz * (acc_J + double(current_config.at(0, _L - 1) * current_config.at(0, 0))) - _h * (acc_h + double(current_config.at(0, _L - 1)));
  else connections.at(0, 0)[0] = -_Jz * acc_J - _h * (acc_h + double(current_config.at(0, _L - 1)));

  //|𝒮'⟩ ≠ |𝒮⟩, non diagonal terms
  //Computing the interaction part σ̂ʸ-σ̂ʸ
  for(int j_flipped = 1; j_flipped < _L; j_flipped++)
    connections.at(0, 0)[j_flipped] += _Jy * double(current_config.at(0, state_prime.at(0, 0).at(0, j_flipped).at(0, 0)) * current_config.at(0, state_prime.at(0, 0).at(0, j_flipped).at(1, 0)));
  if(_PBCs) connections.at(0, 0)[_L] += _Jy * double(current_config.at(0, state_prime.at(0, 0).at(0, _L).at(0, 0)) * current_config.at(0, state_prime.at(0, 0).at(0, _L).at(1, 0)));

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
