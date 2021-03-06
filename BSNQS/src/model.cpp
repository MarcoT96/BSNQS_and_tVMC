#ifndef __MODEL__
#define __MODEL__


/***************************************************************************************************************/
/******************************************  Quantum Hamiltonians  *********************************************/
/***************************************************************************************************************/
/*

  We create several models in order to represent the Many-Body Hamiltonians we want to study.
  In particular, we define the Hamiltonians for discrete Strongly-Correlated quantum systems,
  on both ๐ and ๐ dimensional lattice ๐ฒ ฯต โคแต.
  These classes describe the Hamiltonian operator, in particular its matrix elements, which
  are calculated and stored in a clever way, by searching which configurations of the total
  Hilbert space are connected by the local operators making up the Hamiltonian.
  More to the point, during a Monte Carlo procedure the quantum averages of the system
  properties are computed as

            โจ๐โฉ = โจฮจ(๐ฅ)| ร |ฮจ(๐ฅ)โฉ = ฮฃ๐ฎ ๐ชหกแตแถ(๐ฎ) |ฮจ(๐ฎ,๐ฅ)|^2
            ๐ชหกแตแถ(๐ฎ) = ฮฃ๐ฎห โจ๐ฎ| ๐ช |๐ฎหโฉ โข (ฮฆ(๐ฎห,๐,๐) / ฮฆ(๐ฎ,๐,๐))

  It is obviously not convenient to calculate all the matrix elements present in the sum defined above,
  but rather to consider only those for which

            โจ๐ฎ| ร |๐ฎ'โฉ โ  0

  In this case, we will say that the configuration |๐ฎ'โฉ is connected to the configuration |๐ฎโฉ;
  In the models we want to study the number of non-zero matrix elements (the connections) related to
  the various quantum observables of interest is typically small, and the task of the C++ classes
  defined below is precisely to manage the connections and the associated set of |๐ฎ'โฉ that are
  needed in the calculation of the instantaneous properies during the optimization via Variational
  Monte Carlo, once a system configuration |๐ฎโฉ is sampled.
  The characteristics of the connections and of the | ๐ฎ'โฉ depend on the observable that we want to
  measure and the particular quantum model that we are studying.

  NฬฒOฬฒTฬฒEฬฒ: so far, weโve only implemented one class of Hamiltonians related to Lattice Spin Systems (๐๐๐).
        Other types of Discrete Strongly Correlated systems can be implemented in the future.

*/
/***************************************************************************************************************/


/*###############*/
/*  C++ library  */
/*###############*/
#include <iostream>  // <-- std::cout, std::endl, etcโฆ
#include <cstdlib>  // <-- std::abort()
#include <complex>  // <-- std::complex<>, .real(), .imag()
#include <armadillo>  // <-- arma::Mat, arma::Col, arma::field


using namespace arma;


/*################################*/
/*  ๐๐๐๐ ๐๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐๐๐๐๐  */
/*################################*/
class SpinHamiltonian {

  protected:

    //System features
    const unsigned int _Nspin;  //Number of quantum spins
    unsigned int _LatticeDim;  //Dimensionality of the lattice ๐ฒ ฯต โคแต
    bool _pbc;  //Periodic Boundary Conditions
    const std::complex <double> _i;  //The imaginary unit ๐

    //Matrix representation of the spin observable operators
    field <Row <std::complex <double>>> _Connections;  //Non-zero Matrix Elements (i.e. the connections) of the spin observable operators
    field <field <Mat <int>>> _StatePrime;  //List of the flipped-spin lattice sites associated to each observables connections

  public:

    //Constructor and Destructor
    SpinHamiltonian(unsigned int n_spin, bool pbc=true) : _Nspin(n_spin), _pbc(pbc), _i(0.0, 1.0) {}  //Base constructor of a spin system
    virtual ~SpinHamiltonian() = default;  //Necessary for dynamic allocation

    //Virtual functions
    virtual void Quench(double) = 0;  //Quantum quench of a Hamiltonian coupling for real time dynamics
    virtual void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <Row <std::complex <double>>>&) = 0;  //Finds the connections for a given spin configuration |๐ฎโฉ
    virtual unsigned int MinFlips() const = 0;  //Returns the minimum number of spins to try to move during the MCMC

    //Access functions
    inline unsigned int n_spin() const {return _Nspin;}  //Returns the number of quantum degrees of freedom
    inline unsigned int dimensionality() const {return _LatticeDim;}  //Returns the lattice dimensionality ๐ฝ
    inline bool if_pbc() const {return _pbc;}  //Returns true if periodic boundary conditions are used on the system
    inline std::complex <double> i() const {return _i;}  //Returns the value of the imaginary unit
    inline field <Row <std::complex <double>>> get_connections() const {return _Connections;}  //Returns the list of connections
    inline field <field <Mat <int>>> all_state_prime() const {return _StatePrime;}  //Returns all the |๐ฎ'โฉ connected to the current configuration |๐ฎโฉ of the system
    inline Row <std::complex <double>> EnergyConn() const {return _Connections(0);}  //Returns the list of connections related to โจฤคโฉ
    inline Row <std::complex <double>> SxConn() const {return _Connections(1);}  //Returns the list of connections related to โจฯฬหฃโฉ
    inline Row <std::complex <double>> SyConn() const {return _Connections(2);}  //Returns the list of connections related to โจฯฬสธโฉ
    inline Row <std::complex <double>> SzConn() const {return _Connections(3);}  //Returns the list of connections related to โจฯฬแถปโฉ
    /*
    inline Row <std::complex <double>> SxSxConn() const {return _Connections(4);}  //Returns the list of connections related to โจฯฬหฃฯฬหฃโฉ
    inline Row <std::complex <double>> SySyConn() const {return _Connections(5);}  //Returns the list of connections related to โจฯฬสธฯฬสธโฉ
    inline Row <std::complex <double>> SzSzConn() const {return _Connections(6);}  //Returns the list of connections related to โจฯฬแถปฯฬแถปโฉ
    inline Row <std::complex <double>> SxSyConn() const {return _Connections(7);}  //Returns the list of connections related to โจฯฬหฃฯฬสธโฉ
    inline Row <std::complex <double>> SxSzConn() const {return _Connections(8);}  //Returns the list of connections related to โจฯฬหฃฯฬแถปโฉ
    inline Row <std::complex <double>> SySzConn() const {return _Connections(9);}  //Returns the list of connections related to โจฯฬสธฯฬแถปโฉ
    */

};


/*###########################################*/
/*  ๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐ ๐๐๐๐๐ ๐๐๐๐๐ ๐ข๐ง ๐ = ๐  */
/*##########################################*/
class Ising1d : public SpinHamiltonian {

  private:

    //Coupling constants (real valued)
    const double _J;  //ฯฬแถป-ฯฬแถป exchange interaction
    double _h;  //Transverse magnetic field

  public:

    //Constructor and Destructor
    Ising1d(unsigned int, double, double j=1.0, bool pbc=true);
    ~Ising1d(){}

    //Access functions
    inline double J() const {return _J;}  //Returns the ฯฬแถป-ฯฬแถป exchange interaction
    inline double h() const {return _h;}  //Returns the transverse magnetic field

    //Modifier functions
    void Quench(double);
    void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <Row <std::complex <double>>>&);
    unsigned int MinFlips() const {return 1;}

};


/*###############################*/
/*  ๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐ ๐ข๐ง ๐ = ๐  */
/*##############################*/
class Heisenberg1d : public SpinHamiltonian {

  private:

    //Coupling constants (real valued)
    const double _h;  //External magnetic field
    const double _Jx;  //ฯฬหฃ-ฯฬหฃ exchange interaction
    const double _Jy;  //ฯฬสธ-ฯฬสธ exchange interaction
    double _Jz;  //ฯฬแถป-ฯฬแถป exchange interaction

  public:

    //Constructor and Destructor
    Heisenberg1d(unsigned int, double hfield=0.0, double jx=-1.0, double jy=-1.0, double jz=-1.0, bool pbc=true);
    ~Heisenberg1d(){}

    //Access functions
    inline double h() const {return _h;}  //Returns the external magnetic field
    inline double Jx() const {return _Jx;}  //Returns the ฯฬหฃ-ฯฬหฃ exchange interaction
    inline double Jy() const {return _Jy;}  //Returns the ฯฬสธ-ฯฬสธ exchange interaction
    inline double Jz() const {return _Jz;}  //Returns the ฯฬแถป-ฯฬแถป exchange interaction

    //Modifier functions
    void Quench(double);
    void FindConn(const Mat <int>&, field <field <Mat <int>>>&, field <Row <std::complex <double>>>&);
    unsigned int MinFlips() const {return 2;}

};


/*******************************************************************************************************************************/
/******************************************  ๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐ ๐๐๐๐๐ ๐๐๐๐๐ ๐ข๐ง ๐ = ๐  *******************************************/
/*******************************************************************************************************************************/
Ising1d :: Ising1d(unsigned int n_spin, double h_field, double j, bool pbc)
         : SpinHamiltonian(n_spin, pbc), _J(j), _h(h_field) {

  /*############################################################################################################################*/
  //  Creates the Hamiltonian operator for the ๐๐๐ model on a ๐ = ๐ lattice (๐๐ Quantum Chain)
  //
  //        ฤค = -hโขฮฃโฑผฯฬโฑผหฃ - Jโขฮฃโฑผโฯฬโฑผแถปฯฬโแถป
  //
  //  with j,k n.n., i.e. k = j + 1, on the computational basis
  //
  //        |๐ฎโฉ = |ฯแถป๐ฃ ฯแถป๐ค โฆ ฯแถป๐ญโฉ
  //
  //  The observables connections we want to measure in our stochastic
  //  framework are
  //
  //        _Connections(0) โน--โบ  โจ๐ฎ|  ฤค  |๐ฎ'โฉ
  //        _Connections(1) โน--โบ  โจ๐ฎ|  ฯฬหฃ |๐ฎ'โฉ
  //        _Connections(2) โน--โบ  โจ๐ฎ|  ฯฬสธ |๐ฎ'โฉ
  //        _Connections(3) โน--โบ  โจ๐ฎ|  ฯฬแถป |๐ฎ'โฉ
  //        _Connections(4) โน--โบ  โจ๐ฎ| ฯฬหฃฯฬหฃ |๐ฎ'โฉ
  //        _Connections(5) โน--โบ  โจ๐ฎ| ฯฬสธฯฬสธ |๐ฎ'โฉ
  //        _Connections(6) โน--โบ  โจ๐ฎ| ฯฬแถปฯฬแถป |๐ฎ'โฉ
  //        _Connections(7) โน--โบ  โจ๐ฎ| ฯฬหฃฯฬสธ |๐ฎ'โฉ
  //        _Connections(8) โน--โบ  โจ๐ฎ| ฯฬหฃฯฬแถป |๐ฎ'โฉ
  //        _Connections(9) โน--โบ  โจ๐ฎ| ฯฬสธฯฬแถป |๐ฎ'โฉ
  //
  //
  //  NฬฒOฬฒTฬฒEฬฒ: we find instructive to explain the procedure that leads to the determination
  //        of the connections and of the configurations set |๐ฎ'โฉ for the quantum average of
  //        the energy of this model; the following arguments are easily extended to the
  //        other observable and to the other lattice models.
  //
  //  In the evaluation of the Hamiltonian operator matrix elements we have the
  //  following situation that greatly simplifies the calculation of the connections:
  //  the terms related to the ฯฬแถป-ฯฬแถป interaction connects only the same configuration,
  //  i.e. |๐ฎโฉ = |๐ฎ'โฉ; therefore there is only one non-zero-matrix element for
  //  this piece of Hamiltonian, which we save in the variable ๐๐ง๐๐ซ๐ ๐ฒ_๐๐จ๐ง๐ง(๐) below and
  //  which we have to recalculate every time a new configuration |๐ฎโฉ is sampled in the MCMC.
  //  When |๐ฎ'โฉ โ  |๐ฎโฉ a non-zero matrix element is obtained only when |๐ฎ'โฉ is
  //  identical to |๐ฎโฉ, less than a single flipped spin in a certain position ๐;
  //  In this case the term of the Hamiltonian related to the
  //  transverse field ฯฬ๐หฃ flips that spin on the right, giving
  //
  //        -hโขโจ๐ฎ| ฯฬ๐หฃ |๐ฎ'โฉ = -h
  //
  //  while all the other terms are zero.
  //  In order to manage more easily the calculation of these non-zero matrix elements,
  //  rather than keeping in memory the list of all possible configurations in which the system
  //  can be found (which would be 2แดบ), only the position of the flipped spin
  //  in the various configurations |๐ฎ'โฉ of the sum in the local energy is kept in memory in _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐(๐).
  //  In practice _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐(๐) represents all the possible configurations
  //  |๐ฎ'โฉ in the ฮฃ๐ฎ' โจ๐ฎ| ฤค |๐ฎ'โฉ for which the matrix elements
  //
  //        โจ๐ฎ| ฤค |๐ฎ'โฉ โ  ๐ข
  //
  //  including the case |๐ฎ'โฉ = |๐ฎโฉ which is saved in position ๐ข of _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐(๐,๐),
  //  respect of which no spin is flipped.
  //  These considerations extend easily to all other non-zero matrix elements related
  //  to each local observable we want to measure; therefore, to summarize, each element
  //  _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐(๐) will be the representation, as explained above, of all the primate
  //  configurations |๐ฎ'โฉ related to the various non-zero matrix elements of the appropriate
  //  observable among those listed above, collected in each _๐๐จ๐ง๐ง๐๐๐ญ๐ข๐จ๐ง๐ฌ(๐).
  //
  //  In other words, each element of _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐ is nothing more than the list of |๐ฎ'โฉ
  //  (each of them represented by a matrix of ๐๐ฅ๐ข๐ฉ๐ฉ๐๐_๐ฌ๐ข๐ญ๐ indices as explained in ๐๐ง๐ฌ๐๐ญ๐ณ.๐๐ฉ๐ฉ)
  //  that identify the locations of flipped spin lattice sites in |๐ฎ'โฉ compared to the current configuration
  //  of the system |๐ฎโฉ, and such that โจ๐ฎ| ร |๐ฎ'โฉ โ  ๐ข.
  //  Except in ๐ = ๐, these indices will be multidimensional indices, e.g. in the ๐ = ๐ case
  //  a generic element |๐ฎ'โฉ in the row _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐(๐) will be of the type
  //
  //        โ  0    0  โ   -->  Flip the 1st spin of the 2d lattice
  //        |  1    5  |   -->  ....
  //        |  ......  |   -->  ....
  //        |  ......  |   -->  ....
  //        |  ......  |   -->  ....
  //        โ  4    4  โ   -->  ....
  //
  //  For example, assuming that we are in the spin configuration
  //
  //               | +1 -1 +1     โฆ      +1  \
  //               | +1 -1 -1 +1 -1 +1 โฆ -1   \
  //        |๐ฎโฉ =    :  :  :      โฆ      -1    \
  //               | :  :  :      โฆ      -1    /
  //               | -1 +1 +1 +1 -1 โฆ    -1   /
  //               | :  :  :      โฆ      +1  /
  //
  //  the newly particular defined |๐ฎ'โฉ in the row _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐(๐) would represent the configuration
  //
  //               | -1 -1 +1 โฆ โฆ โฆ โฆ  โฆ +1  \
  //               | +1 -1 -1 +1 -1 -1 โฆ -1   \
  //        |๐ฎ'โฉ =   :  :  :      โฆ      -1    \
  //               | :  :  :      โฆ      -1    /
  //               | -1 +1 +1 +1 +1 โฆ    -1   /
  //               | :  :  :      โฆ      +1  /
  //
  //  As mentioned above, _๐๐ญ๐๐ญ๐๐๐ซ๐ข๐ฆ๐ will be, from a linear algebra point of view, a kind of tensor,
  //  with as many rows as there are observable that we want to measure during the Monte Carlo optimization;
  //  each row will have a certain number of matrices (columns) representing each of the |๐ฎ'โฉ configurations
  //  for which the connection is not zero; we remember once again that these matrices are nothing more
  //  than the list of lattice sites where the spins have been flipped compared to the configuration |๐ฎโฉ in which the
  //  system is located.
  //  At this point the structure of _๐๐จ๐ง๐ง๐๐๐ญ๐ข๐จ๐ง๐ฌ is obvious: each row (Row) will contain the list of non-zero
  //  connections, and will have as many elements as the number of |๐ฎ'โฉ described above.
  //  Schematically:
  //
  //                                         โ                                                           โ
  //        _Connections(0) = EnergyConn โน-โบ |  โจ๐ฎ|ฤค|๐ฎ'1โฉ,  โจ๐ฎ|ฤค|๐ฎ'2โฉ,  โจ๐ฎ|ฤค|๐ฎ'3โฉ,  โฆโฆโฆโฆโฆ,  โจ๐ฎ|ฤค|๐ฎ'โโฉ  |
  //                                         โ                                                           โ
  //
  //                                                               โ
  //
  //                                         โ                                                                               โ
  //        _StatePrime(0) =  {|๐ฎ'โฉ}  โน-โบ    |  |๐ฎ'1โฉ = Mat( โข | โข ),  |๐ฎ'2โฉ = Mat( โข | โข ),  โฆโฆโฆโฆโฆ,  |๐ฎ'โโฉ = Mat( โข | โข )  |
  //                                         โ                                                                               โ
  //
  //  and so on for the other observable connections.
  /*############################################################################################################################*/

  //Information
  std::cout << "#Creation of the model Hamiltonian" << std::endl;

  //Data-members initialization
  _LatticeDim = 1;  //๐๐ Quantum Chain
  _Connections.set_size(4, 1);
  _StatePrime.set_size(4, 1);
  _StatePrime(0, 0).set_size(1, _Nspin+1);  //List of |๐ฎ'โฉ for the energy
  _StatePrime(1, 0).set_size(1, _Nspin);  //List of |๐ฎ'โฉ for ฯฬหฃ
  _StatePrime(2, 0).set_size(1, _Nspin);  //List of |๐ฎ'โฉ for ฯฬสธ
  _StatePrime(3, 0).set_size(1, 1);  //List of |๐ฎ'โฉ for ฯฬแถป

  //Function variables
  Row <std::complex <double>> energy_conn(_Nspin+1, fill::zeros);  //The first element corresponds to the case |๐ฎ'โฉ = |๐ฎโฉ
  Row <std::complex <double>> sigmax_conn(_Nspin, fill::zeros);  //Storage variable

  //Pre-computed connections and associated |๐ฎ'โฉ definitions for โจ๐ฎ| ฤค |๐ฎ'โฉ
  _StatePrime(0, 0)(0, 0).reset();  //empty flipped_site matrix, i.e. |๐ฎ'โฉ = |๐ฎโฉ, diagonal term
  for(unsigned int j_flipped=1; j_flipped<_Nspin+1; j_flipped++){

    _StatePrime(0, 0)(0, j_flipped).set_size(1, 1);  //|๐ฎ'โฉ โ  |๐ฎโฉ due to a flipped spin at lattice site j_flipped-1
    _StatePrime(0, 0)(0, j_flipped)(0, 0) = j_flipped-1;
    energy_conn(j_flipped) = -_h;  //non-diagonal term

  }
  _Connections(0, 0) = energy_conn;

  //Pre-computed connections and associated |๐ฎ'โฉ definitions for โจ๐ฎ| ฯฬโฑผ |๐ฎ'โฉ
  for(unsigned int j_flipped=0; j_flipped<_Nspin; j_flipped++){

    _StatePrime(1, 0)(0, j_flipped).set_size(1, 1);  //ฯฬหฃ, |๐ฎ'โฉ โ  |๐ฎโฉ due to a flipped spin at lattice site j_flipped
    _StatePrime(1, 0)(0, j_flipped)(0, 0) = j_flipped;
    sigmax_conn(j_flipped) = 1.0;  //only non-diagonal term

    _StatePrime(2, 0)(0, j_flipped).set_size(1, 1);  //ฯฬสธ, |๐ฎ'โฉ โ  |๐ฎโฉ due to a flipped spin at lattice site j_flipped
    _StatePrime(2, 0)(0, j_flipped)(0, 0) = j_flipped;

  }
  _StatePrime(3, 0)(0, 0).reset();  //ฯฬแถป, empty flipped_site matrix, i.e. |๐ฎ'โฉ = |๐ฎโฉ, only diagonal term
  _Connections(1, 0) = sigmax_conn;

  //Pre-computed connections and associated |๐ฎ'โฉ definitions for โจ๐ฎ| ฯฬโฑผฯฬโ |๐ฎ'โฉ
  /*
    ..........
    ..........
    ..........
  */

  //Indicates the created model
  std::cout << " Transverse Field Ising model in ๐ = ๐ with " << _Nspin << " Quantum spins in h = " << _h << " magnetic field." << std::endl;
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
  //  on a given sampled spin configuration |๐ฎโฉ named ๐๐ฎ๐ซ๐ซ๐๐ง๐ญ_๐ฌ๐ญ๐๐ญ๐.
  //  In particular it searches all the |๐ฎ'โฉ such that
  //
  //        โจ๐ฎ| ร |๐ฎ'โฉ โ  ๐ข
  //
  //  The configuration |๐ฎ'โฉ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |๐ฎโฉ as abundantly described above.
  //
  //  Note that not all the observable connections change: for example they should
  //  not be recalculated for the transverse polarization ฯฬโ, for which the constructor
  //  calculations are sufficient!
  /*###################################################################################*/

  //Check on the lattice dimensionality
  if(current_config.n_rows != 1 || current_config.n_cols != _Nspin){

    std::cerr << " ##SizeError: the matrix representation of the quantum spin configuration does not match with the dimensionality of the system lattice." << std::endl;
    std::cerr << "   The system lives on a " << _LatticeDim << " dimensional lattice and is composed of " << _Nspin << " quantum spins." << std::endl;
    std::cerr << "   Failed to find the observable connections." << std::endl;
    std::abort();

  }

  //Assign pre-computed connections and |๐ฎ'โฉ
  connections = _Connections;
  state_prime = _StatePrime;

  //Computing ฯฬแถป-ฯฬแถป interaction part for the local energy
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
/*************************************************  ๐๐๐๐๐๐๐๐๐๐ ๐๐๐๐๐ ๐ข๐ง ๐ = ๐  ************************************************/
/*******************************************************************************************************************************/
Heisenberg1d :: Heisenberg1d(unsigned int n_spin, double h_field, double jx, double jy, double jz, bool pbc)
              : SpinHamiltonian(n_spin, pbc), _h(h_field),  _Jx(jx), _Jy(jy), _Jz(jz) {

  /*#################################################################################################*/
  //  Creates the Hamiltonian operator for the Heisenberg model on a ๐ = ๐ lattice (๐๐ Quantum Chain)
  //
  //        ฤค = -ฮฃโฑผโ(Jหฃโขฯฬหฃโฑผฯฬหฃโ + Jสธโขฯฬสธโฑผฯฬสธโ + Jแถปโขฯฬแถปโฑผฯฬแถปโ) - hโขฮฃโฑผฯฬแถปโฑผ
  //        k = j + 1 (and pbc)
  //
  //  on the computational basis
  //
  //        |๐ฎโฉ = |ฯแถป๐ฃ ฯแถป๐ค โฆ ฯแถป๐ญโฉ
  //
  //  For this model we can have various combinations regarding the value of the
  //  coupling constants {h, Jหฃ, Jสธ, Jแถป}, catalogued with the following models:
  //
  //        โข Jหฃ โ  Jสธ โ  Jแถป โ  ๐ข & h โ  ๐ข      โน--โบ  ๐๐๐ Model
  //        โข Jหฃ = Jสธ = J & Jแถป = ฮ & h โ  ๐ข  โน--โบ  ๐๐๐ Model
  //        โข Jหฃ = Jสธ = Jแถป = J & h โ  ๐ข      โน--โบ  ๐๐๐ Model
  //        โข Jหฃ = Jสธ = Jแถป = -1 & h = ๐ข     โน--โบ  ๐๐๐ Model (AntiFerromagnetic Heisenberg Model)
  //
  //  where the ๐๐๐ is the default in this implementation.
  //  The local spin observables we want to measure in our stochastic
  //  framework are
  //
  //        _Connections(0) โน--โบ  โจ๐ฎ|  ฤค  |๐ฎ'โฉ
  //        _Connections(1) โน--โบ  โจ๐ฎ |  ฯฬหฃ |๐ฎ'โฉ
  //        _Connections(2) โน--โบ  โจ๐ฎ |  ฯฬสธ |๐ฎ'โฉ
  //        _Connections(3) โน--โบ  โจ๐ฎ |  ฯฬแถป |๐ฎ'โฉ
  //        _Connections(4) โน--โบ  โจ๐ฎ | ฯฬหฃฯฬหฃ |๐ฎ'โฉ
  //        _Connections(5) โน--โบ  โจ๐ฎ | ฯฬสธฯฬสธ |๐ฎ'โฉ
  //        _Connections(6) โน--โบ  โจ๐ฎ | ฯฬแถปฯฬแถป |๐ฎ'โฉ
  //        _Connections(7) โน--โบ  โจ๐ฎ | ฯฬหฃฯฬสธ |๐ฎ'โฉ
  //        _Connections(8) โน--โบ  โจ๐ฎ | ฯฬหฃฯฬแถป |๐ฎ'โฉ
  //        _Connections(9) โน--โบ  โจ๐ฎ | ฯฬสธฯฬแถป |๐ฎ'โฉ
  //
  //  The considerations relating to the non-zero matrix elements for these observables
  //  are analogous to the previous model, but in this case even the primate configurations
  //  where two adjacent spins are flipped contribute to the non-zero connections.
  /*#################################################################################################*/

  //Information
  std::cout << "#Creation of the model Hamiltonian" << std::endl;

  //Data-members initialization
  _LatticeDim = 1;  //๐๐ Quantum Chain
  _Connections.set_size(4, 1);
  _StatePrime.set_size(4, 1);

  //Function variables
  Row <std::complex <double>> energy_conn;  //Storage variable

  if(_pbc){

    _StatePrime(0, 0).set_size(1, _Nspin+1);  //List of |๐ฎ'โฉ for the energy
    energy_conn.set_size(_Nspin+1);  //The first element corresponds to the case |๐ฎ'โฉ = |๐ฎโฉ

  }
  else{

    _StatePrime(0, 0).set_size(1, _Nspin);  //List of |๐ฎ'โฉ for the energy
    energy_conn.set_size(_Nspin);  //The first element corresponds to the case |๐ฎ'โฉ = |๐ฎโฉ

  }
  _StatePrime(1, 0).set_size(1, _Nspin);  //List of |๐ฎ'โฉ for ฯฬหฃ
  _StatePrime(2, 0).set_size(1, _Nspin);  //List of |๐ฎ'โฉ for ฯฬสธ
  _StatePrime(3, 0).set_size(1, 1);  //List of |๐ฎ'โฉ for ฯฬแถป
  Row <std::complex <double>> sigmax_conn(_Nspin, fill::zeros);

  //Pre-computed connections and associated |๐ฎ'โฉ definitions for โจ๐ฎ| ฤค |๐ฎ'โฉ
  _StatePrime(0, 0)(0, 0).reset();  //empty flipped_site matrix, i.e. |๐ฎ'โฉ = |๐ฎโฉ, diagonal term
  for(unsigned int j_flipped=1; j_flipped<_Nspin; j_flipped++){

    _StatePrime(0, 0)(0, j_flipped).set_size(2, 1);  //|๐ฎ'โฉ โ  |๐ฎโฉ due to two adjacent flipped spin at lattice site j_flipped-1 & j_flipped
    _StatePrime(0, 0)(0, j_flipped)(0, 0) = j_flipped-1;
    _StatePrime(0, 0)(0, j_flipped)(1, 0) = j_flipped;
    energy_conn(j_flipped) = -_Jx;  //non-diagonal term related to the ฯฬหฃ-ฯฬหฃ exchange interaction

  }
  if(_pbc){

    _StatePrime(0, 0)(0, _Nspin).set_size(2, 1);  //|๐ฎ'โฉ โ  |๐ฎโฉ due to two adjacent flipped spin at the edge of the lattice site
    _StatePrime(0, 0)(0, _Nspin)(0, 0) = _Nspin-1;
    _StatePrime(0, 0)(0, _Nspin)(1, 0) = 0;
    energy_conn(_Nspin) = -_Jx;

  }
  _Connections(0, 0) = energy_conn;

  //Pre-computed connections and associated |S'โฉ definitions for โจ๐ฎ| ฯฬโฑผ |๐ฎ'โฉ
  for(unsigned int j_flipped=0; j_flipped<_Nspin; j_flipped++){

    _StatePrime(1, 0)(0, j_flipped).set_size(1, 1);  //ฯฬหฃ, |๐ฎ'โฉ โ  |๐ฎโฉ due to a flipped spin at lattice site j_flipped
    _StatePrime(1, 0)(0, j_flipped)(0, 0) = j_flipped;
    sigmax_conn(j_flipped) = 1.0;  //only non-diagonal term

    _StatePrime(2, 0)(0, j_flipped).set_size(1, 1);  //ฯฬสธ, |๐ฎ'โฉ โ  |๐ฎโฉ due to a flipped spin at lattice site j_flipped
    _StatePrime(2, 0)(0, j_flipped)(0, 0) = j_flipped;

  }
  _StatePrime(3, 0)(0, 0).reset();  //ฯฬแถป, empty flipped_site matrix, i.e. |๐ฎ'โฉ = |๐ฎโฉ, only diagonal term
  _Connections(1, 0) = sigmax_conn;

  //Pre-computed connections and associated |๐ฎ'โฉ definitions for โจ๐ฎ| ฯฬโฑผฯฬโ |๐ฎ'โฉ
  /*
    ..........
    ..........
    ..........
  */

  //Indicates the created model
  if(_Jx!=_Jy &&  _Jy!=_Jz && _Jz!=_Jx)
    std::cout << " XYZ model in ๐ = ๐ with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;
  else if(_Jx==_Jy &&  _Jy!=_Jz && _Jz!=_Jx)
    std::cout << " XXZ model in ๐ = ๐ with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;
  else if(_Jx==_Jy &&  _Jy==_Jz && _Jz==_Jx && _Jx!=-1.0)
    std::cout << " XXX model in ๐ = ๐ with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;
  else if(_Jx==-1.0 && _Jy==-1.0 && _Jz==-1.0)
    std::cout << " AntiFerromagnetic Heisenberg model in ๐ = ๐ with " << _Nspin << " Quantum spins in h = " << _h << " external magnetic field." << std::endl;

  std::cout << " Coupling constants of the Heisenberg model:" << std::endl;
  std::cout << " \tJหฃ = " << _Jx << std::endl;
  std::cout << " \tJสธ = " << _Jy << std::endl;
  std::cout << " \tJแถป = " << _Jz << std::endl << std::endl;

}


void Heisenberg1d :: Quench(double jf) {

  /*##################################################*/
  //  Introduces nontrivial quantum dynamics
  //  by means of an instantaneous change in the
  //  ฯฬแถป-ฯฬแถป exchange interaction from _Jz to jf.
  //  Due to this quantum quench certain observable
  //  needs to be modified, such as the local energy.
  /*##################################################*/

  _Jz = jf;

}


void Heisenberg1d :: FindConn(const Mat <int>& current_config, field <field <Mat <int>>>& state_prime, field <Row <std::complex <double>>>& connections) {

  /*########################################################################################*/
  //  Finds the non-zero matrix elements of the spin observables
  //  on a given sampled configuration passed as the first argument of this function.
  //  In particular it searches all the |๐ฎ'โฉ such that
  //
  //        โจ๐ฎ| ร |๐ฎ'โฉ โ  ๐ข
  //
  //  The configuration |๐ฎ'โฉ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |๐ฎโฉ.
  //  For example, in the evaluation of the Hamiltonian operator we have
  //  the following situation:
  //
  //        โข |๐ฎ'โฉ = |๐ฎโฉ
  //          When no spin flips are performed we have
  //                    โจ๐ฎ| ฯฬหฃ[j]ฯฬหฃ[j+1] |๐ฎ'โฉ = ๐ข    for all j
  //                    โจ๐ฎ| ฯฬสธ[j]ฯฬสธ[j+1] |๐ฎ'โฉ = ๐ข    for all j
  //                    โจ๐ฎ| ฯฬแถป[j]ฯฬแถป[j+1] |๐ฎ'โฉ = ฯแถป[j]ฯแถป[j+1]
  //                    โจ๐ฎ| ฯฬแถป[j] |๐ฎ'โฉ = ฯแถป[j]
  //        โข |๐ฎ'โฉ โ  |๐ฎโฉ
  //          When only one spin is flipped in position ๐ฟ we have
  //                    โจ๐ฎ| ฯฬหฃ[๐ฟ]ฯฬหฃ[๐ฟ+1] |๐ฎ'โฉ = ๐ข    for all ๐ฟ
  //                    โจ๐ฎ| ฯฬสธ[๐ฟ]ฯฬสธ[๐ฟ+1] |๐ฎ'โฉ = ๐ข    for all ๐ฟ
  //                    โจ๐ฎ| ฯฬแถป[๐ฟ]ฯฬแถป[๐ฟ+1] |๐ฎ'โฉ = ๐ข    for all ๐ฟ
  //                    โจ๐ฎ| ฯฬแถป[๐ฟ] |๐ฎ'โฉ = ๐ข           for all ๐ฟ
  //        โข |๐ฎ'โฉ โ  |๐ฎโฉ
  //          When two spins are flipped in position ๐ฟ, ๐ฟ+1 we have
  //                    โจ๐ฎ| ฯฬหฃ[๐ฟ]ฯฬหฃ[๐ฟ+1] |๐ฎ'โฉ = 1    for all ๐ฟ
  //                    โจ๐ฎ| ฯฬสธ[๐ฟ]ฯฬสธ[๐ฟ+1] |๐ฎ'โฉ = -ฯแถป[๐ฟ]ฯแถป[j+๐ฟ]
  //                    โจ๐ฎ| ฯฬแถป[๐ฟ]ฯฬแถป[๐ฟ+1] |๐ฎ'โฉ = ๐ข    for all ๐ฟ
  //                    โจ๐ฎ| ฯฬแถป[๐ฟ] |๐ฎ'โฉ = ๐ข           for all ๐ฟ
  //        โข |๐ฎ'โฉ โ  |๐ฎโฉ
  //          When more than two spins are flipped in position we have
  //                    โจ๐ฎ| ฯฬหฃ[j]ฯฬหฃ[j+1] |๐ฎ'โฉ = ๐ข    for all j
  //                    โจ๐ฎ| ฯฬสธ[j]ฯฬสธ[j+1] |๐ฎ'โฉ = ๐ข    for all j
  //                    โจ๐ฎ| ฯฬแถป[j]ฯฬแถป[j+1] |๐ฎ'โฉ = ๐ข    for all j
  //                    โจ๐ฎ| ฯฬแถป[j] |๐ฎ'โฉ = ๐ข           for all j
  //
  //  The configuration |๐ฎ'โฉ is encoded as the sequence of spin flips
  //  to be performed on the current configuration |๐ฎโฉ as abundantly described above.
  //
  //  Note that not all the observable connections change: for example they should
  //  not be recalculated for the transverse polarization ฯฬโ, for which the constructor
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

  //|๐ฎ'โฉ = |๐ฎโฉ, diagonal term
  connections(0, 0)(0) = 0.0;
  std::complex <double> acc_J = 0.0;  //ฯฬแถป-ฯฬแถป interaction part of the Hamiltonian
  std::complex <double> acc_h = 0.0;  //interaction with the external magnetic field part of the Hamiltonian
  for(unsigned int j=0; j<=(_Nspin-2); j++){

    acc_J += double(current_config(0, j)*current_config(0, j+1));  //Computing the interaction part ฯฬแถป-ฯฬแถป
    acc_h += double(current_config(0, j));  //Computing the magnetic field interaction part

  }
  if(_pbc)
    connections(0, 0)(0) = -_Jz * (acc_J + double(current_config(0, _Nspin-1)*current_config(0, 0))) - _h * (acc_h + double(current_config(0, _Nspin-1)));
  else
    connections(0, 0)(0) = -_Jz * acc_J - _h * (acc_h + double(current_config(0, _Nspin-1)));

  //|๐ฎ'โฉ โ  |๐ฎโฉ, non diagonal terms
  //Computing the interaction part ฯฬสธ-ฯฬสธ
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
