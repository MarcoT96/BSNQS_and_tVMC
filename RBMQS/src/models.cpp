#ifndef __MODELS__
#define __MODELS__


#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>


/*
################################################################################
###################     Hamiltonians of Quantum Models     #####################
###################  Discrete Strongly Correlated Systems  #####################
################################################################################
*/


/*******************************************************************************************************************************/
/***************************************** 1d Transverse-Filed Ising Model *****************************************************/
/*****************************************      TFI model Hamiltonian      *****************************************************/
/*******************************************************************************************************************************/

/*******************/
class Ising1d {

  private:

    const int _Nspin;  //Number of quantum Ising variables
    const double _h;  //Transverse magnetic field
    const bool _pbc;  //whether or not to apply
                      //Periodic Boundary Conditions

    //pre-computed quantities
    std::vector <std::complex <double>> _mel;  //non-zero matrix elements for energy
    std::vector <std::complex <double>> _sigmax;  //non-zero matrix elements for sigmax
    std::vector <std::vector <int>> _flipsh;  //list of the flipped spin-site
                                              //related to each of the _mel[j]

  public:

    //Constructor and Destructor
    Ising1d(int, double, bool pbc=true);
    ~Ising1d(){}

    //Access functions
    inline int Nspin() const {return _Nspin;}
    inline double h() const {return _h;}
    inline bool if_pbc() const {return _pbc;}
    std::vector <std::complex <double>> mel() const {return _mel;}
    std::vector <std::vector <int>> flipsh() const {return _flipsh;}
    std::vector <std::complex <double>> sigmax() const {return _sigmax;}
    std::complex <double> mel_n(unsigned int) const;
    int flipsh_mn(unsigned int, unsigned int) const;

    //Modifier functions
    void set_mel_n(unsigned int, std::complex <double>);
    void set_flipsh_mn(unsigned int, unsigned int, int);
    void quench(double);
    void FindConn(const std::vector <int>&, std::vector <std::vector <int>>&, std::vector <std::complex <double>>&);
    int MinFlips() const;

};
/*******************/

Ising1d :: Ising1d(int nspin, double hfield, bool pbc) :
                   _Nspin(nspin), _h(hfield), _pbc(pbc) {

  //Information
  std::cout << "#Creation of the model Hamiltonian" << std::endl;

  //Appropriate initialization
  //See the Jupyter Notebook
  _mel.resize(_Nspin+1, 0.0);  /*
                                 Here _mel[0] represent the case |S'> = |S>
                                 in the matrix element of the local energy
                                 i.e. the term related to the Sz-Sz interaction
                               */
  _sigmax.resize(_Nspin, 0.0) ;
  _flipsh.resize(_Nspin+1);  /*
                               _flipsh[0] = empty_vector because no spin is flipped
                               since it is related to the non-zero matrix element of
                               the case |S'> = |S>
                             */
  for(int j=0; j<_Nspin; j++){

    /*
      non-zero matrix elements when |S'> \neq |S>
      This means that only one spin is flipped: in this
      situation, the term of the Hamiltonian related to the
      transverse field Sx flips that spin on the right, giving

                  -h <S | Sx | S'> = -h

      All other terms are zero. In _flipsh memory of the position
      of the spin flipped in the various states S' of the sum in the
      local energy is kept.
      In practice the vector _flipsh represents all the possible states
      |S'> in the ∑_S' <S | H | S'> for which the matrix elements

                  <S | H | S'> \neq 0

      including the case |S'> = |S> which is saved in position 0 of _flipsh
      respect of which no spin is flipped.
    */
    _mel[j+1] = -_h;
    _sigmax[j] = 1.0;
    _flipsh[j+1] = std::vector <int>(1, j);

  }

  //Indicate the created model
  std::cout << " 1d Transverse Field Ising model with " << _Nspin << " Quantum spins in h = " << _h << " magnetic field." << std::endl << std::endl;

}


std::complex <double> Ising1d :: mel_n(unsigned int n) const {

  if(n >= _mel.size()){

    std::cerr << " ##Error: accessing non-zero hamiltonian matrix element failed." << std::endl;
    std::cerr << "   Element mel_" << n << " does not exist." << std::endl;
    return -1.0;

  }
  else
    return _mel[n];

}


int Ising1d :: flipsh_mn(unsigned int m, unsigned int n) const {

  if(m >= _flipsh.size() || n >= _flipsh[0].size()){

    std::cerr << " ##Error: accessing blablabla element failed." << std::endl;
    std::cerr << "   Element flipsh_" << m << n << " does not exist." << std::endl;
    return -1.0;

  }
  else
    return _flipsh[m][n];

}


void Ising1d :: set_mel_n(unsigned int n, std::complex <double> mel) {

  if(n >= _mel.size()){

    std::cerr << " ##Error: accessing non-zero hamiltonian matrix element failed." << std::endl;
    std::cerr << "   Element mel_" << n << " does not exist." << std::endl;
    std::abort();

  }
  else
    _mel[n].real(mel.real());
    _mel[n].imag(mel.imag());


}


void Ising1d :: set_flipsh_mn(unsigned int m, unsigned int n, int flipsh) {

  if(m >= _flipsh.size() || n >= _flipsh[0].size()){

    std::cerr << " ##Error: accessing blablabla element failed." << std::endl;
    std::cerr << "   Element flipsh_" << m << n << " does not exist." << std::endl;
    std::abort();

  }
  else
    _flipsh[m][n] = flipsh;

}


void Ising1d :: quench(double hf) {

  /*
    Introduces nontrivial quantum dynamics
    by means of an instantaneous change in the
    transverse field from _h to hf.
    Due to this quantum quench the local Energy
    needs to be modified.
  */
  _mel.resize(_Nspin+1, 0.0);
  for(int j=0; j<_Nspin; j++)
    _mel[j+1] = -hf;

}


void Ising1d :: FindConn(const std::vector <int>& current_state, std::vector <std::vector <int>>& flipsh, std::vector <std::complex <double>>& mel) {

  /*
    Finds the non-zero matrix elements of the Hamiltonian on the given state
    passed as the first argument of this function.
    In particular it searches all the state_prime such that

              <state_prime | H | state> \neq 0

    The configuration state_prime is encoded as the sequence of spin flips
    to be performed on state.
  */

  //Assign pre-computed non-zero matrix elements and spin flips
  mel.resize(_Nspin+1);
  flipsh.resize(_Nspin+1);
  mel = _mel;
  flipsh = _flipsh;

  //Computing interaction part Sz*Sz
  mel[0] = 0.0;
  for(int j=0; j<(_Nspin-1); j++)
    mel[0] -= double(current_state[j]*current_state[j+1]);
  if(_pbc)
    mel[0] -= double(current_state[_Nspin-1]*current_state[0]);

}


int Ising1d :: MinFlips() const {

  return 1;

}

/*******************************************************************************************************************************/
/*******************************************************************************************************************************/
/*******************************************************************************************************************************/


#endif
