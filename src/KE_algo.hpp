#ifndef KE_algo_HPP
#define KE_algo_HPP

#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <vector>
#include <functional>
#include <utility>
#include <tuple>

#include "traits_ke.hpp"




//templates params
//derived class (CRTP), domain dimension, if k is passed as param, if validation errors have to be stored and returned
template< unsigned int IMP > 
class KE_algo
{
  
private:
  
  KE_Traits::StoringMatrix m_X;               //matrix storing time series (m x n), data centered
  std::size_t m_m;                            //number of evaluation of the functional object (m)
  std::size_t m_n;                            //number of time instants (n)
  KE_Traits::StoringArray m_means;            //vector storing the time series means (m x 1)  
  KE_Traits::StoringMatrix m_Cov;             //matrix estimating the covariance operator (m x m)
  KE_Traits::StoringMatrix m_CrossCov;        //matrix estimating the cross-covariance operator (m x m)
  KE_Traits::StoringMatrix m_psi;             //matrix containing the estimate of the operator for doing 1-step ahead prediction (m x m)
  std::vector<double> m_explanatory_power;    //vector containing the cumulative explanatory power (will have size k)
  int m_k;                                    //number of PPCs retained
  int m_number_threads;                       //number of threads for OMP
  
  
public:
  

  KE_algo(const KE_Traits::StoringMatrix& X,int k,int number_threads)
    :   
    m_X{X},
    m_m(X.rows()),
    m_n(X.cols()),
    m_k(k),
    m_number_threads(number_threads)
    {  
      
      //evaluating row mean and saving it in the m_means
      m_means = (m_X.rowwise().sum())/m_n;
      
      //normalizing

      for (size_t i = 0; i < m_n; ++i)
      {
        m_X.col(i) = m_X.col(i).array() - m_means;
      }
      
      // X * X' 
      m_Cov =  ((m_X*m_X.transpose()).array())/static_cast<double>(m_n);
      
      
      
      // (X[,2:n]*(X[,1:(n-1)])')/(n-1)
      m_CrossCov =  ((m_X.rightCols(m_n-1)*m_X.leftCols(m_n-1).transpose()).array())/(static_cast<double>(m_n-1));
      
    }
  
  
  
  //Getters and setters
  
  /*!
   * Getter for m_m
   */
  inline std::size_t m() const {return m_m;};
  
  /*!
   * Getter for m_n
   */
  inline std::size_t n() const {return m_n;};
  
  /*!
   * Getter for m_X
   */
  inline KE_Traits::StoringMatrix X() const {return m_X;};
  
  /*!
   * Getter for m_means
   */
  inline KE_Traits::StoringArray means() const {return m_means;};
  
  /*!
   * Getter for m_Cov
   */
  inline KE_Traits::StoringMatrix Cov() const {return m_Cov;};
  
  /*!
   * Getter for m_rho
   */
  inline KE_Traits::StoringMatrix psi() const {return m_psi;};
  
  /*!
   * Getter for m_explanatory_power
   */
  inline std::vector<double> explanatory_power() const {return m_explanatory_power;};
  
  /*!
   * Getter for m_k
   */
  inline int k() const {return m_k;};
  
  /*!
   * Setter for m_k
   */
  inline int & k() {return m_k;};
  
  /*!
   * Getter for m_number_threads
   */
  inline int number_threads() const {return m_number_threads;};
  
  
  //KO algorithm, once parameters have been set
  void KE_algorithm();
  
  //one-step ahead prediction
  KE_Traits::StoringArray prediction() const;
  
};


#include "KE_algo_imp.hpp"

#endif  //KE_PPC_CRTP_HPP