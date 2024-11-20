#ifndef KE_CV_HPP
#define KE_CV_HPP

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
#include "KE_algo.hpp"
#include "cv_eval_valid_err.hpp"





//templates params
//derived class (CRTP), domain dimension, if k is passed as param, if validation errors have to be stored and returned
template< unsigned int IMP > 
class KE_algo_cv
{
  
private:
  
  std::size_t m_first_train_set_dim;
  std::size_t m_last_train_set_dim;            //number of evaluation of the functional object (m)
  KE_Traits::StoringMatrix m_X;               //matrix storing time series (m x n), data centered
  KE_Traits::StoringMatrix m_X_cent;
  KE_Traits::StoringArray m_means;            //vector storing the time series means (m x 1)  
  KE_Traits::StoringMatrix m_Cov;             //matrix estimating the covariance operator (m x m)
  double m_trace_cov;                         //trace of the covariance estimator
  KE_Traits::StoringMatrix m_psi_hat;         //matrix containing the estimate of the operator for doing 1-step ahead prediction (m x m)
  std::vector<int> m_k_s;
  std::vector<double> m_explanatory_power;    //vector containing the cumulative explanatory power (will have size k)
  int m_number_threads;                       //number of threads for OMP
  std::vector<int>  m_ti_ts;
  double m_toll;
  
  int m_k;                                    //number of PCs retained
  KE_Traits::StoringVector m_pred;
  
  
  
public:
  
  template<typename STOR_OBJ>
  KE_algo_cv(STOR_OBJ&& X,std::vector<int> k_s,int first_ts,int last_ts,int toll,int number_threads)
    :   
    m_X{std::forward<STOR_OBJ>(X)},
    m_k_s(k_s),
    m_first_train_set_dim(first_ts),
    m_last_train_set_dim(last_ts),
    m_number_threads(number_threads)
    {  
      int m_n = X.cols();
      //evaluating row mean and saving it in the m_means
      m_means = (m_X.rowwise().sum())/m_n;
      
      
      
      //normalizing

      for (size_t i = 0; i < m_n; ++i)
      {
        m_X_cent.col(i) = m_X.col(i).array() - m_means;
      }
      
      // X * X' 
      m_Cov =  ((m_X_cent*m_X_cent.transpose()).array())/static_cast<double>(m_n);
      
      // trace of covariance
      double m_trace_cov = m_Cov.trace();
      
      int tot_CV_it_single_k = m_last_train_set_dim - m_first_train_set_dim + 1;
      
      m_ti_ts.resize(tot_CV_it_single_k);
      std::iota(m_ti_ts.begin(),m_ti_ts.end(),m_first_train_set_dim);
      
      m_toll = toll*m_trace_cov;
    }
  
  
  
  
  
  /*!
   * Getter for m_X
   */
  inline KE_Traits::StoringMatrix X() const {return m_X;};
  
  /*!
   * Getter for m_rho
   */
  inline KE_Traits::StoringMatrix psi_hat() const {return m_psi_hat;};
  
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
  
  inline auto pred() const {return m_pred;};

  
  /*!
   * Getter for m_number_threads
   */
  inline int number_threads() const {return m_number_threads;};
  
  
  void KE_CV_algo();

  
};

#include "KE_algo_CV_imp.hpp"


#endif  //KE_CV_HPP