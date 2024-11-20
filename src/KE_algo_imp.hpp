#include "KE_algo.hpp"





template< unsigned int IMP >
void
KE_algo<IMP>::KE_algorithm()
{
  int m = m_X.rows();
  int n = m_X.cols();
  

  
  Eigen::SelfAdjointEigenSolver<KE_Traits::StoringMatrix> eigensolver(m_Cov);
  
  
  //eigenvalues (descending order)
  KE_Traits::StoringVector eigvals = eigensolver.eigenvalues().reverse();
  
  double tot_eigvals = m_Cov.trace();
  
  m_explanatory_power.resize(m_k);
  for(std::size_t i = 0; i < m_k; ++i)
  {
    m_explanatory_power[i] = eigvals.head(i+1).sum()/tot_eigvals;
  }
 
  
  
  if constexpr(IMP == 1u)
  {
    double improvement = (3/2)*(eigvals(0)+eigvals(1));
    
    for(std::size_t i = 2; i < m_k; ++i){eigvals(i)=eigvals(i)+improvement;}
  }
  
  
  
  //spectral theorem
  std::transform(eigvals.begin(),eigvals.end(),
                 eigvals.begin(),
                 [](double s){return static_cast<double>(1)/std::sqrt(s);});
  
  
  KE_Traits::StoringMatrix L_alpha = eigvals.head(m_k).asDiagonal();
  const KE_Traits::StoringMatrix V_hat = eigensolver.eigenvectors().rowwise().reverse().leftCols(m_k);
  
  KE_Traits::StoringMatrix Psi_mat = (1/std::pow(m,2))*L_alpha*(V_hat.transpose())*m_CrossCov*V_hat;
  m_psi = (1/m)*V_hat*Psi_mat*V_hat.transpose();
  
}




template< unsigned int IMP >
KE_Traits::StoringArray
KE_algo<IMP>::prediction()
const
{
  return (m_psi*m_X.col(m_n-1)).array() + m_means;
}
