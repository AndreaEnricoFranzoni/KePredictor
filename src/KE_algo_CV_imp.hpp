#include "KE_algo_CV.hpp"



template< unsigned int IMP >
void
KE_algo_cv<IMP>::KE_CV_algo()
{
  std::size_t tot_k = m_k_s.size();
  
  
  std::vector<double> valid_err;
  valid_err.reserve(tot_k);
  
  double previous_error(static_cast<double>(0));
  
  
  
  for(std::size_t i = 0; i < tot_k; ++i)
  {
    int k = m_k_s[i];
    
    std::vector<double> err_k;
    err_k.resize(m_ti_ts.size());
    
    int tot_iter = m_ti_ts.size();
    

    for(std::size_t j = 0; j < tot_iter; ++j)
    {
      KE_Traits::StoringMatrix train_set = m_X.leftCols(m_ti_ts[j]);
      KE_Traits::StoringMatrix valid_set = m_X.col(m_ti_ts[j]);
      
      KE_algo<IMP> ke(train_set,k,m_number_threads);
      ke.KE_algorithm();
      
      err_k[j] = mse<double>( valid_set.array() - ke.prediction() );
    }
    
    double curr_err = std::reduce(err_k.begin(),err_k.end(),0.0,std::plus<double>{});
    
    valid_err.emplace_back(curr_err);
    
    if(std::abs(curr_err - previous_error) < m_toll) {break;} else {previous_error = curr_err;}
    
  }
  
  
  //Shrinking
  valid_err.shrink_to_fit();
  
  //best validation error
  auto min_err = (std::min_element(valid_err.begin(),valid_err.end()));

  //optimal param
  m_k = m_k_s[std::distance(valid_err.begin(),min_err)];
  
  
  valid_err.clear(); 
  
  KE_algo<IMP> ke_best(m_X,m_k,m_number_threads);
  
  ke_best.KE_algorithm();
  
  m_pred = ke_best.prediction();
  m_explanatory_power = ke_best.explanatory_power();
}