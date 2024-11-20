#ifndef KE_CV_EVAL_VALID_ERR_HPP
#define KE_CV_EVAL_VALID_ERR_HPP

#include <algorithm>
#include <numeric>

#include "traits_ke.hpp"




template<typename T>
double mse(const KE_Traits::StoringVector &diff)
{
  double mse = 0.0;
  int num_comp = diff.size();
  


  mse = std::transform_reduce(diff.begin(),
                              diff.end(),
                              0.0,
                              std::plus{},
                              [] (T const &x) {return std::pow(x,2);});

  
  return mse/static_cast<double>(num_comp);
};

#endif /*KE_CV_EVAL_VALID_ERR_HPP*/