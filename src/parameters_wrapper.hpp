#ifndef KE_WRAP_PARAMS_HPP
#define KE_WRAP_PARAMS_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <utility>
#include <string>
#include <stdexcept>





//utilities to wrap the input parameters of the R function

//algo implemented
struct CV_algo
{
  static constexpr std::string CV1 = "KE";
  static constexpr std::string CV2 = "KEI";
};


inline 
std::string
wrap_string_CV_to_be_printed(const std::string &id_cv)
{
  CV_algo cv_algo;
  
  if(id_cv==cv_algo.CV1){  return "normal estimate";}
  if(id_cv==cv_algo.CV2){  return "improved estimate";}
  else
  {
    std::string error_message = "Wrong input string";
    throw std::invalid_argument(error_message);
  }
}


//check that threshold_ppc is in the correct range
inline
void
check_threshold_ppc(const double &threshold_ppc)
{
  if(threshold_ppc<=0 || threshold_ppc>=1)
  {
    std::string error_message = "threshold_ppc has to be in (0,1)";
    throw std::invalid_argument(error_message);
  }
}







//check that k is in the correct range
inline
void
check_k(const int &k, const int &max_k)
{
  if( k < 0 )
  {
    std::string error_message = "k has to be a positive integer or 0";
    throw std::invalid_argument(error_message);
  }
  if( k > max_k )
  {
    std::string error_message = "k has to be lower than the maximum number of PPCs";
    throw std::invalid_argument(error_message);
  }
}







//to wrap the vector of k passed in input
inline
std::vector<int>
wrap_k_vec(Rcpp::Nullable<Rcpp::IntegerVector> k_vec, int k_max)
{
  //if no k is given, the default is looking for all the possible directions (number of discrete evaluations innthe domain of the functional object)
  if(k_vec.isNull())
  {
    std::vector<int> k_s;
    k_s.resize(k_max);
      
    std::iota(k_s.begin(),k_s.end(),static_cast<int>(1));
      
    return k_s;
  }
    
    
  std::vector<int> k_s = Rcpp::as<std::vector<int>>(k_vec);
    
  //sorting into ascending order the alphas to be coherent during the algorithm
  std::sort(k_s.begin(), k_s.end());
  
  //checking
  if(k_s[0] < 1)
  {
    std::string error_message1 = "k has to be at least 1";
    throw std::invalid_argument(error_message1);
  }
  if(k_s.back() > k_max)
  {
    std::string error_message2 = "k cannot be greater than the number of discrete evaluation of the functional object in the domain (" + std::to_string(k_max) + ")";
    throw std::invalid_argument(error_message2);
  }
    
  return k_s;
}



//to wrap the min a max dimension of ts
inline
std::pair<int,int>
wrap_sizes_set_CV(Rcpp::Nullable<int> min_size_ts, Rcpp::Nullable<int> max_size_ts, int number_time_instants)    //dim: row of x
{ 
  int min_dim_ts = min_size_ts.isNull() ? static_cast<int>(std::ceil(static_cast<double>(number_time_instants)/static_cast<double>(2)))  : Rcpp::as<int>(min_size_ts);
  int max_dim_ts = max_size_ts.isNull() ? number_time_instants  : (Rcpp::as<int>(max_size_ts)+1); 
    
  if (!(min_dim_ts>1))
  {
    std::string error_message1 = "Min size of train set has to be at least 2";
    throw std::invalid_argument(error_message1);
  }
    
  if (!(max_dim_ts<=number_time_instants))
  {
    std::string error_message2 = "Max size of train set has to be at most the total number of time instants ("  + std::to_string(number_time_instants) + ") minus 1 (to leave room for the validation set)";
    throw std::invalid_argument(error_message2);
  }
    
  if(min_dim_ts >= max_dim_ts)
  {
    std::string error_message = "Min size of train set (" + std::to_string(min_dim_ts) + " has to be less than the max one ("  + std::to_string(max_dim_ts) + ")";
    throw std::invalid_argument(error_message);
  }
    
  return std::make_pair(min_dim_ts,max_dim_ts);
}






//removing NaNs
enum REM_NAN
{ 
  NR = 0,      //not replacing NaN
  MR = 1,      //replacing nans with mean (easily changes the mean of the distribution)
  ZR = 2,      //replacing nans with 0s (easily changes the sd of the distribution)
};


//reads the input string an gives back the correct value of the enumerator for replacing nans
inline
REM_NAN
wrap_id_rem_nans(Rcpp::Nullable<std::string> id_rem_nan)
{
  if(id_rem_nan.isNull())
  { 
    return REM_NAN::MR;
  }
  if(Rcpp::as< std::string >(id_rem_nan) == "NO")
  {
    return REM_NAN::NR;
  }
  if(Rcpp::as< std::string >(id_rem_nan) == "MR")
  {
    return REM_NAN::MR;
  }
  if(Rcpp::as< std::string >(id_rem_nan) == "ZR")
  {
    return REM_NAN::ZR;
  }
  else
  {
    std::string error_message = "Wrong input string for handling NANs";
    throw std::invalid_argument(error_message);
  }
  
};


#endif  /*KE_WRAP_PARAMS_HPP*/