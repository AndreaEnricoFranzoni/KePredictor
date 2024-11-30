#include <RcppEigen.h>

#include <string>
#include <iostream>
#include "traits_ke.hpp"
#include "parameters_wrapper.hpp"
#include "utils.hpp"
#include "data_reader.hpp"

#include "KE_algo_CV.hpp"



using namespace Rcpp;


//
// [[Rcpp::depends(RcppEigen)]]



//
// [[Rcpp::export]]
Rcpp::List KE(Rcpp::NumericMatrix           X,
              Rcpp::Nullable<IntegerVector> k_vec         = R_NilValue,
              double                        toll          = 1e-4,
              Rcpp::Nullable<int>           min_size_ts   = R_NilValue,
              Rcpp::Nullable<int>           max_size_ts   = R_NilValue,
              Rcpp::Nullable<int>           num_threads   = R_NilValue,
              Rcpp::Nullable<std::string>   id_rem_nan    = R_NilValue)
{ 
  //1D DOMAIN
  using T = double;       //version for real-values time series
  Rcout << "Kernel Estimate predictor with CV for retained number of PCs" << std::endl;
  
  
  std::vector<int> k_s               = wrap_k_vec(k_vec,X.nrow());
  auto sizes_CV_sets                 = wrap_sizes_set_CV(min_size_ts,max_size_ts,X.ncol());
  int min_dim_train_set              = sizes_CV_sets.first;
  int max_dim_train_set              = sizes_CV_sets.second;
  const REM_NAN id_RN                = wrap_id_rem_nans(id_rem_nan);

  auto data_read = reader_data<T>(X,id_RN);
  KE_Traits::StoringMatrix x = data_read.first;
  
  KE_algo_cv<2u> ke_solver(std::move(x),k_s,min_dim_train_set,max_dim_train_set,toll, 1);
  ke_solver.KE_CV_algo();
  
  auto pred = ke_solver.pred();
  int ret_comp = ke_solver.k();
  auto exp_pow = ke_solver.explanatory_power();
  
  
  //returning element
  Rcpp::List l;

  l["One-step ahead prediction"] = pred;
  l["Number of Components retained"]   = ret_comp;
  l["Explained variance"]        = exp_pow;

  return l;
}



//
// [[Rcpp::export]]
Rcpp::List KEI(Rcpp::NumericMatrix           X,
               Rcpp::Nullable<IntegerVector> k_vec         = R_NilValue,
               double                        toll          = 1e-4,
               Rcpp::Nullable<int>           min_size_ts   = R_NilValue,
               Rcpp::Nullable<int>           max_size_ts   = R_NilValue,
               Rcpp::Nullable<int>           num_threads   = R_NilValue,
               Rcpp::Nullable<std::string>   id_rem_nan    = R_NilValue)
{ 
  //1D DOMAIN
  //1D DOMAIN
  using T = double;       //version for real-values time series
  Rcout << "Improved Kernel Estimate predictor with CV for retained number of PCs" << std::endl;
  
  std::vector<int> k_s               = wrap_k_vec(k_vec,X.nrow());
  auto sizes_CV_sets                 = wrap_sizes_set_CV(min_size_ts,max_size_ts,X.ncol());
  int min_dim_train_set              = sizes_CV_sets.first;
  int max_dim_train_set              = sizes_CV_sets.second;
  const REM_NAN id_RN                = wrap_id_rem_nans(id_rem_nan);
  
  auto data_read = reader_data<T>(X,id_RN);
  KE_Traits::StoringMatrix x = data_read.first;
  
  KE_algo_cv<1u> ke_solver(std::move(x),k_s,min_dim_train_set,max_dim_train_set,toll, 1);
  ke_solver.KE_CV_algo();
  
  auto pred = add_nans_vec(ke_solver.pred(),data_read.second,X.nrow());
  int ret_comp = ke_solver.k();
  auto exp_pow = ke_solver.explanatory_power();
  
  
  //returning element
  Rcpp::List l;
  
  l["One-step ahead prediction"] = pred;
  l["Number of Components retained"]   = ret_comp;
  l["Explained variance"]        = exp_pow;
  
  return l;
}