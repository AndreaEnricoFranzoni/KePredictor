#include <RcppEigen.h>

#include <string>
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
Rcpp::List KE_1d(Rcpp::NumericMatrix           X,
                  std::string                   id_algo       = "KE",
                  Rcpp::Nullable<IntegerVector> k_vec         = R_NilValue,
                  double                        toll          = 1e-4,
                  Rcpp::Nullable<int>           min_size_ts   = R_NilValue,
                  Rcpp::Nullable<int>           max_size_ts   = R_NilValue,
                  Rcpp::Nullable<int>           num_threads   = R_NilValue,
                  Rcpp::Nullable<std::string>   id_rem_nan    = R_NilValue
)
{ 
  //1D DOMAIN
  using T = double;       //version for real-values time series
  
  
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
Rcpp::List KEI_1d(Rcpp::NumericMatrix           X,
                  std::string                   id_algo       = "KE",
                  Rcpp::Nullable<IntegerVector> k_vec         = R_NilValue,
                  double                        toll          = 1e-4,
                  Rcpp::Nullable<int>           min_size_ts   = R_NilValue,
                  Rcpp::Nullable<int>           max_size_ts   = R_NilValue,
                  Rcpp::Nullable<int>           num_threads   = R_NilValue,
                  Rcpp::Nullable<std::string>   id_rem_nan    = R_NilValue
)
{ 
  //1D DOMAIN
  //1D DOMAIN
  using T = double;       //version for real-values time series
  
  
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








//
// [[Rcpp::export]]
Rcpp::List KE_2d(Rcpp::NumericMatrix           X,
                 std::string                   id_algo          = "KE",
                     int                           k                = 0, 
                     double                        threshold_ppc    = 0.95,
                     Rcpp::Nullable<IntegerVector> k_vec            = R_NilValue,
                     double                        toll             = 1e-4,
                     int                           num_disc_ev_x1   = 10,
                     int                           num_disc_ev_x2   = 10,
                     Rcpp::Nullable<int>           min_size_ts      = R_NilValue,
                     Rcpp::Nullable<int>           max_size_ts      = R_NilValue,
                     Rcpp::Nullable<int>           num_threads      = R_NilValue,
                     Rcpp::Nullable<std::string>   id_rem_nan       = R_NilValue
)
{ 
  //2D DOMAIN
  using T = double;       //version for real-values time series
  
  
  std::vector<int> k_s               = wrap_k_vec(k_vec,X.nrow());
  auto sizes_CV_sets                 = wrap_sizes_set_CV(min_size_ts,max_size_ts,X.ncol());
  int min_dim_train_set              = sizes_CV_sets.first;
  int max_dim_train_set              = sizes_CV_sets.second;
  const REM_NAN id_RN                = wrap_id_rem_nans(id_rem_nan);
  
  auto data_read = reader_data<T>(X,id_RN);
  KE_Traits::StoringMatrix x = data_read.first;
  
  KE_algo_cv<2u> ke_solver(std::move(x),k_s,min_dim_train_set,max_dim_train_set,toll, 1);
  ke_solver.KE_CV_algo();
  
  auto one_step_ahead_pred  = from_col_to_matrix(add_nans_vec(ke_solver.pred(),data_read.second,X.nrow()),num_disc_ev_x1,num_disc_ev_x2);  //estimate of the prediction (NaN for the points in which you do not have measurements)
  int ret_comp = ke_solver.k();
  auto exp_pow = ke_solver.explanatory_power();
  
  
  //returning element
  Rcpp::List l;
  
  l["One-step ahead prediction"] = one_step_ahead_pred;
  l["Number of Components retained"]   = ret_comp;
  l["Explained variance"]        = exp_pow;
  
  return l;
}





//
// [[Rcpp::export]]
Rcpp::List KEI_2d(Rcpp::NumericMatrix           X,
                 std::string                   id_algo          = "KE",
                 int                           k                = 0, 
                 double                        threshold_ppc    = 0.95,
                 Rcpp::Nullable<IntegerVector> k_vec            = R_NilValue,
                 double                        toll             = 1e-4,
                 int                           num_disc_ev_x1   = 10,
                 int                           num_disc_ev_x2   = 10,
                 Rcpp::Nullable<int>           min_size_ts      = R_NilValue,
                 Rcpp::Nullable<int>           max_size_ts      = R_NilValue,
                 Rcpp::Nullable<int>           num_threads      = R_NilValue,
                 Rcpp::Nullable<std::string>   id_rem_nan       = R_NilValue
)
{ 
  //2D DOMAIN
  using T = double;       //version for real-values time series
  
  
  std::vector<int> k_s               = wrap_k_vec(k_vec,X.nrow());
  auto sizes_CV_sets                 = wrap_sizes_set_CV(min_size_ts,max_size_ts,X.ncol());
  int min_dim_train_set              = sizes_CV_sets.first;
  int max_dim_train_set              = sizes_CV_sets.second;
  const REM_NAN id_RN                = wrap_id_rem_nans(id_rem_nan);
  
  auto data_read = reader_data<T>(X,id_RN);
  KE_Traits::StoringMatrix x = data_read.first;
  
  KE_algo_cv<1u> ke_solver(std::move(x),k_s,min_dim_train_set,max_dim_train_set,toll, 1);
  ke_solver.KE_CV_algo();
  
  auto one_step_ahead_pred  = from_col_to_matrix(add_nans_vec(ke_solver.pred(),data_read.second,X.nrow()),num_disc_ev_x1,num_disc_ev_x2);  //estimate of the prediction (NaN for the points in which you do not have measurements)
  int ret_comp = ke_solver.k();
  auto exp_pow = ke_solver.explanatory_power();
  
  
  //returning element
  Rcpp::List l;
  
  l["One-step ahead prediction"] = one_step_ahead_pred;
  l["Number of Components retained"]   = ret_comp;
  l["Explained variance"]        = exp_pow;
  
  return l;
}







//
// [[Rcpp::export]]
Rcpp::NumericMatrix data_2d_wrapper_from_list_ke(Rcpp::List Xt)
{
  using T = double;
  
  //this works only for 1-step time series
  int number_time_instants = Xt.size();
  if(number_time_instants==0)
  {
    std::string error_message1 = "Empty list";
    throw std::invalid_argument(error_message1);
  }
  
  int number_point_evaluations = as<NumericMatrix>(Xt[0]).nrow()*as<NumericMatrix>(Xt[0]).ncol();
  if(number_point_evaluations==0)
  {
    std::string error_message2 = "List of empty matrices";
    throw std::invalid_argument(error_message2);
  }
  
  Rcpp::NumericMatrix x(number_point_evaluations,number_time_instants);
  
  for(std::size_t i = 0; i < static_cast<std::size_t>(number_time_instants); ++i)
  {
    KE_Traits::StoringMatrix col = reader_data<T>(as<NumericMatrix>(Xt[i]),REM_NAN::NR).first;
    KE_Traits::StoringVector first_col = from_matrix_to_col(col);
    Rcpp::NumericVector col_wrapped = Rcpp::wrap(first_col);
    x(_,i) = col_wrapped;
  }
  
  return x;
}
