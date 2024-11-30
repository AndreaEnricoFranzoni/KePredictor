# KePredictor
The package offer a very trivial, non efficient implementation of the Kernel Estimate algorithm to forecast time series of curves and surfaces.

The main function of this repository is to implement KE with CV to be able to compare it with [PPC KO algorithm](https://github.com/AndreaEnricoFranzoni/PPCforAutoregressiveOperator/tree/main). The instrunctions for installation can be found in the previous link.


~~~
PPC_KO( Rcpp::NumericMatrix           X,
        std::string                   id_CV         = "NoCV",
        double                        alpha         = 0.75,
        int                           k             = 0,
        double                        threshold_ppc = 0.95,
        Rcpp::Nullable<NumericVector> alpha_vec     = R_NilValue,
        Rcpp::Nullable<IntegerVector> k_vec         = R_NilValue,
        double                        toll          = 1e-4,
        Rcpp::Nullable<NumericVector> disc_ev       = R_NilValue,
        double                        left_extreme  = 0,
        double                        right_extreme = 1,
        Rcpp::Nullable<int>           min_size_ts   = R_NilValue,
        Rcpp::Nullable<int>           max_size_ts   = R_NilValue,
        int                           err_ret       = 0,
        Rcpp::Nullable<int>           num_threads   = R_NilValue,
        Rcpp::Nullable<std::string>   id_rem_nan    = R_NilValue)
~~~
