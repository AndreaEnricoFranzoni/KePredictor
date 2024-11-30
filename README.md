# KePredictor
The package offer a very trivial, non efficient implementation of the Kernel Estimate algorithm to to perform one-step ahead prediction of time series of curves and surfaces.

The main function of this repository is to implement KE with CV to be able to compare it with [PPC KO algorithm](https://github.com/AndreaEnricoFranzoni/PPCforAutoregressiveOperator/tree/main). The instrunctions for installation can be found in the previous link.


~~~
KE_1d(Rcpp::NumericMatrix           X,
      std::string                   id_algo       = "KE",
      Rcpp::Nullable<IntegerVector> k_vec         = R_NilValue,
      double                        toll          = 1e-4,
      Rcpp::Nullable<int>           min_size_ts   = R_NilValue,
      Rcpp::Nullable<int>           max_size_ts   = R_NilValue,
      Rcpp::Nullable<int>           num_threads   = R_NilValue,
      Rcpp::Nullable<std::string>   id_rem_nan    = R_NilValue)
~~~
The input can be seen as the same of PPC, expect for `k`, that represents the number of retained fPCAs.
