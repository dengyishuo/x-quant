# 定义环境重置函数
reset_strategy_env <- function() {
  if (exists(".strategy", envir = globalenv())) {
    strategy_env <- get(".strategy", envir = globalenv())
    strategies <- ls(strategy_env)
    for (strat in strategies) {
      if (strat != ".strategy") {
        rm.strat(strat)
      }
    }
  }

  if (exists(".blotter", envir = globalenv())) {
    blotter_env <- get(".blotter", envir = globalenv())
    objects <- ls(envir = blotter_env)
    for (obj in objects) {
      if (obj != ".blotter") {
        rm(list = obj, envir = blotter_env)
      }
    }
  }

  cat("策略环境已重置\n")
}
