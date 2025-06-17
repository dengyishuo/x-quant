# 从tradeStats输出中计算胜率的函数
calculate_portfolio_win_rates <- function(trade_stats_output) {
  # 检查输入是否为tradeStats的输出格式
  if (!is.data.frame(trade_stats_output) || nrow(trade_stats_output) == 0) {
    stop("输入必须是tradeStats函数的输出数据框")
  }

  # 提取关键列
  total_trades <- trade_stats_output$Num.Trades
  win_trades <- ifelse("Win_Trades" %in% colnames(trade_stats_output),
    trade_stats_output$Win_Trades,
    trade_stats_output$Num.Trades * trade_stats_output$Percent.Positive / 100
  )

  # 计算胜率
  win_rate <- ifelse(total_trades > 0, win_trades / total_trades, 0)

  # 创建结果数据框
  result <- data.frame(
    Asset = rownames(trade_stats_output),
    Total_Trades = total_trades,
    Win_Trades = win_trades,
    Win_Rate = win_rate,
    Win_Rate_Percentage = paste0(round(win_rate * 100, 2), "%")
  )

  # 打印结果
  cat("=== 基于tradeStats的胜率计算结果 ===\n")
  # 返回结果
  return(result)
}

# 使用示例
# 假设ts_output是tradeStats(portfolio.st)的输出
# win_rate_result <- calculate_win_rate(ts_output)
