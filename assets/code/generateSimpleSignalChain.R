generateSimpleSignalChain <- function(txn_data, type = "character") {
  # 确保数据按时间排序
  txn_data <- txn_data[order(index(txn_data)), ]

  # 验证type参数有效性
  if (!type %in% c("character", "numeric")) {
    stop("type参数必须为'character'或'numeric'")
  }

  # 初始化信号向量
  signals <- rep(NA, nrow(txn_data))

  # 根据Txn.Qty列生成信号
  for (i in 1:nrow(txn_data)) {
    if (txn_data$Txn.Qty[i] > 0) {
      signals[i] <- ifelse(type == "character", "Buy", 1)
    } else if (txn_data$Txn.Qty[i] < 0) {
      signals[i] <- ifelse(type == "character", "Sell", -1)
    } else {
      signals[i] <- ifelse(type == "character", "Hold", 0)
    }
  }

  # 创建与原始时间戳一致的信号链
  signal_chain <- xts(signals, order.by = index(txn_data))
  colnames(signal_chain) <- "Signal"

  return(signal_chain)
}
