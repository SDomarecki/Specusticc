def ratio_to_class(ratio: float) -> int:
    # if ratio > 1.2:
    #     return 4  # Strong Buy
    # elif ratio > 1.05:
    #     return 3  # Buy
    # elif ratio > 0.95:
    #     return 2  # Hold
    # elif ratio > 0.8:
    #     return 1  # Sell
    # else:
    #     return 0  # Strong Sell

    if ratio > 1.05:
        return 3  # Buy
    elif ratio > 0.95:
        return 2  # Hold
    else:
        return 1  # Sell
