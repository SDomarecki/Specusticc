def ratio_to_class(ratio: float) -> []:
    return _ratio_to_5_class_array(ratio)


def _ratio_to_3_class_array(ratio: float) -> []:
    if ratio > 1.05:
        return [0, 0, 1]
    elif ratio > 0.95:
        return [0, 1, 0]
    else:
        return [1, 0, 0]


def _ratio_to_5_class_array(ratio: float) -> []:
    if ratio > 1.2:
        return [0, 0, 0, 0, 1]  # Strong Buy
    elif ratio > 1.05:
        return [0, 0, 0, 1, 0]  # Buy
    elif ratio > 0.95:
        return [0, 0, 1, 0, 0]  # Hold
    elif ratio > 0.8:
        return [0, 1, 0, 0, 0]  # Sell
    else:
        return [1, 0, 0, 0, 0]  # Strong Sell


def _ratio_to_5_class(ratio: float) -> int:
    if ratio > 1.2:
        return 4  # Strong Buy
    elif ratio > 1.05:
        return 3  # Buy
    elif ratio > 0.95:
        return 2  # Hold
    elif ratio > 0.8:
        return 1  # Sell
    else:
        return 0  # Strong Sell


def _ratio_to_3_class(ratio: float) -> int:
    if ratio > 1.05:
        return 2  # Buy
    elif ratio > 0.95:
        return 1  # Hold
    else:
        return 0  # Sell
