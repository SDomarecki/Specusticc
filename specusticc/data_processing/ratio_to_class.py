def ratio_to_class(ratio: float, model_type: str):
    if model_type == 'neural_network':
        return ratio_to_class_number(ratio)
    elif model_type == 'decision_tree':
        return ratio_to_class_name(ratio)
    else:
        raise NotImplementedError


def ratio_to_class_name(ratio: float) -> str:
    if ratio > 1.2:
        return 'Strong Buy'
    elif ratio > 1.05:
        return 'Buy'
    elif ratio > 0.95:
        return 'Hold'
    elif ratio > 0.8:
        return 'Sell'
    else:
        return 'Strong Sell'


def ratio_to_class_number(ratio: float) -> int:
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


def ratio_to_class_array(ratio: float) -> []:
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
