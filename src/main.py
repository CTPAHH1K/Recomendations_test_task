from pathlib import Path

import numpy as np
from ml_metrics import mapk

from src.load import DataProvider
from src.utils import get_model, transform_to_item_user_csr_matrix, get_recommendations, get_purchases_by_customer


def main():
    data_provider = DataProvider(data_directory=Path('./data'))
    item_users = transform_to_item_user_csr_matrix(data_provider.get_purchases_train())

    # baseline model
    model = get_model()
    np.random.seed(42)
    model.fit(item_users=item_users)

    test_user_ids, test_purchases = get_purchases_by_customer(data_provider.get_purchases_test())
    recommendations = get_recommendations(model, test_user_ids, item_users)
    score = mapk(test_purchases, recommendations, k=10)
    return score
