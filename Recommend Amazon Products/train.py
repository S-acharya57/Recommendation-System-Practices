import logging
from logging import getLogger
import recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model, get_trainer

config_dict = {
    # dataset config : Sequential Recommendation
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'RATING_FIELD' : 'rating',
    'load_col':{
        'inter': ['user_id', 'item_id','rating', 'timestamp']
    },
    'ITEM_LIST_LENGTH_FIELD': 'item_length',
    'LIST_SUFFIX': '_list',
    'MAX_ITEM_LIST_LENGTH': '5',

    # model config
    'embedding_size': '64',
    'hidden_size': '128',
    'num_layers': '1',
    'dropout_prob': '0.3',
    'loss_type': 'CE',

    # Training and evaluation config
    'epochs': '50',
    'train_batch_size': '4096',
    'eval_batch_size': '4096',
    'train_neg_sample_args': None,
    'eval_args':{
        'group_by': 'user',
        'order': 'TO',
        'split': {'LS': 'valid_and_test'},
        'mode': 'full',
},
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision'],
    'topk': '10',
    'valid_metric': 'MRR@10',
}

config = Config(model='GRU4Rec', dataset='amazon', config_dict=config_dict)

init_seed(config['seed'], config['reproducibility'])

# logger initialization
init_logger(config)
logger = getLogger()
# Create handlers
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
logger.addHandler(c_handler)

# write config info into log
logger.info(config)


print('\n\n\t\t\tCREATING DATASET\n\n\n')
dataset = create_dataset(config)
print(dataset)
logger.info(dataset)

print('\n\n\t\t\tSPLITTING DATASET\n\n\n')
train_data, valid_data, test_data = data_preparation(config, dataset)

# print(train_data.shape, valid_data.shape, test_data.shape)


for i in train_data:
    print(i)
    break

gru4rec_model = get_model(config["model"])
gru4rec_model, config["device"]


model = gru4rec_model(config, train_data.dataset).to(config['device'])
logger.info(model)
print(model)

trainer = Trainer(config, model)


best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=1)

# trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

# # When calculate ItemCoverage metrics, we need to run this code for set item_nums in eval_collector.
# trainer.eval_collector.data_collect(train_data)

# checkpoint_file = "saved/trained_model.pth"
# test_result = trainer.evaluate(test_data, model_file=checkpoint_file)
# print(test_result)
