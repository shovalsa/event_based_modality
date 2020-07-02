local bert_model = "bert-base-uncased";
local max_length = 128;

local bert_dim = 768;
{
    "dataset_reader": {
        "type": "bert_sequence_tagging_mod",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "max_length": 128,
                "model_name": "bert-base-uncased"
            }
        }
    },
    "model": {
        "type": "simple_tagger_mod",
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "dropout": 0.5,
            "hidden_size": 200,
            "input_size": 768,
            "num_layers": 1
        },
        "initializer": {
            "regexes": [
                [
                    ".*linear_layers.*weight",
                    {
                        "type": "xavier_normal"
                    }
                ],
                [
                    ".*scorer._module.weight",
                    {
                        "type": "xavier_normal"
                    }
                ],
                [
                    "_distance_embedding.weight",
                    {
                        "type": "xavier_normal"
                    }
                ],
                [
                    "_span_width_embedding.weight",
                    {
                        "type": "xavier_normal"
                    }
                ],
                [
                    "_context_layer._module.weight_ih.*",
                    {
                        "type": "xavier_normal"
                    }
                ],
                [
                    "_context_layer._module.weight_hh.*",
                    {
                        "type": "orthogonal"
                    }
                ]
            ]
        },
        "regularizer": {
            "regexes": [
                [
                    "scalar_parameters",
                    {
                        "alpha": 0.1,
                        "type": "l2"
                    }
                ]
            ]
        },
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "max_length": 128,
                    "model_name": "bert-base-uncased"
                }
            }
        }
    },
    "train_data_path": "data/0/train_fine.txt",
    "validation_data_path": "data/0/dev_fine.txt",
    "trainer": {
        "checkpointer": {
            "num_serialized_models_to_keep": 3
        },
        "cuda_device": 1,
        "grad_norm": 5,
        "num_epochs": 75,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 0.001,
            "parameter_groups": [
                [
                    [
                        ".*transformer.*"
                    ],
                    {
                        "lr": 1e-05
                    }
                ]
            ],
            "weight_decay": 0.01
        },
        "patience": 6,
        "validation_metric": "+tagging_f1"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 8,
            "padding_noise": 0
        }
    }
}
