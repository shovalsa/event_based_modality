{

  "dataset_reader": {
    "type": "sequence_tagging_mod",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
      "elmo": {
        "type": "elmo_characters"
     }
    }
  },
  "train_data_path": "data/4/train_fine.txt",
  "validation_data_path": "data/4/dev_fine.txt",
  "model": {
    "type": "crf_tagger_mod",
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
        "elmo":{
            "type": "elmo_token_embedder",
        "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
        "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
            "do_layer_norm": false,
            "dropout": 0.0
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 128,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu"
            }
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 1202,
      "hidden_size": 200,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
    "regularizer": {
      "regexes": [
        ["scalar_parameters", {"type": "l2", "alpha": 0.1}]
      ]

    }
  },
  "data_loader": {
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 3,
    },
    "num_epochs": 75,
    "grad_norm": 5.0,
    "validation_metric": "+tagging_f1",
    "patience": 9,
    "cuda_device": 0
  }
}