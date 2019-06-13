# Requirements
`pip install -r requirements.txt`

# How to run?
1. `mkdir models`
2. Download model from http://www.mediafire.com/file/i592s2jwasa3m5u/convnet_model_48x48_future.zip/file
3. Unzip model directory to folder models
4. Run: `python3 test.py`

# How to train?
1. Run: `mkdir -p dataset dataset_divided/train dataset_divided/test`
2. Setup `symbols.json` in utility
3. Run: `python3 divide_dataset.py`
4. Run: `python3 create_np_objects.py`
5. Run: `python3 train.py`