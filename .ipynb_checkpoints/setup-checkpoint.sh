torch-model-archiver --model-name trocr --version 1.0 --handler handler.py --serialized-file none
mv trocr.mar models/
torchserve --start --model-store ./models/ --models all