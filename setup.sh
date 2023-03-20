torch-model-archiver --model-name trocr-handwritten --version 1.0 --requirements-file ./requirements.txt --handler handler.py --serialized-file ./none
mv trocr-handwritten.mar models/
torchserve --start --model-store ./models/ --ts-config ./config.properties --models all
