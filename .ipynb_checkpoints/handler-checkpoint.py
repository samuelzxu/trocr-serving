# Create model object
"""
ModelHandler defines a custom model handler.
"""

import torch
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from PIL import Image
import requests

from ts.torch_handler.base_handler import BaseHandler
model = None


def entry_point_function_name(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model
    global processor

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        # model_pt_path = os.path.join(model_dir, serialized_file)
        # if not os.path.isfile(model_pt_path):
        #     raise RuntimeError("Missing the model.pt file")

        model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-handwritten")
            # model = torch.jit.load(model_pt_path)
    else:
        # infer and return result
        generated_ids = model.generate(data)
        return generated_ids


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten")
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")
        image = Image.open(requests.get(str(preprocessed_data), stream=True).raw).convert("RGB")
        print(str(preprocessed_data))
        pixel_values = processor(image, return_tensors="pt").pixel_values
        return pixel_values


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        generated_text = processor.batch_decode(inference_output, skip_special_tokens=True)[0]
        return generated_text

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)