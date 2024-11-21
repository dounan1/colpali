import torch
from PIL import Image
from io import BytesIO
import base64

from colpali_engine.models import ColQwen2, ColQwen2Processor
model_name = "vidore/colqwen2-v1.0"

class EndpointHandler:
    def __init__(self, path=""):
        self.model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # This will use CUDA if available, otherwise CPU
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(model_name)

    def __call__(self, data):
        # Extract inputs from the request data
        images_data = data.pop("images", [])
        queries = data.pop("queries", [])

        # Process images
        images = []
        for img_data in images_data:
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes))
            images.append(img)

        # Process the inputs
        batch_images = self.processor.process_images(images).to(self.model.device)
        batch_queries = self.processor.process_queries(queries).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
            query_embeddings = self.model(**batch_queries)

        # Calculate scores
        scores = self.processor.score_multi_vector(query_embeddings, image_embeddings)

        return {"scores": scores.tolist()}
