import numpy as np
import timm
import torch
from tqdm import tqdm
import utils
import config
from os.path import join
import pandas as pd

# Preparing data
test_loader = utils.get_data_loader(
    data_path=config.data_path,
    data_file=config.test_data_filename,
    phase="test"
)

# Preparing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = timm.create_model(config.model_name, pretrained=True, num_classes=4)
model.load_state_dict(torch.load(str(join(config.data_path, config.checkpoint_filename))))
model.to(device)
model.eval()

# Testing
results = []
image_ids = []

with torch.no_grad():
    for images, ids in tqdm(test_loader):
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        results.extend(probs)
        image_ids.extend(ids)

# Creating a submission file
submission = pd.DataFrame(results, columns=["healthy", "multiple_diseases", "rust", "scab"])
submission.insert(0, "image_id", np.asarray(image_ids))
submission.to_csv(str(join(config.data_path, config.submission_filename)), index=False)
