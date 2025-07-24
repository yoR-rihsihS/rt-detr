import json
import numpy as np

colors = np.random.randint(0, 255, size=(80, 3)).tolist()
with open("./coco_colors.json", 'w') as f:
    json.dump(colors, f, indent=4)