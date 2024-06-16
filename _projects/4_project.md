---
layout: page
title: Cat image
description: the famous cat image within adversarial perturbation
img: assets/img/cat.jpg
importance: 3
category: fun
---

![cat](/assets/img/cat.jpg)

This is how to download the images:

```python
import PIL
import numpy as np
from urllib.request import urlretrieve
# img_path, _ = urlretrieve('https://www.anishathalye.com/media/2017/07/25/cat.jpg')
img_path, _ = urlretrieve('https://github.com/lorenz-peter/lorenz-peter.github.io/raw/master/assets/img/cat.jpg')
img_class = 281
img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
img = (np.asarray(img) / 255.0).astype(np.float32)
```

[Code](https://github.com/adverML/synthesizing_robust_adversarial)
