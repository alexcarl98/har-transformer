# Other paperks to look at

**Apparent deep learning models for har:**
- [4] F. Ordoñez and D. Roggen, “Deep convolutional and lstm recurrent neural networks for multimodal wearable activity recognition,” Sensors, vol. 16, no. 1, p. 115, 2016.
- [11] J. Yang, M. N. Nguyen, P. P. San, X. L. Li, and S. Krishnaswamy, “Deep convolutional neural networks on multichannel time series for human activity recognition,” in Proc. 24th Int. Joint Conf. Artif. Intell., 2015, pp. 3995–4001
- [12] L. Peng, L. Chen, Z. Ye, and Y. Zhang, “Aroma: A deep multi-task learning based simple and complex human activity recognition method using wearable sensors,” in Proc. ACM Interactive, Mobile, Wearable Ubiquitous Technol., 2018, pp. 1–16.
- [13] S. Yao, S. Hu, Y. Zhao, A. Zhang, and T. Abdelzaher, “Deepsense: A unified deep learning framework for time-series mobile sensing data processing,” in Proc. 26th Int. Conf. World Wide Web, 2017, pp. 351–360.




---
# Tsne


## ✅ t-SNE Visualization in Python (with PyTorch example)

This assumes:

* You have a trained model
* You can extract the **features** (embeddings) and **labels** from your test/validation set

---

### 🐍 Code

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

def extract_features(model, dataloader, device):
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model.extract_features(inputs)  # You define this
            features.append(outputs.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    features = np.concatenate(features)
    return features, np.array(labels)

def tsne_plot(features, labels, title='t-SNE Visualization'):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    # Encode labels if they are strings
    if not np.issubdtype(labels.dtype, np.number):
        labels = LabelEncoder().fit_transform(labels)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title(title)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

---

### 🧪 Example Usage:

```python
# After training your model
features, labels = extract_features(model, test_loader, device)
tsne_plot(features, labels, title="t-SNE on HAR Features")
```

---

## 🔧 Notes

* `model.extract_features(inputs)` should return intermediate layer output (not logits). For example:

  ```python
  def extract_features(self, x):
      x = self.encoder(x)  # Or wherever your deep features are
      return x
  ```
* You can adjust `perplexity`, `n_iter`, and `learning_rate` based on dataset size.
* If you use a transformer or LSTM, you might take the final hidden state or pooled token.

---

Would you like help modifying this to work with your specific HAR model architecture?
