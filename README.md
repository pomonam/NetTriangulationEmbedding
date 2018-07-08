# End-to-End Trainable Triangulation Embedding
This code implements end-to-end trainable Triangulation Embedding layer. The work is inspired by NetVLAD, an end-to-end trainable VLAD layer. 

The module was implemented & tested in TensorFlow 1.8.0. NetTriangulationEmbedding is distributed under Apache-2 License (see the `LICENCE` file). 

# Usage
NetTriangulationEmbedding has potential usages in image classification and retrieval tasks. It is applicable to end-to-end trainable models, which is an improvement from origianl Triangulation Embedding method described in [1].
```
from NetTriangulationEmbedding import TriangulationEmbeddingModule, 
                                      TemporalDifferenceDescriptors
import tensorflow as tf

te = TriangulationEmbedding(feature_size=1024, 
                            num_descriptors=30, 
                            num_anchor=16, 
                            add_batch_norm=True, 
                            is_training=True)

with tf.variable_scope("t_emb"):
  activation = te.forward(tensor)
```

# References
Please note that we are **not** the author of the following reference.
[1] Jégou, Hervé, and Andrew Zisserman. "Triangulation embedding and democratic aggregation for image search." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.

# Changes
- **1.00** (08 July 2018)
    - Initial public release
    
# Contributors
- [Juhan Bae](https://github.com/pomonam)
- [Ruijian An](https://github.com/RuijianSZ)

