# Vision-Language-Model

Topics

- Vision Transformer
- Contrastive Learning (CLIP, SigLip)
- Language Model (Gemma)
- KV-Chache
- Rotary Positional Encoding 
- Normalization (Batch, Layer and RMS)

[Contrastive Vision Encoder] ( SigLIP ) -> [Linear Projection] -> [ Transformer 
								  |
						[ Tokenizer ] --> |
								  |
								  L	


Image is split into blocks of pixels and each block, grid, will be converted into an embedding. 

These embedding will be concatinated with the tokens then feed into the decoder.


Contrastive Learning: take an encoded image and associated encoded text compute the dot product and where the two, text + image, correspond the model should output a high value while producing a low value for every other image text combination, within the associated row/column.

Hence, Cross Entropy Loss -> softmax to represent output as a distribution.

Numerical stability -> ensure the exponential does get to high.

Clip is very computationally expensive.

Treat problem as a binary classification task. Use Sigmoid. Allows parallelization aswell.

Vision Transformer:

	image apply convolution + flatten, then add positional encoding, learned parameter, feed it into transformer to add context.

Output: Contextualized Embeddings!


