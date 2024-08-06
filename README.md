### CLIP DeepDream 🤖💭

## Update 06/AUG/24

- Added option to use penultimate (or earlier) layer output to all scripts.
- Added "run_full_PGD_dream_with_CLIP-conf_penultimate.py" as example for layer [-5].

![layer-5-example](https://github.com/user-attachments/assets/f1b99e1a-ffaa-49e1-8caa-71e596cbf6b7)

- Note: Due to the not-quite-intuitive way CLIP's neurons composit, results may be unexpected. For example:
- 👓 + US-FLAG = Waldo, and "Where is Waldo?" is also known as "horrororienteering" to CLIP's text transformer. 🙃

-----
## Update 02/AUG/24:

## 🌟🌠 🤯 Deep Dream with a *full* CLIP model! (*=> using output*) 🤯 🌌✨
### Or, in other words: CLIP as a stand-alone generative AI. Strangely. 🙃

- Gradient Ascent: Optimize Text embeddings for cosine similarity with image embeddings
- Projected Gradient Descent: Add Gaussian noise + Perturb towards CLIP 'opinion' (text)
- Config examples:

- `python run_full_PGD_dream_with_CLIP-conf_deepdream.py`
- Bound by strong influence of original image; 'deepdream-like'

- `python run_full_PGD_dream_with_CLIP-conf_unleashed.py`
- Creative freedom for CLIP's multimodal neurons

- `python run_full_PGD_dream_with_CLIP-conf_nobounds.py`
- Unbound & guided by text embeddings, original image features (edges) disappear

- Pass `--im path/to/img.jpg (.png, .jpeg)` to use your own image
- 👉 Check the code for detailed information! ✅
----
![PGD-demo-image-best](https://github.com/user-attachments/assets/03b203d4-bf0e-4d12-aaff-5b62f56eb517)

----

Detailed explanation: GA

- Use gradient ascent (rather than a human prompt) to optimize the text embeddings towards cosine similarity with the image embeddings.
- Reason: Imagine you have a photo of a cat. You use "cat" and get a good cosine similarity. You and CLIP agree... But do you? If you had tested "tabby", you would have seen an even better cosine similarity. And if you exchange the background behind the cat from a sofa to a forest - without altering the cat in the foreground - you'd have "wildcat" with the highest cosine similarity. Because CLIP considers the background, the entire image, to make meaning. It doesn't matter if it's still your same old fluffy tabby pet cat - it's "wildcat" now (CLIP's truth).
- Forcing human bias on CLIP with 'cat' can be useful in many scenarios. But not here. We want CLIP's textual 'opinion' of the image embeddings in the vision transformer. A best match.
- And CLIP's 'opinion' is not even really as simple and 'makes sense!' as the above example. You'll see when you run the code. =)
------
Detailed explanation: PGD

- Yes, that's technically an adversarial attack / -training method. But we're optimizing the image the opposite way.
- We're optimizing towards the text embeddings - the words describing the salient features / what CLIP 'saw' in the image.
- We're also heavily perturbing the image, for maximum visibility. And with heavy regularization. See code for details.
- Like looking at a complex thing, then drawing it from mind. But in high-dimensional tensorial AI-weirdness. =)

----

## 🌟🌠 Dreaming with CLIP Vision Transformers 🌌✨

- Basically, it's Feature Activation Max Visualization.
- But with an input image instead of Gaussian noise.
- And guided by neuron activation values (target feature = auto / CLIP's choice).
- And with 4 tiles so we get a larger image and more details.

As an image is worth 16x16 words, here's a CLIP Dream:

![clip-deepdream-my-ai](https://github.com/user-attachments/assets/6c9d8300-82cb-4dd4-b5f2-5a903496c3fd)


- Just run `python run_deepdream-with-CLIP.py` and start dreaming - check the code comments for how-to-use!
- Pass your own image: `python run_deepdream-with-CLIP.py --im path/to/img.png`

------
🌟🌠🌌✨ Delve Deeper: ✨🌌🌠🌟
------
CLIP Vision Transformer Model Structure:

```
(CLIP(
  (visual): VisionTransformer(
    (conv1): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
    (ln_pre): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    (transformer): Transformer(
      (resblocks): Sequential(


        (0): ResidualAttentionBlock(
          (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
          )
          (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (mlp): Sequential(
            (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
          )
          (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )

```
Above is resblock (Residual Attention Block) number (0), the first one (input layer), of which ViT-L/14 has 24 [0-23].

- MHA (Multi-Head Attention) + MLP (Multi-Layer Perceptron).
- MLP: c_fc (fully connected layer) -> GELU activation function -> projection layer.
- The "4096" in this case (CLIP ViT-L/14) - those are the AI's "features" that encode visual concepts.
------
- This code visualizes the features that have the highest activation value, w.r.t. the neurons that 'fire' most strongly during a forward pass ('salient features').
- Next, we are alas creating a Vision Transformer's Deep Dream, as CLIP will be "dreaming" (gradient ascent) with the neurons that amplify what is 'salient' to the model.
- That won't necessarily be what you expect, though. For example, a reflection on the retina of a cat might just activate an angel neuron in CLIP (true story, I had that happen!), so CLIP might end up dreaming an angel onto your cat. =)
------
## In general:

- The first layers have simple geometric structures, lines, zigzags. The intermediate layers have textures -> complex concepts. 
- The layers just at the output are overwhelmingly complex and not very aesthetic to humans ("a noisy mess of many many things").
- For aesthetic deep dreams, use "center + X". The number of layers for the model will be printed at the start of the run.
- For example, if the model has 24 layers [0-23], layers 12-19 would be most promising for 'subjective aesthetics'.
- A neuron may "fire" because of a particular thing in a small area of your input image (not present in all 4 tiles).
- If CLIP's features don't find anything to "latch onto" in the current tile, that tile may look "blurry" and lacking features.
------
- Built on the awesome code of [github.com/hamidkazemi22/vit-visualization](https://github.com/hamidkazemi22/vit-visualization)
- Original CLIP Gradient Ascent Script by Twitter / X: [@advadnoun](https://twitter.com/advadnoun)
- Requires / is made for [OpenAI/CLIP](https://github.com/openai/CLIP)
------
PS: I, a human, chose the word "delve". Very deliberately. The code was largely written by GPT-4o, based on my ideas (hybrid work). This readme.md was entirely written by human (me). I just love to cause a confusion and inverse roles, as GPT-4o never got to 'delve' in the code! =)
