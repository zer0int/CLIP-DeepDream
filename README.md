### CLIP DeepDream 🤖💭
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
- Requires / is made for [OpenAI/CLIP](https://github.com/openai/CLIP)
------
PS: I, a human, chose the word "delve". Very deliberately. The code was largely written by GPT-4o, based on my ideas (hybrid work). This readme.md was entirely written by human (me). I just love to cause a confusion and inverse roles, as GPT-4o never got to 'delve' in the code! =)
