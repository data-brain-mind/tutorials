---
layout: distill

title: "The Hidden Cost of Downsampling: Rethinking Resolution within Medical Deep Learning"

description: "Deep learning models have transformed medical imaging, yet their reliance on standardized input often obscures a fundamental challenge: the critical impact of downsampling operations. This blog post shifts focus from initial data preprocessing to the often-overlooked internal downsampling within neural networks. We analyze how ubiquitous operations like MaxPooling and strided convolutions can introduce significant aliasing artifacts and information loss, compromising diagnostic accuracy. Exploring advanced alternatives such as BlurPooling and k-space downsampling, we highlight their potential to mitigate these pitfalls. This analysis underscores the importance of a mindful approach to internal network design to foster more robust, accurate, and trustworthy AI solutions in medical imaging."

date: 2025-09-08

future: true

htmlwidgets: true

# hidden: true

authors:
  - name: Anonymous
  - name: Anonymous
  - name: Anonymous

bibliography: brain_sampling.bib  

toc:
- name: "The Foundation of Deep Learning : Standard Input Formats"
- name: "Beyond Preprocessing: The Crucial Role of Downsampling within Deep Learning Networks"
- name: "Rethinking Downsampling: Smarter Alternatives"
- name: "Summarizing Thoughts and Takeaways"
---

In the rapidly evolving landscape of medical imaging and artificial intelligence, deep learning models have demonstrated unprecedented capabilities <d-cite key="ronneberger2015u"></d-cite>. Yet, beneath the surface of impressive benchmarks lies a critical, often-overlooked challenge: the internal handling of image resolution. Most computer vision models work best with standardized images, <d-cite key="he2015spatial"></d-cite> but medical imaging is far from standardized. Even within a single patient's brain scan, different types of scans can produce varying resolutions and levels of detail <d-cite key="wakeman2015multi"></d-cite>. The challenge and the key to a correct diagnosis is learning to interpret and use all of them in tandem.

Traditionally, the focus has been on initial preprocessing, by e.g. rescaling images to fit model requirements. This process is very transparent, showing exactly how the input images are changed. This is a key benefit, as it allows for verification that the images still contain all the relevant information and that the downscaling hasn't introduced any significant artifacts.
More critically the downsampling operations *within* our deep learning architectures can exacerbate these issues, stealthily degrading image fidelity and reintroducing distortions that clinicians cautiously avoid during data acquisition and preprocessing.

This blog post aims to shine some light onto this pivotal area, moving beyond basic preprocessing to critically examine the ubiquitous, yet often problematic, downsampling mechanisms embedded within state-of-the-art deep learning models. We will explore how these internal operations impact data integrity and discuss innovative strategies to mitigate their "hidden cost." Join us as we uncover how rethinking downsampling within our networks can pave the way for more robust, accurate, and clinically relevant AI solutions.

## The Foundation of Deep Learning: Standard Input Formats

At the heart of most deep learning architectures, lies the expectation of standardized input. Typically, this means data presented as RGB or grayscale images, often in common formats like PNG or JPEG in the image domain and NIfTI for medical image data. The process of loading this kind of data has become remarkably streamlined, with pre-defined libraries such as OpenCV <d-cite key="opencv"></d-cite> or NiBabel <d-cite key="nibabel"></d-cite> facilitating the conversion into PyTorch tensors, ready for immediate processing and analysis by the neural network. A key challenge with medical images is that even a single patient's brain scans can have different resolutions, simply because various scan types are used.
Consider the stark contrast in visual information presented by different medical imaging modalities:

{% include figure.html path="fig1.png" class="img-fluid" %}
Figure 1. Exemplary MR image slices with different contrasts: Magnetization Prepared Rapid Gradient Echo (MPRAGE), Fast Low Angle Shot (FLASH), Diffusion-Weighted Imaging (DWI), and Blood Oxygenation Level Dependent imaging (BOLD). Physicians rely on such contrasts for accurate diagnosis of pathologies. However, physical limitations and acquisition strategies impose resolution constraints on these images, resulting in varying resolutions. Therefore, proper pre-processing is crucial to ensure accurate processing and analysis by the neural network. (Brain scans source <d-cite key="wakeman2015multi"></d-cite>).


As demonstrated in Figure 1, each contrast provides unique insights at different scales. An MPRAGE scan can capture fine anatomical detail of the brain, whereas a DWI or BOLD scan highlights physiological processes, though typically at a lower spatial resolution. Attempting to uniformly resize these disparate inputs to a single dimension without careful consideration, can degrade the quality of higher-resolution images or introduce artificial information into lower-resolution ones. However, this initial preprocessing is only the first step. A far more insidious and often overlooked issue lies in the resizing and downsampling operations inherently performed within many deep learning networks themselves. This internal, often opaque, data manipulation is where the true, hidden challenge and the central focus of this blog post truly begins.

## Beyond Preprocessing: The Crucial Role of Downsampling within Deep Learning Networks

While external data preprocessing is a widely addressed topic, especially in workshops like this, an equally critical set of data manipulation methods occurs within the deep learning models themselves: sampling. Specifically, the ubiquitous operations of downsampling and upsampling. In the initial preprocessing stage, a user can still visually inspect and judge the quality of an image after transformation. This direct oversight, however, is lost once an image enters the opaque layers of a deep neural network.

{% include figure.html path="fig2.png" class="img-fluid" %}
Figure 2. Unmasking internal downsampling pitfalls: MaxPooling (middle) fundamentally discards critical diagnostic information, while strided convolution (right) reintroduces aliasing artifacts, appearing as scattered edges and subtle grid patterns. Both demonstrate how seemingly innocuous operations within deep learning networks can degrade the fidelity of medical images. (Brain scans source <d-cite key="wakeman2015multi"></d-cite>).

Consider the U-Net <d-cite key="ronneberger2015u"></d-cite>, a widely adopted architecture for medical image segmentation, which incorporates simple downsampling operations in the form of MaxPooling. While efficient, MaxPooling fundamentally discards information, potentially defacing diagnostically important information as demonstrated in Figure 2 (middle). As networks often employ several hierarchical layers of such operations, even just two consecutive applications, as shown in Figure 2, can lead to a complete destruction of recognizable patterns, rendering the original data unrecognizable and compromising the integrity of the analysis.
Substituting MaxPooling with strided convolutional operations (stride of two) can better preserve image structure. However, this naive application of strided convolutions can introduce aliasing artifacts as illustrated in Figure 2 (right). While clinicians cautiously try to mitigate aliasing during image acquisition, these artifacts are reintroduced by standard strided convolution operations within the network. Research in computer vision indicates that these aliasing artifacts increase network susceptibility to noise, corruptions <d-cite key="vasconcelos2021impact"></d-cite>, and adversarial attacks <d-cite key="grabinski2022aliasing"></d-cite>.
These intrinsic architectural pitfalls can lead to significant information loss and erroneous predictions in medical diagnostic applications. Consequently, model predictions may become unreliable, as the network could be reacting to internally generated, network-specific artifacts rather than accurately representing the nuanced content of the original data. Therefore, the way we sample internally should be examined just as closely as our initial data preprocessing, since it directly affects how accurate and trustworthy our AI models are.

## Rethinking Downsampling: Smarter Alternatives

We've established that the way we downsample in our networks is crucial. Now, let's explore some alternative downsampling methods designed to mitigate aliasing, a common problem stemming from traditional downsampling. Keep in mind, though, these alternatives also come with their own set of compromises, which we'll delve into later.
The simplest way to reduce aliasing is to use traditional AveragePooling. Think of it as a gentle blurring step before downsampling. Blurring helps reduce high-frequency noise, which is a primary cause of aliasing. However, standard AveragePooling typically uses small 2x2 kernels, limiting its effectiveness against aliasing. While larger kernels would offer more blurring, they risk over-blurring the image features and losing important fine details.

{% include figure.html path="blurpool_method.png" class="img-fluid" %}
Figure 3. BlurPooling method <d-cite key="zhang2019making"></d-cite>: This approach reduces aliasing by first applying an original operation (e.g., here MaxPooling) with a stride of one, followed by a convolution with fixed, Gaussian-like blur kernels and a stride of two. This strategic blurring step effectively suppresses high-frequency information that could cause aliasing before the downsampling occurs, preserving image fidelity.

A more effective approach that builds on the blurring concept is BlurPooling <d-cite key="zhang2019making"></d-cite>. This method employs pre-defined blur kernels, usually of size 5x5, to blur the feature maps before downsampling. This technique generally preserves more of the original information compared to standard AveragePooling. The process of BlurPooling is depicted in Figure 3. The implementation of BlurPooling is straight forward an can easily be used as described in the following code <d-cite key="zhang2019makingGIT"></d-cite>:

```python
import antialiased_cnns

C = 5

# MaxPool --> MaxBlurPool
baseline = nn.MaxPool2d(kernel_size=2, stride=2)
antialiased = [nn.MaxPool2d(kernel_size=2, stride=1), 
    antialiased_cnns.BlurPool(C, stride=2)]
    
# Conv --> ConvBlurPool
baseline = [nn.Conv2d(Cin, C, kernel_size=3, stride=2, padding=1), 
    nn.ReLU(inplace=True)]
antialiased = [nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    antialiased_cnns.BlurPool(C, stride=2)]

# AvgPool --> BlurPool
baseline = nn.AvgPool2d(kernel_size=2, stride=2)
antialiased = antialiased_cnns.BlurPool(C, stride=2)
```

{% include figure.html path="flc_method.png" class="img-fluid" %}
Figure 4. FLC Pooling method <d-cite key="grabinski2022frequencylowcut"></d-cite>: This approach performs downsampling directly in the frequency domain (k-space). First, the feature map undergoes a Fast Fourier Transform (FFT). Within, k-space, high-frequency components, which can cause aliasing when a stride of two is applied, are precisely removed. The remaining low-frequency components are then transformed back into the spatial domain via an Inverse Fast Fourier Transform (IFFT), ensuring an aliasing-free downsampled output.

Another intriguing approach involves downsampling in the frequency domain (k-space) called FLC Pooling <d-cite key="grabinski2022frequencylowcut"></d-cite>, which leverages explicit knowledge to guarantee aliasing-free results. The FLC Pooling method, illustrated in Figure 4, proposes to cut-out the relevant low-frequency information and discards high-frequency components that would lead to aliasing otherwise. Similar to the integration of BlurPooling, FLC Pooling can be integrated with the following code <d-cite key="grabinski2022frequencylowcutGIT"></d-cite>:

```python
from flc_pooling import FLC_Pooling

# MaxPool --> MaxBlurPool
baseline = nn.MaxPool2d(kernel_size=2, stride=2)
aliasingfree = [FLC_Pooling(),
          nn.MaxPool2d(kernel_size=2, stride=1)]
    
# Conv --> ConvBlurPool
baseline = [nn.Conv2d(Cin, C, kernel_size=3, stride=2, padding=1)]
aliasingfree = [FLC_Pooling()
        nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1)]

# AvgPool --> BlurPool
baseline = nn.AvgPool2d(kernel_size=2, stride=2)
aliasingfree = FLC_Pooling()
```


However, FLC Pooling isn't without its drawbacks; it can introduce sinc-interpolation artifacts due to the sharp, rectangular filter applied in k-space. Subsequent research <d-cite key="grabinski2023fix"></d-cite> has explored using Hamming filters to reduce these sinc interpolation artifacts, named ASAP, leading to fewer spectral distortions. Still, this can sometimes sacrifice relevant fine details due to the extensive blurring involved. Another promising downsampling variant is Detail-Preserving Pooling <d-cite key="Saeedan_2018_CVPR"></d-cite>, which preserves important structural detail, taking inspiration from the human visual system.
We present the effect of these different downsampling methods in Figure 5.

{% include figure.html path="fig3.png" class="img-fluid" %}
Figure 5. The effects of alternative downsampling methods, such as AveragePooling, BlurPooling <d-cite key="zhang2019making"></d-cite>, FLC Pooling <d-cite key="grabinski2022frequencylowcut"></d-cite>, or ASAP <d-cite key="grabinski2023fix"></d-cite>. These methods demonstrate superior information preservation and significantly reduced aliasing compared to traditional MaxPooling or convolution with stride two. However, it's crucial to note that each technique inherently compresses the image, leading to some degree of information loss. The optimal choice of downsampling method will, therefore, depend on the specific task requirements and the desired balance between aliasing reduction and feature retention. (Brain scans source <d-cite key="wakeman2015multi"></d-cite>).

Ultimately, the choice of downsampling operation requires careful consideration. It's also beneficial to think beyond the spatial domain, especially since many medical scans are originally acquired in k-space and only later transformed to the spatial domain for analysis.

## Summarizing Thoughts and Takeaways

We've explored the hidden costs of downsampling within deep learning networks for medical imaging. This journey has shown how the seemingly simple act of preparing images for our models, and even the internal workings of the models themselves, can subtly introduce distortions and lose crucial diagnostic information.

The Takeaways:
- **Internal Downsampling is Critical**: Operations like MaxPooling and strided convolutions are not neutral. They inherently deface the input or reintroduce aliasing, degrade image quality, and increase network susceptibility to noise, making our models learn from artifacts, not just authentic data.
- **Beyond Naive Approaches**: While initial data resolution variability is a starting point, the true challenge lies in how networks internally handle scale. Simple scaling strategies fall short, both in preprocessing and within the model.
- **Smarter Internal Strategies Exist**: Techniques like BlurPooling and k-space downsampling (FLC Pooling) offer promising, aliasing-mitigating alternatives to conventional internal downsampling. However, each comes with its own trade-offs, from potential over-blurring to interpolation artifacts.
- **Architectural Awareness is Key**: A deep understanding of how medical images are acquired, especially their k-space origins, can (and should) inform more intelligent and robust internal downsampling within network architectures.

Ultimately, building truly reliable and clinically relevant AI solutions for medical imaging means embracing the complexity of multi-resolution data. It requires a mindful approach to every step, from initial data handling to the internal architecture of our networks. By doing so, we move closer to unlocking the full potential of AI in medicine, ensuring our models learn from the real, nuanced data, not just artifacts of our processing.