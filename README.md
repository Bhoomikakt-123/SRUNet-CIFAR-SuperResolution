# SRUNet-CIFAR: Super-Resolution Using CNN

SRUNet-CIFAR is a deep learning-based super-resolution model built using PyTorch. It upscales low-resolution CIFAR-10 images (32×32) to high-resolution (128×128) using a custom encoder-decoder CNN architecture called SRUNet.

## 🧠 Key Features
- Upscales images from 32×32 to 128×128.
- Trained on the CIFAR-10 dataset.
- Evaluated using PSNR and SSIM metrics.
- Visual comparison between input, ground truth, and predicted output.

  
## ⚙️ Setup
### 1.. Clone the Repository

    ```bash
    git clone https://github.com/yourusername/SRUNet-CIFAR-SuperResolution.git
    cd SRUNet-CIFAR-SuperResolution

### 2.. Set Up a Virtual Environment (Recommended)
     ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use: venv\\Scripts\\activate

### 3. Install Dependencies
     ```bash
    pip install -r requirements.txt

🚀 Run the Script
     ```bash
    python src/srunet_cifar.py

🖼️ Sample Output
The model will be trained for 5 epochs and the result will be saved as:
    models/srunet_model.pth – trained model
    outputs/sample_output.png – output comparison image

📊 Metrics
    PSNR (Peak Signal-to-Noise Ratio): Measures reconstruction quality.
    SSIM (Structural Similarity Index): Measures similarity between generated and real images.

🧠 Applications
This approach is useful in:
    Satellite imaging
    Video surveillance enhancement
    Medical imaging
    Photo restoration

🤖 Model Architecture
The architecture includes:
    Encoder: 3 convolutional layers with ReLU activation
    Decoder: 3 transposed convolutional layers to upscale images
