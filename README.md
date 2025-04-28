# MoonArc: Real-Time Lunar Phase Detection
## Overview
MoonArc is a deep learning-based application designed to detect and classify lunar phases in real-time from uploaded or captured images. The project leverages a pre-trained deep learning model to identify moon phases and provides a user-friendly interface for both educational and practical purposes. Additionally, it includes a lunar calendar feature to offer historical and predictive insights about lunar phases.

## Features
- **Real-Time Lunar Phase Detection: Upload an image or capture one using your camera to detect the current lunar phase.**

- **Deep Learning Model: Utilizes a pre-trained TensorFlow model for accurate classification.**

- **User-Friendly Interface: Built with Streamlit for an intuitive and interactive experience.**

- **Lunar Calendar Integration: Provides historical and predictive lunar phase data.**

- **Multiple Use Cases: Useful for astronomy education, cultural practices, tide predictions, and biological studies.**

## Installation
To run MoonArc locally, follow these steps:

**1. Clone the Repository:**

```bash
git clone https://github.com/MuskanKaushal11/MoonArc.git
cd MoonArc 
```

**2. Set Up a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the Application:**

```bash
streamlit run app.py
```
**5. Access the Application:**

Open your web browser and navigate to http://localhost:8501.


## Usage
**1. Upload an Image:**

- Click on "Upload Image" and select an image of the moon from your device.

- The application will process the image and display the detected lunar phase.

**2. Open Camera:**

- Click on "Open Camera" to capture a real-time image using your device's camera.

- The application will process the captured image and display the detected lunar phase.

## Project Structure
```
MoonArc/
├── MoonArc.py                  # Main Streamlit application script
├── MoonArcModel.keras           # Pre-trained deep learning model
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
└── images/                 # Directory for sample images(optional)
```
## Dependencies
- Streamlit
- TensorFlow
- NumPy
- OpenCV
- Pillow
<!-- 
## Contributing
We welcome contributions to MoonArc! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

**1. Fork the Repository:**

```bash
git clone https://github.com/MuskanKaushal11/MoonArc.git
cd MoonArc
```

**2. Create a New Branch:**

```bash
git checkout -b feature/YourFeatureName
```

**3. Commit Your Changes:**

```bash
git commit -m "Add some feature"
```

**4. Push to the Branch:**

```bash
git push origin feature/YourFeatureName
```

**5. Open a Pull Request: Go to the original repository and open a pull request with your changes.** -->

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Thanks to the TensorFlow and Streamlit communities for their excellent tools and documentation.

- Special thanks to all contributors and users of MoonArc.

## Contact
For any inquiries or feedback, please contact [Your Name] at [your.email@example.com].

Enjoy exploring the moon phases with MoonArc! 🌕🌖🌗🌘🌑🌒🌓🌔
