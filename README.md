# Traffic Sign Detection System

A web-based application for detecting and classifying traffic signs using deep learning models. The system uses a combination of segmentation and classification models to accurately identify traffic signs in images.

## Features

- Traffic sign detection and classification
- Image segmentation for improved accuracy
- Top prediction visualization
- User feedback collection
- Web-based interface for easy interaction

## Project Structure

```
.
├── app.py                  # Main Flask application
├── services/              # Service modules
│   ├── top_predict.py     # Top prediction visualization
│   └── real_time_detections.py  # Real-time detection functionality
├── templates/             # HTML templates
│   ├── index.html         # Main application interface
│   └── thank_you.html     # Feedback confirmation page
├── traffic_model.h5       # Trained traffic sign classification model
├── segmentation_model.h5  # Trained segmentation model
└── feedback_log.csv       # User feedback storage
```

## Requirements

- Python 3.x
- Flask
- TensorFlow
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd traffic
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload an image containing traffic signs

4. View the detection results, including:
   - Original image
   - Processed image
   - Segmented image
   - Top predictions with confidence scores
   - Chart of the top 3

5. Provide feedback on the detection results

## Models

The system uses two pre-trained models:

1. **Segmentation Model**: Identifies and isolates traffic signs in the image
2. **Classification Model**: Classifies the detected traffic signs into one of 43 categories

## Feedback System

The application includes a feedback mechanism where users can provide input on the accuracy of detections. Feedback is stored in `feedback_log.csv` for future model improvements.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 