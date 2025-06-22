# ML Object Tracker with Velocity Prediction

A real-time object tracking system that uses machine learning to predict future trajectories and movement patterns of detected objects.

## Overview

This system combines computer vision with deep learning to:
- Detect objects in real-time using YOLOv8
- Track multiple objects simultaneously 
- Predict future movement trajectories using LSTM neural networks
- Analyze movement patterns and classify maneuver types
- Calculate optimal intercept trajectories

## Features

### Core Functionality
- **Real-time Object Detection**: Uses YOLOv8 for fast and accurate object detection
- **Multi-object Tracking**: Tracks position, velocity, and acceleration of multiple objects
- **ML-based Trajectory Prediction**: LSTM neural network predicts future movement paths
- **Movement Pattern Classification**: Identifies different types of maneuvers (straight, weaving, spiral, etc.)
- **Velocity Analysis**: Calculates speed, acceleration, and directional changes
- **Live Visualization**: Real-time display of tracks, predictions, and statistics

### Advanced Features
- **Attention Mechanism**: Focuses on critical movement patterns for better predictions
- **Online Learning**: Continuously improves predictions based on observed trajectories
- **Physics-based Fallback**: Uses kinematic models when ML predictions are uncertain
- **Intercept Calculations**: Computes optimal interception points and trajectories
- **Confidence Scoring**: Provides reliability metrics for all predictions

## Installation

### Requirements
```bash
pip install ultralytics opencv-python torch torchvision scikit-learn numpy
```

### Files
- `object-tracker-velocity.py` - Main system implementation
- `yolov8n.pt` - YOLOv8 model weights
- `updated_model.pth` - Trained LSTM model for trajectory prediction

## Usage

### Basic Operation
```bash
python object-tracker-velocity.py
```

### Controls
- **Q** - Quit the application
- **S** - Save the current ML model
- **R** - Reset all active tracks

### Configuration
The system can be configured for different input sources:
- Webcam (default): `video_source=0`
- Video file: `video_source="path/to/video.mp4"`
- IP camera: `video_source="rtsp://camera_url"`

## System Architecture

### Core Components

1. **MLEnhancedInterceptSystem**
   - Main system orchestrator
   - Handles video input and processing pipeline
   - Coordinates detection, tracking, and prediction

2. **MissileManeuverPredictor**
   - LSTM-based neural network
   - Multi-head attention mechanism
   - Predicts future positions and movement types

3. **Object Detection**
   - YOLOv8 implementation
   - Real-time object detection and classification
   - Bounding box extraction and confidence scoring

4. **Tracking System**
   - Position history maintenance
   - Velocity and acceleration calculation
   - Feature extraction for ML prediction

### Data Flow
1. Camera input captures frames
2. YOLOv8 detects objects and generates bounding boxes
3. Tracking system maintains object histories
4. Feature extraction prepares data for ML prediction
5. LSTM network predicts future trajectories
6. Visualization renders results on screen
7. Model continuously learns from observed patterns

## Technical Details

### Machine Learning Model
- **Architecture**: LSTM with attention mechanism
- **Input Features**: Position (x,y,z), velocity (vx,vy,vz), acceleration (ax,ay,az)
- **Output**: Future trajectory points and maneuver classification
- **Training**: Online learning with experience replay

### Movement Classification
The system recognizes seven types of movement patterns:
- Straight line motion
- Weaving patterns
- Spiral movements
- Barrel roll maneuvers
- Split-S inversions
- Chandelle climbs
- Random jinking (evasive)

### Performance Metrics
- **Detection Speed**: ~40ms per frame (25 FPS)
- **Tracking Accuracy**: Real-time multi-object capability
- **Prediction Horizon**: Up to 30 future time steps
- **Memory Usage**: Optimized for real-time operation

## Applications

### Defense and Security
- Missile trajectory prediction
- Drone tracking and interception
- Surveillance system enhancement

### Transportation
- Vehicle behavior analysis
- Traffic pattern prediction
- Autonomous vehicle path planning

### Sports Analytics
- Player movement analysis
- Game strategy optimization
- Performance tracking

### Research
- Motion analysis studies
- Behavioral pattern recognition
- Predictive modeling research

## File Structure

```
tracker-speed/
├── object-tracker-velocity.py    # Main implementation
├── yolov8n.pt                   # YOLOv8 model weights
├── updated_model.pth            # Trained LSTM model
└── README.md                    # This documentation
```

## Algorithm Details

### Feature Extraction
For each tracked object, the system calculates:
- 3D position coordinates
- Velocity vectors in all dimensions
- Acceleration components
- Temporal relationships between measurements

### Prediction Pipeline
1. **Data Preprocessing**: Normalize input features
2. **Sequence Processing**: LSTM processes temporal sequences
3. **Attention Weighting**: Focus on relevant movement patterns
4. **Trajectory Generation**: Predict future position sequences
5. **Confidence Estimation**: Calculate prediction reliability

### Learning System
- **Buffer Management**: Maintains history of completed trajectories
- **Batch Training**: Periodic model updates with accumulated data
- **Validation**: Continuous performance monitoring
- **Adaptation**: Real-time parameter adjustment

## Performance Optimization

### Real-time Processing
- Efficient tensor operations using PyTorch
- Optimized OpenCV operations for video processing
- Memory management for continuous operation
- GPU acceleration when available

### Scalability
- Configurable prediction horizon
- Adjustable tracking parameters
- Modular architecture for easy extension
- Multi-threading support for parallel processing

## Troubleshooting

### Common Issues
- **Camera not detected**: Check camera permissions and connections
- **Slow performance**: Reduce prediction horizon or lower video resolution
- **Model errors**: Ensure all dependencies are properly installed
- **Memory issues**: Adjust buffer sizes in configuration

### Debug Mode
Enable detailed logging by modifying the system initialization:
```python
system = MLEnhancedInterceptSystem(
    video_source=0,
    model_path="updated_model.pth",
    debug=True
)
```

## Future Enhancements

### Planned Features
- 3D trajectory visualization
- Advanced sensor fusion
- Real-time model training interface
- Distributed processing support
- Enhanced prediction algorithms

### Research Directions
- Integration with other sensing modalities
- Advanced attention mechanisms
- Reinforcement learning for optimal tracking
- Federated learning for model sharing

## License

This project is developed for research and educational purposes. Please ensure compliance with local regulations when using for surveillance or tracking applications.

## Contributors

This system represents advanced work in computer vision, machine learning, and real-time processing, combining multiple state-of-the-art technologies for practical trajectory prediction applications. 