from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import threading
import time
import json
import os
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TrainingState:
    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.current_batch = 0
        self.loss_history = []
        self.accuracy_history = []
        self.model = None
        self.stop_training = False

training_state = TrainingState()

def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                             shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_model(epochs, learning_rate):
    training_state.is_training = True
    training_state.stop_training = False
    training_state.current_epoch = 0
    training_state.loss_history = []
    training_state.accuracy_history = []
    
    socketio.emit('training_status', {'status': 'starting', 'device': str(device)})
    
    trainloader, testloader = prepare_data()
    
    training_state.model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(training_state.model.parameters(), lr=learning_rate)
    
    socketio.emit('training_status', {'status': 'training_started', 'total_epochs': epochs})
    
    start_time = time.time()
    
    for epoch in range(epochs):
        if training_state.stop_training:
            break
            
        training_state.current_epoch = epoch + 1
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            if training_state.stop_training:
                break
                
            training_state.current_batch = i + 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = training_state.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                avg_loss = running_loss / 100
                accuracy = 100 * correct / total
                
                training_state.loss_history.append(avg_loss)
                training_state.accuracy_history.append(accuracy)
                
                socketio.emit('training_update', {
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'time_elapsed': time.time() - start_time
                })
                
                running_loss = 0.0
        
        if not training_state.stop_training:
            epoch_accuracy = 100 * correct / total
            socketio.emit('epoch_complete', {
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy
            })
    
    if not training_state.stop_training:
        test_accuracy = evaluate_model(testloader)
        training_time = time.time() - start_time
        
        torch.save(training_state.model.state_dict(), 'mnist_cnn.pth')
        
        socketio.emit('training_complete', {
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'model_saved': True
        })
    else:
        socketio.emit('training_stopped', {})
    
    training_state.is_training = False

def evaluate_model(testloader):
    if training_state.model is None:
        return 0
    
    correct = 0
    total = 0
    training_state.model.eval()
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = training_state.model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    training_state.model.train()
    return 100 * correct / total

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'is_training': training_state.is_training,
        'current_epoch': training_state.current_epoch,
        'current_batch': training_state.current_batch,
        'device': str(device),
        'model_loaded': training_state.model is not None
    })

@app.route('/api/start_training', methods=['POST'])
def start_training():
    if training_state.is_training:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json
    epochs = data.get('epochs', 3)
    learning_rate = data.get('learning_rate', 0.001)
    
    thread = threading.Thread(target=train_model, args=(epochs, learning_rate))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Training started', 'epochs': epochs, 'learning_rate': learning_rate})

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    if not training_state.is_training:
        return jsonify({'error': 'No training in progress'}), 400
    
    training_state.stop_training = True
    return jsonify({'message': 'Training stop requested'})

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify({
        'loss_history': training_state.loss_history,
        'accuracy_history': training_state.accuracy_history
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if training_state.model is None:
        return jsonify({'error': 'No model loaded. Train a model first.'}), 400
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Remove data:image/png;base64, prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale and resize to 28x28
        if image.mode != 'L':
            image = image.convert('L')
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize (same as training)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        training_state.model.eval()
        with torch.no_grad():
            outputs = training_state.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Only set back to train mode if currently training
        if training_state.is_training:
            training_state.model.train()
        
        # Get all class probabilities
        class_probs = {}
        for i in range(10):
            class_probs[str(i)] = probabilities[0][i].item()
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probs
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_model', methods=['POST'])
def load_model():
    try:
        if not os.path.exists('mnist_cnn.pth'):
            return jsonify({'error': 'No saved model found. Train a model first.'}), 400
        
        training_state.model = SimpleCNN().to(device)
        training_state.model.load_state_dict(torch.load('mnist_cnn.pth'))
        training_state.model.eval()
        
        return jsonify({'message': 'Model loaded successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    emit('connected', {'data': 'Connected to training server'})

if __name__ == '__main__':
    print(f"Server starting on device: {device}")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)