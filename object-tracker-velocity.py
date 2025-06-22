import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import os
from scipy.signal import savgol_filter

class KalmanFilter:
    """
    Filtro di Kalman per tracking preciso di posizione e velocità
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(3) * 1.0
        
        # Error covariance
        self.P = np.eye(6) * 100.0
        
        self.initialized = False
    
    def predict(self):
        """Predict step"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """Update step with new measurement"""
        if not self.initialized:
            self.state[:3] = measurement
            self.initialized = True
            return
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update
        y = measurement - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def get_position(self):
        return self.state[:3]
    
    def get_velocity(self):
        return self.state[3:6]

@dataclass
class MissileTrajectory:
    """Dati completi traiettoria missile"""
    positions: np.ndarray  # [N, 3] posizioni
    velocities: np.ndarray  # [N, 3] velocità
    accelerations: np.ndarray  # [N, 3] accelerazioni
    timestamps: np.ndarray  # [N] tempi
    missile_type: str  # tipo di missile
    maneuver_pattern: str  # pattern di manovra

class MissileManeuverPredictor(nn.Module):
    """
    Rete neurale LSTM avanzata per predire manovre future dei missili.
    Input: sequenza di stati passati (pos, vel, acc)
    Output: traiettoria futura predetta con alta precisione
    """
    
    def __init__(self, input_size=15, hidden_size=256, num_layers=4, 
                 output_size=9, prediction_horizon=30):
        super(MissileManeuverPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # Feature embedding per catturare relazioni complesse
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1)
        )
        
        # LSTM principale con più capacità
        self.lstm = nn.LSTM(64, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Self-attention avanzato
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Decoder più profondo per predizione multi-step
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size * prediction_horizon)
        )
        
        # Classificatore tipo manovra migliorato
        self.maneuver_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 7 tipi di manovra
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, prediction_horizon)
        )
        
    def forward(self, x, return_attention=False):
        # x shape: [batch, sequence_length, features]
        batch_size = x.size(0)
        
        # Feature embedding
        embedded = self.feature_embedding(x)
        
        # LSTM encoding (bidirectional)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Self-attention su sequenza
        attn_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Residual connection + layer norm
        combined = self.layer_norm(lstm_out + attn_out)
        
        # Usa ultimo hidden state per predizione
        last_hidden = combined[:, -1, :]
        
        # Predici traiettoria futura
        future_trajectory = self.decoder(last_hidden)
        future_trajectory = future_trajectory.view(
            batch_size, self.prediction_horizon, -1
        )
        
        # Classifica tipo di manovra
        maneuver_type = self.maneuver_classifier(last_hidden)
        
        # Uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_head(last_hidden))
        
        if return_attention:
            return future_trajectory, maneuver_type, uncertainty, attention_weights
        return future_trajectory, maneuver_type, uncertainty

class MLInterceptSystem:
    """
    Sistema di intercettazione con ML per predizione manovre
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Modello ML
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = MissileManeuverPredictor().to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                self.predictor.load_state_dict(torch.load(model_path))
                print(f"Modello caricato da {model_path}")
            except RuntimeError as e:
                print(f"Modello incompatibile, inizializzo nuovo modello: {e}")
                print("Il modello migliorato ha un'architettura diversa")
        
        self.predictor.eval()
        
        # Scaler per normalizzazione dati
        self.scaler = StandardScaler()
        
        # Buffer per training online
        self.trajectory_buffer = deque(maxlen=1000)
        self.online_training = True
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=0.001)
        
        # Tipi di manovre riconosciute
        self.maneuver_types = {
            0: "straight",      # Traiettoria rettilinea
            1: "weaving",       # Serpentina
            2: "spiral",        # Spirale
            3: "barrel_roll",   # Tonneau
            4: "split_s",       # Split-S (inversione)
            5: "chandelle",     # Chandelle (salita con virata)
            6: "jinking"        # Manovre casuali anti-intercettazione
        }
        
        # Parametri sistema
        self.sequence_length = 20  # Frame di storia per predizione
        self.prediction_confidence_threshold = 0.7
        
    def extract_features(self, track: Dict) -> np.ndarray:
        """
        Estrae features avanzate e precise per ML da track
        """
        positions = np.array(track['positions'])
        timestamps = np.array(track['timestamps'])
        
        if len(positions) < 3:
            return None
        
        # Smooth delle posizioni con filtro Savitzky-Golay se abbastanza punti
        if len(positions) >= 5:
            smoothed_positions = np.zeros_like(positions)
            for dim in range(3):
                try:
                    smoothed_positions[:, dim] = savgol_filter(positions[:, dim], 
                                                             window_length=min(5, len(positions)), 
                                                             polyorder=2)
                except:
                    smoothed_positions[:, dim] = positions[:, dim]
        else:
            smoothed_positions = positions
        
        # Calcola velocità con derivata centrale per maggiore precisione
        velocities = np.zeros((len(smoothed_positions), 3))
        for i in range(len(smoothed_positions)):
            if i == 0:
                # Forward difference
                dt = timestamps[1] - timestamps[0]
                velocities[i] = (smoothed_positions[1] - smoothed_positions[0]) / dt
            elif i == len(smoothed_positions) - 1:
                # Backward difference
                dt = timestamps[i] - timestamps[i-1]
                velocities[i] = (smoothed_positions[i] - smoothed_positions[i-1]) / dt
            else:
                # Central difference (più precisa)
                dt = timestamps[i+1] - timestamps[i-1]
                velocities[i] = (smoothed_positions[i+1] - smoothed_positions[i-1]) / dt
        
        # Calcola accelerazioni con derivata centrale
        accelerations = np.zeros((len(velocities), 3))
        for i in range(len(velocities)):
            if i == 0:
                dt = timestamps[1] - timestamps[0]
                accelerations[i] = (velocities[1] - velocities[0]) / dt
            elif i == len(velocities) - 1:
                dt = timestamps[i] - timestamps[i-1]
                accelerations[i] = (velocities[i] - velocities[i-1]) / dt
            else:
                dt = timestamps[i+1] - timestamps[i-1]
                accelerations[i] = (velocities[i+1] - velocities[i-1]) / dt
        
        # Features avanzate
        features = []
        
        for i in range(len(smoothed_positions)):
            # Features base
            pos = smoothed_positions[i]
            vel = velocities[i]
            acc = accelerations[i]
            
            # Metriche derivate
            speed = np.linalg.norm(vel)
            acc_magnitude = np.linalg.norm(acc)
            
            # Angoli di direzione
            if speed > 0.1:
                heading = np.arctan2(vel[1], vel[0])
                pitch = np.arcsin(np.clip(vel[2] / speed, -1, 1))
            else:
                heading = pitch = 0
            
            # Curvatura (rate di cambio di heading)
            curvature = 0
            if i > 0 and speed > 0.1:
                prev_speed = np.linalg.norm(velocities[i-1])
                if prev_speed > 0.1:
                    prev_heading = np.arctan2(velocities[i-1][1], velocities[i-1][0])
                    dt = timestamps[i] - timestamps[i-1]
                    if dt > 0:
                        curvature = (heading - prev_heading) / dt
            
            # Jerk (derivata dell'accelerazione)
            jerk_magnitude = 0
            if i > 0:
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:
                    jerk = (accelerations[i] - accelerations[i-1]) / dt
                    jerk_magnitude = np.linalg.norm(jerk)
            
            # Normalizzazione dell'altitudine per stabilità numerica
            normalized_alt = pos[2] / 100.0  # Scala metri
            
            # Combina tutte le features (15 dimensioni)
            enhanced_features = np.concatenate([
                pos,  # 3D position
                vel,  # 3D velocity
                acc,  # 3D acceleration
                [speed, acc_magnitude, heading, pitch, curvature, jerk_magnitude]  # 6 features derivate
            ])
            
            features.append(enhanced_features)
        
        return np.array(features) if features else None
    
    def predict_trajectory(self, track: Dict, horizon_seconds: float = 5.0) -> Dict:
        """
        Predice traiettoria futura usando ML
        """
        features = self.extract_features(track)
        
        if features is None or len(features) < self.sequence_length:
            # Fallback a predizione fisica se non abbastanza dati
            return self.physics_based_prediction(track, horizon_seconds)
        
        # Prepara input per rete neurale
        sequence = features[-self.sequence_length:]
        
        # Normalizza
        if hasattr(self.scaler, 'mean_'):
            sequence_normalized = self.scaler.transform(
                sequence.reshape(-1, sequence.shape[-1])
            ).reshape(sequence.shape)
        else:
            sequence_normalized = sequence
            
        # Converti in tensore
        input_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
        
        # Predizione
        with torch.no_grad():
            future_trajectory, maneuver_logits, uncertainty = self.predictor(input_tensor)
            
        # Converti output
        future_trajectory = future_trajectory.squeeze(0).cpu().numpy()
        maneuver_probs = torch.softmax(maneuver_logits, dim=1).squeeze(0).cpu().numpy()
        uncertainty = uncertainty.squeeze(0).cpu().numpy()
        
        # Identifica manovra più probabile
        maneuver_id = np.argmax(maneuver_probs)
        maneuver_confidence = maneuver_probs[maneuver_id]
        maneuver_type = self.maneuver_types[maneuver_id]
        
        # Ricostruisci posizioni future
        dt = 0.1  # 10 Hz
        future_positions = []
        current_pos = track['positions'][-1].copy()
        
        for i in range(len(future_trajectory)):
            # Estrai predizione [pos, vel, acc]
            pred = future_trajectory[i]
            if len(pred) >= 3:
                # Aggiorna posizione basata su predizione
                current_pos = pred[:3]
                future_positions.append(current_pos.copy())
        
        return {
            'predicted_positions': np.array(future_positions),
            'maneuver_type': maneuver_type,
            'maneuver_confidence': maneuver_confidence,
            'confidence_per_step': self.calculate_confidence_decay(
                maneuver_confidence, len(future_positions)
            ),
            'alternative_maneuvers': {
                self.maneuver_types[i]: prob 
                for i, prob in enumerate(maneuver_probs)
                if prob > 0.1
            }
        }
    
    def calculate_confidence_decay(self, initial_confidence: float, 
                                 steps: int) -> np.ndarray:
        """
        Calcola decay della confidenza su predizione
        """
        # Esponenziale decay
        decay_rate = 0.95
        confidences = initial_confidence * np.power(decay_rate, np.arange(steps))
        return confidences
    
    def physics_based_prediction(self, track: Dict, horizon_seconds: float) -> Dict:
        """
        Fallback: predizione basata su fisica quando ML non disponibile
        """
        positions = np.array(track['positions'])
        timestamps = np.array(track['timestamps'])
        
        if len(positions) < 2:
            return {
                'predicted_positions': np.array([positions[-1]]),
                'maneuver_type': 'unknown',
                'maneuver_confidence': 0.0
            }
        
        # Calcola velocità e accelerazione medie
        recent_positions = positions[-5:]
        recent_times = timestamps[-5:]
        
        if len(recent_positions) >= 2:
            velocities = []
            for i in range(1, len(recent_positions)):
                dt = recent_times[i] - recent_times[i-1]
                if dt > 0:
                    vel = (recent_positions[i] - recent_positions[i-1]) / dt
                    velocities.append(vel)
            
            avg_velocity = np.mean(velocities, axis=0)
            
            # Predici con velocità costante + gravità
            dt = 0.1
            steps = int(horizon_seconds / dt)
            future_positions = []
            current_pos = positions[-1].copy()
            current_vel = avg_velocity.copy()
            
            for _ in range(steps):
                current_pos = current_pos + current_vel * dt
                current_vel[2] -= 9.81 * dt  # Gravità
                future_positions.append(current_pos.copy())
            
            return {
                'predicted_positions': np.array(future_positions),
                'maneuver_type': 'ballistic',
                'maneuver_confidence': 0.5
            }
        
        return {
            'predicted_positions': np.array([positions[-1]]),
            'maneuver_type': 'unknown',
            'maneuver_confidence': 0.0
        }
    
    def train_online(self, completed_trajectory: MissileTrajectory):
        """
        Training online con nuove traiettorie osservate
        """
        if not self.online_training:
            return
            
        # Aggiungi a buffer
        self.trajectory_buffer.append(completed_trajectory)
        
        # Train ogni 10 traiettorie
        if len(self.trajectory_buffer) % 10 == 0:
            self._train_batch()
    
    def _train_batch(self):
        """
        Esegue training su batch di traiettorie
        """
        if len(self.trajectory_buffer) < 10:
            return
            
        self.predictor.train()
        
        # Prepara batch
        batch_inputs = []
        batch_targets = []
        
        for trajectory in list(self.trajectory_buffer)[-50:]:  # Ultimi 50
            # Crea sequenze di training
            positions = trajectory.positions
            velocities = trajectory.velocities
            accelerations = trajectory.accelerations
            
            for i in range(self.sequence_length, 
                         len(positions) - self.predictor.prediction_horizon):
                # Input: storia
                input_seq = np.concatenate([
                    positions[i-self.sequence_length:i],
                    velocities[i-self.sequence_length:i],
                    accelerations[i-self.sequence_length:i]
                ], axis=1)
                
                # Target: futuro
                target_seq = np.concatenate([
                    positions[i:i+self.predictor.prediction_horizon],
                    velocities[i:i+self.predictor.prediction_horizon],
                    accelerations[i:i+self.predictor.prediction_horizon]
                ], axis=1)
                
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
        
        if not batch_inputs:
            return
            
        # Converti in tensori
        inputs = torch.FloatTensor(batch_inputs).to(self.device)
        targets = torch.FloatTensor(batch_targets).to(self.device)
        
        # Training step
        self.optimizer.zero_grad()
        predictions, _ = self.predictor(inputs)
        
        loss = nn.MSELoss()(predictions, targets)
        loss.backward()
        self.optimizer.step()
        
        self.predictor.eval()
        print(f"Training loss: {loss.item():.4f}")
    
    def calculate_optimal_intercept(self, missile_pos: np.ndarray, 
                                  missile_vel: float,
                                  target_track: Dict,
                                  ml_prediction: Dict) -> Dict:
        """
        Calcola intercettazione ottimale usando predizione ML
        """
        predicted_positions = ml_prediction['predicted_positions']
        confidence_per_step = ml_prediction['confidence_per_step']
        maneuver_type = ml_prediction['maneuver_type']
        
        best_solution = None
        min_time = float('inf')
        
        # Valuta ogni punto della traiettoria predetta
        for i, (target_pos, confidence) in enumerate(
            zip(predicted_positions, confidence_per_step)):
            
            if confidence < self.prediction_confidence_threshold:
                break  # Non fidarsi di predizioni con bassa confidenza
            
            # Tempo per raggiungere questo punto
            distance = np.linalg.norm(target_pos - missile_pos)
            intercept_time = distance / missile_vel
            
            # Verifica fattibilità temporale
            if intercept_time <= i * 0.1:  # Missile arriverebbe prima del target
                continue
                
            # Calcola probabilità successo
            hit_probability = self.calculate_hit_probability(
                missile_pos, missile_vel, target_pos, 
                confidence, maneuver_type, intercept_time
            )
            
            # Aggiorna se migliore
            if hit_probability > 0.5 and intercept_time < min_time:
                min_time = intercept_time
                best_solution = {
                    'intercept_point': target_pos,
                    'intercept_time': intercept_time,
                    'hit_probability': hit_probability,
                    'target_maneuver': maneuver_type,
                    'confidence': confidence
                }
        
        # Se nessuna soluzione ML valida, usa metodo tradizionale
        if best_solution is None:
            return self.proportional_navigation_intercept(
                missile_pos, missile_vel, target_track
            )
        
        return best_solution
    
    def calculate_hit_probability(self, missile_pos: np.ndarray,
                                missile_vel: float,
                                target_pos: np.ndarray,
                                prediction_confidence: float,
                                maneuver_type: str,
                                intercept_time: float) -> float:
        """
        Calcola probabilità di successo considerando tipo di manovra
        """
        # Base probability
        base_prob = prediction_confidence
        
        # Fattori di difficoltà per tipo di manovra
        maneuver_difficulty = {
            'straight': 0.9,
            'weaving': 0.7,
            'spiral': 0.6,
            'barrel_roll': 0.5,
            'split_s': 0.6,
            'chandelle': 0.7,
            'jinking': 0.3,
            'ballistic': 0.8,
            'unknown': 0.4
        }
        
        maneuver_factor = maneuver_difficulty.get(maneuver_type, 0.5)
        
        # Fattore distanza
        distance = np.linalg.norm(target_pos - missile_pos)
        distance_factor = np.exp(-distance / 5000)
        
        # Fattore tempo
        time_factor = np.exp(-intercept_time / 20)
        
        # Probabilità finale
        hit_prob = base_prob * maneuver_factor * distance_factor * time_factor
        
        return np.clip(hit_prob, 0, 1)
    
    def proportional_navigation_intercept(self, missile_pos: np.ndarray,
                                        missile_vel: float,
                                        target_track: Dict) -> Dict:
        """
        Fallback: Proportional Navigation classica
        """
        # Implementazione PN standard
        target_pos = target_track['positions'][-1]
        target_vel = target_track['velocities'][-1] if 'velocities' in target_track else np.zeros(3)
        
        # Vettore line-of-sight
        los = target_pos - missile_pos
        los_range = np.linalg.norm(los)
        
        if los_range == 0:
            return None
            
        # Calcola tempo di intercettazione approssimato
        closing_speed = missile_vel
        intercept_time = los_range / closing_speed
        
        # Punto di intercettazione stimato
        intercept_point = target_pos + target_vel * intercept_time
        
        return {
            'intercept_point': intercept_point,
            'intercept_time': intercept_time,
            'hit_probability': 0.6,  # PN standard
            'target_maneuver': 'unknown',
            'confidence': 0.6
        }
    
    def generate_evasion_recommendation(self, incoming_missiles: List[Dict]) -> Dict:
        """
        Genera raccomandazioni di evasione basate su pattern missili in arrivo
        """
        if not incoming_missiles:
            return {'maneuver': 'none', 'confidence': 1.0}
        
        # Analizza minacce
        threats = []
        for missile in incoming_missiles:
            if 'ml_prediction' in missile:
                impact_time = missile['ml_prediction'].get('intercept_time', float('inf'))
                threat_level = 1.0 / (impact_time + 1)
                threats.append({
                    'missile': missile,
                    'impact_time': impact_time,
                    'threat_level': threat_level
                })
        
        # Ordina per livello di minaccia
        threats.sort(key=lambda x: x['threat_level'], reverse=True)
        
        # Determina manovra evasiva ottimale
        if len(threats) == 1:
            # Singola minaccia: manovra perpendicolare
            return {
                'maneuver': 'barrel_roll',
                'direction': self._calculate_evasion_direction(threats[0]['missile']),
                'confidence': 0.8
            }
        else:
            # Multiple minacce: manovra complessa
            return {
                'maneuver': 'jinking',
                'pattern': self._generate_jinking_pattern(threats),
                'confidence': 0.7
            }
    
    def _calculate_evasion_direction(self, missile: Dict) -> np.ndarray:
        """Calcola direzione ottimale di evasione"""
        # Perpendicolare alla direzione di arrivo
        approach_vector = missile['velocity'] / np.linalg.norm(missile['velocity'])
        
        # Trova vettore perpendicolare
        if abs(approach_vector[2]) < 0.9:
            perp = np.cross(approach_vector, np.array([0, 0, 1]))
        else:
            perp = np.cross(approach_vector, np.array([1, 0, 0]))
            
        return perp / np.linalg.norm(perp)
    
    def _generate_jinking_pattern(self, threats: List[Dict]) -> List[Dict]:
        """Genera pattern di manovre evasive casuali"""
        pattern = []
        
        for i in range(5):
            # Alterna direzioni per confondere predittori
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            pattern.append({
                'time': i * 0.5,
                'direction': direction,
                'acceleration': 50 + np.random.rand() * 50
            })
        
        return pattern

# Integrazione con sistema principale
class MLEnhancedInterceptSystem(MLInterceptSystem):
    """
    Sistema completo con tracking video e ML
    """
    
    def __init__(self, video_source=None, model_path=None):
        super().__init__(model_path)
        
        # Setup video
        self.cap = cv2.VideoCapture(video_source or 0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # YOLO per detection
        from ultralytics import YOLO
        self.detector = YOLO('yolov8n.pt')
        
        # Tracking
        self.tracks = {}
        self.kalman_filters = {}  # Filtri di Kalman per ogni track
        self.missiles = {}
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Processa frame con ML predictions
        """
        # Detection
        results = self.detector(frame, conf=0.3)
        
        # Update tracks
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Processa detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Estrai informazioni bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Centro dell'oggetto
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Crea/aggiorna track con tracking più intelligente
                    best_track_id = None
                    min_distance = float('inf')
                    
                    # Cerca track esistente più vicino
                    for existing_id, existing_track in self.tracks.items():
                        if existing_track['class'] == cls and len(existing_track['positions']) > 0:
                            last_pos = existing_track['positions'][-1]
                            distance = np.linalg.norm([center_x - last_pos[0], center_y - last_pos[1]])
                            if distance < min_distance and distance < 100:  # Soglia di distanza
                                min_distance = distance
                                best_track_id = existing_id
                    
                    if best_track_id is None:
                        # Crea nuovo track
                        track_id = f"track_{cls}_{int(current_time*1000)}"
                        self.tracks[track_id] = {
                            'positions': [],
                            'timestamps': [],
                            'class': cls,
                            'confidence': [],
                            'kalman_positions': [],
                            'kalman_velocities': []
                        }
                        # Inizializza filtro di Kalman
                        self.kalman_filters[track_id] = KalmanFilter()
                    else:
                        track_id = best_track_id
                    
                    # Posizione misurata
                    measured_pos = np.array([center_x, center_y, 0])
                    
                    # Aggiorna filtro di Kalman
                    if track_id in self.kalman_filters:
                        kf = self.kalman_filters[track_id]
                        kf.predict()
                        kf.update(measured_pos)
                        
                        # Usa posizione filtrata
                        filtered_pos = kf.get_position()
                        filtered_vel = kf.get_velocity()
                        
                        self.tracks[track_id]['kalman_positions'].append(filtered_pos)
                        self.tracks[track_id]['kalman_velocities'].append(filtered_vel)
                    else:
                        filtered_pos = measured_pos
                    
                    # Aggiungi posizione
                    self.tracks[track_id]['positions'].append(filtered_pos)
                    self.tracks[track_id]['timestamps'].append(current_time)
                    self.tracks[track_id]['confidence'].append(conf)
                    
                    # Mantieni solo ultimi N punti
                    if len(self.tracks[track_id]['positions']) > 50:
                        self.tracks[track_id]['positions'] = self.tracks[track_id]['positions'][-50:]
                        self.tracks[track_id]['timestamps'] = self.tracks[track_id]['timestamps'][-50:]
                        self.tracks[track_id]['confidence'] = self.tracks[track_id]['confidence'][-50:]
                    
                    # Disegna detection
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id[:10]}... ({conf:.2f})", 
                              (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Rimuovi track scaduti (non aggiornati da più di 2 secondi)
        expired_tracks = []
        for track_id, track in self.tracks.items():
            if len(track['timestamps']) > 0:
                time_since_update = current_time - track['timestamps'][-1]
                if time_since_update > 2.0:
                    expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.tracks[track_id]
            if track_id in self.kalman_filters:
                del self.kalman_filters[track_id]
        
        # ML predictions per ogni track
        ml_predictions = {}
        
        for track_id, track in self.tracks.items():
            if len(track['positions']) >= min(5, self.sequence_length):  # Ridotto per test
                try:
                    ml_pred = self.predict_trajectory(track)
                    ml_predictions[track_id] = ml_pred
                    
                    # Visualizza predizione
                    self._draw_ml_prediction(frame, track, ml_pred)
                    
                    # Visualizza velocità misurata vs filtrata
                    if 'kalman_velocities' in track and len(track['kalman_velocities']) > 0:
                        velocity = track['kalman_velocities'][-1]
                        speed = np.linalg.norm(velocity)
                        last_pos = track['positions'][-1]
                        velocity_text = f"V: {speed:.1f} m/s"
                        cv2.putText(frame, velocity_text, 
                                  (int(last_pos[0]) + 10, int(last_pos[1]) + 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
                except Exception as e:
                    print(f"Errore predizione per {track_id}: {e}")
        
        # Statistiche avanzate su frame
        cv2.putText(frame, f"Active Tracks: {len(self.tracks)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"ML Predictions: {len(ml_predictions)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Kalman Filters: {len(self.kalman_filters)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, ml_predictions
    
    def _draw_ml_prediction(self, frame: np.ndarray, track: Dict, prediction: Dict):
        """Visualizza predizione ML su frame"""
        if 'predicted_positions' not in prediction:
            return
            
        predicted_positions = prediction['predicted_positions']
        confidence = prediction.get('maneuver_confidence', 0.5)
        maneuver_type = prediction.get('maneuver_type', 'unknown')
        
        # Colore basato su confidenza
        color = (0, int(255 * confidence), int(255 * (1 - confidence)))
        
        # Disegna traiettoria predetta (semplificato per 2D)
        for i in range(len(predicted_positions) - 1):
            if i < len(predicted_positions) - 1:
                pt1 = (int(predicted_positions[i][0]), int(predicted_positions[i][1]))
                pt2 = (int(predicted_positions[i + 1][0]), int(predicted_positions[i + 1][1]))
                cv2.line(frame, pt1, pt2, color, 2)
        
        # Disegna traiettoria passata
        positions = track['positions']
        if len(positions) > 1:
            for i in range(len(positions) - 1):
                pt1 = (int(positions[i][0]), int(positions[i][1]))
                pt2 = (int(positions[i + 1][0]), int(positions[i + 1][1]))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)  # Blu per passato
        
        # Annota tipo di manovra
        if len(track['positions']) > 0:
            last_pos = track['positions'][-1]
            text = f"{maneuver_type} ({confidence:.0%})"
            cv2.putText(frame, text, 
                      (int(last_pos[0]) + 10, int(last_pos[1]) - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _world_to_pixel(self, world_pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """Converte coordinate mondo in pixel"""
        # Implementa trasformazione inversa
        # ... (dipende dalla calibrazione camera)
        return None  # Placeholder
    
    def save_model(self, path: str):
        """Salva modello ML addestrato"""
        torch.save(self.predictor.state_dict(), path)
        print(f"Modello salvato in {path}")

# Esempio di utilizzo
if __name__ == "__main__":
    # Crea sistema con ML - usa webcam (0) o file video
    system = MLEnhancedInterceptSystem(
        video_source=0,  # Webcam
        model_path="updated_model.pth"  # Modello esistente
    )
    
    print("=" * 60)
    print("🚀 SISTEMA ML TRACKING OGGETTI ATTIVO")
    print("=" * 60)
    print("Comandi:")
    print("  Q - Quit (uscire)")
    print("  S - Salva modello")
    print("  R - Reset tracks")
    print("=" * 60)
    
    try:
        # Training iniziale se hai dati storici
        # system.train_on_historical_data("missile_trajectories.pkl")
        
        frame_count = 0
        
        # Esegui sistema
        while True:
            ret, frame = system.cap.read()
            if not ret:
                print("❌ Impossibile leggere dalla camera. Verificare la connessione.")
                break
                
            frame_count += 1
            
            try:
                processed_frame, predictions = system.process_frame(frame)
                
                # Aggiungi informazioni di stato
                cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(processed_frame, "Premi 'Q' per uscire", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Mostra risultati
                cv2.imshow("🚀 ML Object Tracker & Velocity System", processed_frame)
                
                # Comandi keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("🛑 Uscita richiesta dall'utente")
                    break
                elif key == ord('s') or key == ord('S'):
                    system.save_model("updated_model.pth")
                    print("💾 Modello salvato!")
                elif key == ord('r') or key == ord('R'):
                    system.tracks.clear()
                    print("🔄 Tracks reset!")
                    
            except Exception as e:
                print(f"⚠️  Errore processing frame {frame_count}: {e}")
                continue
        
    except KeyboardInterrupt:
        print("\n🛑 Interruzione da tastiera")
    except Exception as e:
        print(f"❌ Errore generale: {e}")
    finally:
        # Salva modello aggiornato
        try:
            system.save_model("updated_model.pth")
            print("💾 Modello finale salvato")
        except Exception as e:
            print(f"⚠️  Errore salvataggio: {e}")
        
        system.cap.release()
        cv2.destroyAllWindows()
        print("✅ Sistema chiuso correttamente")