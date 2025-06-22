import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pickle
from sklearn.preprocessing import StandardScaler
import os

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
    Rete neurale LSTM per predire manovre future dei missili.
    Input: sequenza di stati passati (pos, vel, acc)
    Output: traiettoria futura predetta
    """
    
    def __init__(self, input_size=9, hidden_size=128, num_layers=3, 
                 output_size=9, prediction_horizon=30):
        super(MissileManeuverPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM per catturare pattern temporali
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Attention mechanism per focus su manovre critiche
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Decoder per predizione multi-step
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size * prediction_horizon)
        )
        
        # Classificatore tipo manovra
        self.maneuver_classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # 7 tipi di manovra
        )
        
    def forward(self, x, return_attention=False):
        # x shape: [batch, sequence_length, features]
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention su sequenza
        attn_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Combina LSTM e attention
        combined = lstm_out + attn_out
        
        # Usa ultimo hidden state per predizione
        last_hidden = combined[:, -1, :]
        
        # Predici traiettoria futura
        future_trajectory = self.decoder(last_hidden)
        future_trajectory = future_trajectory.view(
            batch_size, self.prediction_horizon, -1
        )
        
        # Classifica tipo di manovra
        maneuver_type = self.maneuver_classifier(last_hidden)
        
        if return_attention:
            return future_trajectory, maneuver_type, attention_weights
        return future_trajectory, maneuver_type

class MLInterceptSystem:
    """
    Sistema di intercettazione con ML per predizione manovre
    """
    
    def __init__(self, model_path: Optional[str] = None):
        # Modello ML
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = MissileManeuverPredictor().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.predictor.load_state_dict(torch.load(model_path))
            print(f"Modello caricato da {model_path}")
        
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
        Estrae features rilevanti per ML da track
        """
        positions = np.array(track['positions'])
        timestamps = np.array(track['timestamps'])
        
        if len(positions) < 2:
            return None
            
        # Calcola velocità e accelerazioni
        velocities = []
        accelerations = []
        
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                vel = (positions[i] - positions[i-1]) / dt
                velocities.append(vel)
                
                if i > 1 and len(velocities) > 1:
                    acc = (velocities[-1] - velocities[-2]) / dt
                    accelerations.append(acc)
                else:
                    accelerations.append(np.zeros(3))
        
        velocities = np.array(velocities)
        accelerations = np.array(accelerations[:-1] if accelerations else [])
        
        # Features aggiuntive
        features = []
        
        for i in range(len(velocities)):
            if i < len(accelerations):
                # Features base: posizione, velocità, accelerazione
                base_features = np.concatenate([
                    positions[i+1],
                    velocities[i],
                    accelerations[i]
                ])
                
                # Features derivate
                speed = np.linalg.norm(velocities[i])
                acceleration_magnitude = np.linalg.norm(accelerations[i])
                
                # Angoli di attacco e virata
                if speed > 0:
                    heading = np.arctan2(velocities[i][1], velocities[i][0])
                    pitch = np.arcsin(velocities[i][2] / speed)
                else:
                    heading = pitch = 0
                
                # Rate di cambio angolare
                if i > 0:
                    prev_speed = np.linalg.norm(velocities[i-1])
                    if prev_speed > 0:
                        prev_heading = np.arctan2(velocities[i-1][1], velocities[i-1][0])
                        heading_rate = (heading - prev_heading) / dt
                    else:
                        heading_rate = 0
                else:
                    heading_rate = 0
                
                # Usa solo le features base (9 dimensioni: pos[3] + vel[3] + acc[3])
                features.append(base_features)
        
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
            future_trajectory, maneuver_logits = self.predictor(input_tensor)
            
        # Converti output
        future_trajectory = future_trajectory.squeeze(0).cpu().numpy()
        maneuver_probs = torch.softmax(maneuver_logits, dim=1).squeeze(0).cpu().numpy()
        
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
                    
                    # Crea/aggiorna track
                    track_id = f"track_{cls}_{int(center_x)}_{int(center_y)}"
                    
                    if track_id not in self.tracks:
                        self.tracks[track_id] = {
                            'positions': [],
                            'timestamps': [],
                            'class': cls,
                            'confidence': []
                        }
                    
                    # Aggiungi posizione (simuliamo 3D per ora)
                    world_pos = np.array([center_x, center_y, 0])
                    self.tracks[track_id]['positions'].append(world_pos)
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
        
        # ML predictions per ogni track
        ml_predictions = {}
        
        for track_id, track in self.tracks.items():
            if len(track['positions']) >= min(5, self.sequence_length):  # Ridotto per test
                try:
                    ml_pred = self.predict_trajectory(track)
                    ml_predictions[track_id] = ml_pred
                    
                    # Visualizza predizione
                    self._draw_ml_prediction(frame, track, ml_pred)
                except Exception as e:
                    print(f"Errore predizione per {track_id}: {e}")
        
        # Statistiche su frame
        cv2.putText(frame, f"Tracks: {len(self.tracks)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"ML Predictions: {len(ml_predictions)}", (10, 60), 
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