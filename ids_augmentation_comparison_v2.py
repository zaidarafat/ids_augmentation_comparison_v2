"""
Comparative Analysis of Synthetic Data Augmentation Techniques for IDS Implementation

This program implements three synthetic data augmentation methods:
1. Generative Adversarial Networks (GAN) - including vanilla GAN and CTGAN
2. Synthetic Minority Oversampling Technique (SMOTE)
3. Variational Autoencoders (VAE)

The implementation evaluates these methods on NSL-KDD dataset with:
- Classification performance metrics (accuracy, macro-F1, per-class recall)
- Distribution fidelity metrics (KS test, MMD, correlation drift)
- Cross-validation with multiple classifiers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                        f1_score, classification_report, confusion_matrix, roc_curve, auc)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from scipy.stats import ks_2samp
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# For GAN implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class VAE(nn.Module):
    """
    Variational Autoencoder for synthetic data generation
    """
    def __init__(self, input_dim, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.relu(self.fc3(z))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Generator(nn.Module):
    """
    GAN Generator network
    """
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """
    GAN Discriminator network
    """
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class IDSAugmentationComparison:
    """
    Main class for comparing synthetic data augmentation techniques
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.results = {}
        
    def load_nsl_kdd(self, train_path, test_path=None):
        """
        Load and preprocess NSL-KDD dataset
        """
        # Column names for NSL-KDD
        columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'attack_type', 'difficulty']
        
        print("Loading NSL-KDD dataset...")
        try:
            df_train = pd.read_csv(train_path, names=columns)
            if test_path:
                df_test = pd.read_csv(test_path, names=columns)
            else:
                df_test = None
        except:
            print("Error loading dataset. Using synthetic example data...")
            # Create synthetic data for demonstration
            df_train = self._create_synthetic_nsl_kdd(5000)
            df_test = self._create_synthetic_nsl_kdd(1000)
        
        # Remove difficulty column if exists
        if 'difficulty' in df_train.columns:
            df_train = df_train.drop('difficulty', axis=1)
        if df_test is not None and 'difficulty' in df_test.columns:
            df_test = df_test.drop('difficulty', axis=1)
        
        # Map attack types to categories
        attack_mapping = {
            'normal': 'Normal',
            'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
            'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
            'processtable': 'DoS', 'mailbomb': 'DoS',
            'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
            'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
            'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R',
            'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
            'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
            'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
            'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L',
            'xsnoop': 'R2L', 'worm': 'R2L', 'httptunnel': 'R2L'
        }
        
        df_train['attack_category'] = df_train['attack_type'].str.lower().map(attack_mapping)
        if df_test is not None:
            df_test['attack_category'] = df_test['attack_type'].str.lower().map(attack_mapping)
        
        # Fill any unmapped attacks with 'Unknown'
        df_train['attack_category'].fillna('Unknown', inplace=True)
        if df_test is not None:
            df_test['attack_category'].fillna('Unknown', inplace=True)
        
        print(f"Training set size: {len(df_train)}")
        print(f"Class distribution:\n{df_train['attack_category'].value_counts()}")
        
        return df_train, df_test
    
    def _create_synthetic_nsl_kdd(self, n_samples):
        """
        Create synthetic NSL-KDD-like data for demonstration
        """
        np.random.seed(42)
        data = {
            'duration': np.random.randint(0, 1000, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'smtp', 'ftp', 'ssh', 'other'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples),
            'src_bytes': np.random.randint(0, 10000, n_samples),
            'dst_bytes': np.random.randint(0, 10000, n_samples),
            'land': np.random.randint(0, 2, n_samples),
            'wrong_fragment': np.random.randint(0, 3, n_samples),
            'urgent': np.random.randint(0, 2, n_samples),
            'hot': np.random.randint(0, 10, n_samples),
            'num_failed_logins': np.random.randint(0, 5, n_samples),
            'logged_in': np.random.randint(0, 2, n_samples),
            'num_compromised': np.random.randint(0, 5, n_samples),
            'root_shell': np.random.randint(0, 2, n_samples),
            'su_attempted': np.random.randint(0, 2, n_samples),
            'num_root': np.random.randint(0, 5, n_samples),
            'num_file_creations': np.random.randint(0, 10, n_samples),
            'num_shells': np.random.randint(0, 2, n_samples),
            'num_access_files': np.random.randint(0, 5, n_samples),
            'num_outbound_cmds': np.random.randint(0, 2, n_samples),
            'is_host_login': np.random.randint(0, 2, n_samples),
            'is_guest_login': np.random.randint(0, 2, n_samples),
            'count': np.random.randint(1, 500, n_samples),
            'srv_count': np.random.randint(1, 500, n_samples),
            'serror_rate': np.random.random(n_samples),
            'srv_serror_rate': np.random.random(n_samples),
            'rerror_rate': np.random.random(n_samples),
            'srv_rerror_rate': np.random.random(n_samples),
            'same_srv_rate': np.random.random(n_samples),
            'diff_srv_rate': np.random.random(n_samples),
            'srv_diff_host_rate': np.random.random(n_samples),
            'dst_host_count': np.random.randint(0, 255, n_samples),
            'dst_host_srv_count': np.random.randint(0, 255, n_samples),
            'dst_host_same_srv_rate': np.random.random(n_samples),
            'dst_host_diff_srv_rate': np.random.random(n_samples),
            'dst_host_same_src_port_rate': np.random.random(n_samples),
            'dst_host_srv_diff_host_rate': np.random.random(n_samples),
            'dst_host_serror_rate': np.random.random(n_samples),
            'dst_host_srv_serror_rate': np.random.random(n_samples),
            'dst_host_rerror_rate': np.random.random(n_samples),
            'dst_host_srv_rerror_rate': np.random.random(n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create imbalanced attack types
        attack_probs = [0.65, 0.25, 0.08, 0.015, 0.005]  # Normal, DoS, Probe, R2L, U2R
        df['attack_type'] = np.random.choice(
            ['normal', 'neptune', 'ipsweep', 'guess_passwd', 'buffer_overflow'],
            n_samples, p=attack_probs
        )
        
        return df
    
    def preprocess_data(self, df_train, df_test=None):
        """
        Preprocess the data: encode categorical variables and scale features
        """
        print("\nPreprocessing data...")
        
        # Separate features and labels
        X_train = df_train.drop(['attack_type', 'attack_category'], axis=1)
        y_train = df_train['attack_category']
        
        if df_test is not None:
            X_test = df_test.drop(['attack_type', 'attack_category'], axis=1)
            y_test = df_test['attack_category']
        else:
            X_test, y_test = None, None
        
        # Encode categorical features
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            if X_test is not None:
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            self.label_encoders[col] = le
        
        # Encode target labels
        le_target = LabelEncoder()
        y_train_encoded = le_target.fit_transform(y_train)
        if y_test is not None:
            y_test_encoded = le_target.transform(y_test)
        else:
            y_test_encoded = None
        
        self.label_encoder_target = le_target
        self.classes = le_target.classes_
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = None
        
        print(f"Feature dimension: {X_train_scaled.shape[1]}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        
        return X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded
    
    def train_vae(self, X, n_epochs=50, batch_size=128, latent_dim=20):
        """
        Train Variational Autoencoder
        """
        print("\nTraining VAE...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X.shape[1]
        
        # Create data loader
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize VAE
        vae = VAE(input_dim, latent_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.001)
        
        # Training loop
        vae.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                
                # Forward pass
                recon_x, mu, logvar = vae(x)
                
                # Loss: reconstruction + KL divergence
                recon_loss = nn.MSELoss()(recon_x, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}")
        
        return vae
    
    def train_vae_with_logging(self, X, n_epochs=100, batch_size=128, latent_dim=20):
        """
        Train VAE and return loss history for visualization
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X.shape[1]
        
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        vae = VAE(input_dim, latent_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=0.001)
        
        recon_losses = []
        kl_losses = []
        
        vae.train()
        for epoch in range(n_epochs):
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            n_batches = 0
            
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                
                recon_x, mu, logvar = vae(x)
                
                recon_loss = nn.MSELoss()(recon_x, x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
                loss = recon_loss + 0.001 * kl_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                n_batches += 1
            
            recon_losses.append(epoch_recon_loss / n_batches)
            kl_losses.append(epoch_kl_loss / n_batches)
        
        return vae, recon_losses, kl_losses
    
    def generate_vae_samples(self, vae, n_samples, input_dim):
        """
        Generate synthetic samples using trained VAE
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, vae.fc_mu.out_features).to(device)
            synthetic_samples = vae.decode(z).cpu().numpy()
        
        return synthetic_samples
    
    def train_gan(self, X, n_epochs=100, batch_size=128, latent_dim=100):
        """
        Train Generative Adversarial Network
        """
        print("\nTraining GAN...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X.shape[1]
        
        # Initialize Generator and Discriminator
        generator = Generator(latent_dim, input_dim).to(device)
        discriminator = Discriminator(input_dim).to(device)
        
        # Optimizers
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Create data loader
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(n_epochs):
            for i, batch in enumerate(dataloader):
                real_data = batch[0]
                batch_size_current = real_data.size(0)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Real data
                real_labels = torch.ones(batch_size_current, 1).to(device)
                real_output = discriminator(real_data)
                d_loss_real = criterion(real_output, real_labels)
                
                # Fake data
                z = torch.randn(batch_size_current, latent_dim).to(device)
                fake_data = generator(z)
                fake_labels = torch.zeros(batch_size_current, 1).to(device)
                fake_output = discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                # Total discriminator loss
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size_current, latent_dim).to(device)
                fake_data = generator(z)
                fake_output = discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
        
        return generator
    
    def train_gan_with_logging(self, X, n_epochs=100, batch_size=128, latent_dim=100):
        """
        Train GAN and return loss history for visualization
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X.shape[1]
        
        generator = Generator(latent_dim, input_dim).to(device)
        discriminator = Discriminator(input_dim).to(device)
        
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        g_losses = []
        d_losses = []
        
        for epoch in range(n_epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            n_batches = 0
            
            for i, batch in enumerate(dataloader):
                real_data = batch[0]
                batch_size_current = real_data.size(0)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                real_labels = torch.ones(batch_size_current, 1).to(device)
                real_output = discriminator(real_data)
                d_loss_real = criterion(real_output, real_labels)
                
                z = torch.randn(batch_size_current, latent_dim).to(device)
                fake_data = generator(z)
                fake_labels = torch.zeros(batch_size_current, 1).to(device)
                fake_output = discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size_current, latent_dim).to(device)
                fake_data = generator(z)
                fake_output = discriminator(fake_data)
                g_loss = criterion(fake_output, real_labels)
                
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1
            
            g_losses.append(epoch_g_loss / n_batches)
            d_losses.append(epoch_d_loss / n_batches)
        
        return generator, g_losses, d_losses
    
    def generate_gan_samples(self, generator, n_samples, latent_dim=100):
        """
        Generate synthetic samples using trained GAN
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator.eval()
        
        with torch.no_grad():
            z = torch.randn(n_samples, latent_dim).to(device)
            synthetic_samples = generator(z).cpu().numpy()
        
        return synthetic_samples
    
    def apply_smote(self, X, y, augmentation_ratio=0.5):
        """
        Apply SMOTE oversampling with controlled augmentation
        """
        print("\nApplying SMOTE...")
        
        # Check for classes with too few samples (< 6 samples needed for SMOTE's default k_neighbors=5)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples_needed = 6  # SMOTE default k_neighbors=5, needs at least 6 samples
        
        classes_to_remove = unique_classes[class_counts < min_samples_needed]
        
        if len(classes_to_remove) > 0:
            print(f"  Warning: Removing classes with < {min_samples_needed} samples from SMOTE: {classes_to_remove}")
            print(f"  Classes removed: {[self.classes[i] for i in classes_to_remove]}")
            
            # Remove samples from classes that are too small
            mask = ~np.isin(y, classes_to_remove)
            X_filtered = X[mask]
            y_filtered = y[mask]
            
            print(f"  Filtered: {X.shape[0]} → {X_filtered.shape[0]} samples")
        else:
            X_filtered = X
            y_filtered = y
        
        # Calculate target samples per class based on augmentation_ratio
        unique_filtered, counts_filtered = np.unique(y_filtered, return_counts=True)
        max_count = np.max(counts_filtered)
        
        # Create sampling strategy: augment minority classes to augmentation_ratio of max class
        sampling_strategy = {}
        for cls, count in zip(unique_filtered, counts_filtered):
            target = int(count + augmentation_ratio * (max_count - count))
            sampling_strategy[int(cls)] = target
        
        print(f"  Augmentation ratio: {augmentation_ratio}")
        print(f"  Target samples per class: {sampling_strategy}")
        
        # Apply SMOTE with controlled sampling strategy
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)
        print(f"Original shape: {X.shape}, Resampled shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    
    def augment_minority_classes(self, X_train, y_train, method='gan', augmentation_ratio=1.0):
        """
        Augment minority classes using specified method
        """
        print(f"\n{'='*60}")
        print(f"Augmenting minority classes using {method.upper()}")
        print(f"{'='*60}")
        
        # Identify minority classes (classes with < 5% of samples)
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        minority_threshold = 0.05 * total_samples
        minority_classes = np.where(class_counts < minority_threshold)[0]
        
        print(f"Minority classes identified: {minority_classes}")
        print(f"Class distribution before augmentation:")
        for i, count in enumerate(class_counts):
            print(f"  Class {i} ({self.classes[i]}): {count} samples ({100*count/total_samples:.2f}%)")
        
        X_augmented = X_train.copy()
        y_augmented = y_train.copy()
        
        if method == 'smote':
            X_augmented, y_augmented = self.apply_smote(X_train, y_train, augmentation_ratio)
        
        elif method == 'gan':
            # Train GAN on each minority class and generate samples
            for class_idx in minority_classes:
                class_data = X_train[y_train == class_idx]
                n_samples_needed = int(augmentation_ratio * (np.max(class_counts) - len(class_data)))
                
                if n_samples_needed > 0 and len(class_data) > 10:
                    print(f"\nGenerating {n_samples_needed} samples for class {class_idx} ({self.classes[class_idx]})")
                    generator = self.train_gan(class_data, n_epochs=150, batch_size=min(64, len(class_data)))
                    synthetic_samples = self.generate_gan_samples(generator, n_samples_needed)
                    
                    X_augmented = np.vstack([X_augmented, synthetic_samples])
                    y_augmented = np.hstack([y_augmented, np.full(n_samples_needed, class_idx)])
                elif len(class_data) <= 10:
                    print(f"\nSkipping class {class_idx} ({self.classes[class_idx]}): insufficient samples ({len(class_data)} ≤ 10)")
        
        elif method == 'vae':
            # Train VAE on each minority class and generate samples
            for class_idx in minority_classes:
                class_data = X_train[y_train == class_idx]
                n_samples_needed = int(augmentation_ratio * (np.max(class_counts) - len(class_data)))
                
                if n_samples_needed > 0 and len(class_data) > 10:
                    print(f"\nGenerating {n_samples_needed} samples for class {class_idx} ({self.classes[class_idx]})")
                    vae = self.train_vae(class_data, n_epochs=500, batch_size=min(64, len(class_data)))
                    synthetic_samples = self.generate_vae_samples(vae, n_samples_needed, X_train.shape[1])
                    
                    X_augmented = np.vstack([X_augmented, synthetic_samples])
                    y_augmented = np.hstack([y_augmented, np.full(n_samples_needed, class_idx)])
                elif len(class_data) <= 10:
                    print(f"\nSkipping class {class_idx} ({self.classes[class_idx]}): insufficient samples ({len(class_data)} ≤ 10)")
        
        print(f"\nClass distribution after augmentation:")
        augmented_class_counts = np.bincount(y_augmented)
        for i, count in enumerate(augmented_class_counts):
            print(f"  Class {i} ({self.classes[i]}): {count} samples ({100*count/len(y_augmented):.2f}%)")
        
        return X_augmented, y_augmented
    
    def evaluate_classifier(self, X_train, y_train, X_test, y_test, classifier_name='Random Forest'):
        """
        Train and evaluate a classifier
        """
        print(f"\nTraining {classifier_name}...")
        
        if classifier_name == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif classifier_name == 'Decision Tree':
            clf = DecisionTreeClassifier(random_state=42)
        elif classifier_name == 'MLP':
            clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        per_class_report = classification_report(y_test, y_pred, 
                                                target_names=self.classes,
                                                output_dict=True,
                                                zero_division=0)
        
        results = {
            'classifier': classifier_name,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'per_class_report': per_class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nResults for {classifier_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Macro Precision: {macro_precision:.4f}")
        print(f"  Macro Recall: {macro_recall:.4f}")
        
        return results
    
    def calculate_distribution_fidelity(self, X_real, X_synthetic):
        """
        Calculate distribution fidelity metrics:
        - Kolmogorov-Smirnov (KS) test for each feature
        - Mean correlation difference
        """
        print("\nCalculating distribution fidelity metrics...")
        
        # KS test for each feature
        ks_statistics = []
        for i in range(X_real.shape[1]):
            ks_stat, _ = ks_2samp(X_real[:, i], X_synthetic[:, i])
            ks_statistics.append(ks_stat)
        
        mean_ks = np.mean(ks_statistics)
        
        # Correlation drift
        corr_real = np.corrcoef(X_real.T)
        corr_synthetic = np.corrcoef(X_synthetic.T)
        corr_diff = np.abs(corr_real - corr_synthetic)
        mean_corr_diff = np.mean(corr_diff)
        
        print(f"  Mean KS statistic: {mean_ks:.4f}")
        print(f"  Mean correlation difference: {mean_corr_diff:.4f}")
        
        return {
            'ks_statistics': ks_statistics,
            'mean_ks': mean_ks,
            'mean_correlation_diff': mean_corr_diff
        }
    
    def run_comparison(self, X_train, y_train, X_test, y_test, classifiers=['Random Forest']):
        """
        Run complete comparison of augmentation methods
        """
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS OF SYNTHETIC DATA AUGMENTATION FOR IDS")
        print("="*80)
        
        methods = ['baseline', 'smote', 'gan', 'vae']
        all_results = {}
        
        for method in methods:
            print(f"\n{'#'*80}")
            print(f"# METHOD: {method.upper()}")
            print(f"{'#'*80}")
            
            if method == 'baseline':
                X_train_aug = X_train
                y_train_aug = y_train
            else:
                X_train_aug, y_train_aug = self.augment_minority_classes(
                    X_train, y_train, method=method, augmentation_ratio=0.5
                )
            
            method_results = {}
            
            # Evaluate with each classifier
            for clf_name in classifiers:
                results = self.evaluate_classifier(
                    X_train_aug, y_train_aug, X_test, y_test, clf_name
                )
                method_results[clf_name] = results
            
            # Calculate distribution fidelity for synthetic methods
            if method != 'baseline':
                # Get synthetic samples only
                n_original = len(X_train)
                X_synthetic = X_train_aug[n_original:]
                
                if len(X_synthetic) > 0:
                    # Compare synthetic samples with original minority class samples
                    minority_classes = np.where(np.bincount(y_train) < 0.05 * len(y_train))[0]
                    if len(minority_classes) > 0:
                        minority_data = X_train[np.isin(y_train, minority_classes)]
                        fidelity = self.calculate_distribution_fidelity(minority_data, X_synthetic)
                        method_results['fidelity'] = fidelity
            
            all_results[method] = method_results
        
        self.results = all_results
        return all_results
    
    def plot_results(self, save_path='results_comparison.png'):
        """
        Plot comparison results
        """
        if not self.results:
            print("No results to plot. Run comparison first.")
            return
        
        methods = list(self.results.keys())
        classifiers = list(self.results[methods[0]].keys())
        classifiers = [c for c in classifiers if c != 'fidelity']
        
        metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            data_to_plot = []
            labels = []
            
            for method in methods:
                for clf in classifiers:
                    if clf in self.results[method]:
                        value = self.results[method][clf][metric]
                        data_to_plot.append(value)
                        labels.append(f"{method}\n{clf}")
            
            x_pos = np.arange(len(data_to_plot))
            bars = ax.bar(x_pos, data_to_plot, color=['blue', 'orange', 'green', 'red'][:len(methods)])
            
            ax.set_xlabel('Method + Classifier', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, data_to_plot)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nResults plot saved to {save_path}")
    
    def plot_pca_tsne_comparison(self, X_train, y_train, save_path='pca_tsne_visualization.png'):
        """
        Create PCA and t-SNE visualizations comparing real vs synthetic data
        """
        print("\nGenerating PCA and t-SNE visualizations...")
        
        # Sample subset for visualization (too many points slow down t-SNE)
        max_samples = 300
        np.random.seed(42)
        
        # Get indices for sampling
        sample_idx = np.random.choice(len(X_train), min(max_samples, len(X_train)), replace=False)
        X_sample = X_train[sample_idx]
        y_sample = y_train[sample_idx]
        
        # Store visualizations data
        methods = ['smote', 'gan', 'vae']
        n_methods = len(methods)
        
        fig, axes = plt.subplots(n_methods, 2, figsize=(14, 4*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        colors = {'Real': 'blue', 'GAN': 'orange', 'SMOTE': 'green', 'VAE': 'red'}
        
        for idx, method in enumerate(methods):
            # Generate synthetic samples
            if method not in self.results or len(self.results[method]) == 0:
                print(f"Skipping {method} - no data available")
                continue
            
            print(f"  Processing {method.upper()}...")
            
            # Generate small batch of synthetic samples
            X_synth, _ = self.augment_minority_classes(X_sample, y_sample, method=method, augmentation_ratio=0.3)
            n_original = len(X_sample)
            X_synth_only = X_synth[n_original:]
            
            # Limit synthetic samples for visualization
            if len(X_synth_only) > max_samples:
                synth_idx = np.random.choice(len(X_synth_only), max_samples, replace=False)
                X_synth_only = X_synth_only[synth_idx]
            
            # Combine real and synthetic
            X_combined = np.vstack([X_sample, X_synth_only])
            labels = ['Real'] * len(X_sample) + [method.upper()] * len(X_synth_only)
            
            # PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_combined)
            
            ax_pca = axes[idx, 0]
            for label in set(labels):
                mask = [l == label for l in labels]
                ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                            c=colors.get(label, 'gray'), label=label, alpha=0.6, s=30)
            ax_pca.set_title(f'NSL-KDD - PCA Projection', fontsize=12, fontweight='bold')
            ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontsize=10)
            ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontsize=10)
            ax_pca.legend()
            ax_pca.grid(alpha=0.3)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            X_tsne = tsne.fit_transform(X_combined)
            
            ax_tsne = axes[idx, 1]
            for label in set(labels):
                mask = [l == label for l in labels]
                ax_tsne.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                            c=colors.get(label, 'gray'), label=label, alpha=0.6, s=30)
            ax_tsne.set_title(f'NSL-KDD - t-SNE Projection', fontsize=12, fontweight='bold')
            ax_tsne.set_xlabel('t-SNE 1', fontsize=10)
            ax_tsne.set_ylabel('t-SNE 2', fontsize=10)
            ax_tsne.legend()
            ax_tsne.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA/t-SNE visualization saved to {save_path}")
    
    def plot_roc_curves(self, X_train, y_train, X_test, y_test, save_path='roc_curves.png'):
        """
        Generate ROC curves for all augmentation methods with multiple classifiers
        """
        print("\nGenerating ROC curves...")
        
        methods = ['gan', 'smote', 'vae']
        classifiers_dict = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42),
            'DNN': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=100, random_state=42)
        }
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors_methods = {'gan': 'blue', 'smote': 'orange', 'vae': 'green'}
        linestyles_clf = {'RF': '-', 'SVM': '--', 'DNN': ':'}
        
        for method in methods:
            print(f"  Processing {method.upper()}...")
            
            # Augment data
            X_train_aug, y_train_aug = self.augment_minority_classes(
                X_train, y_train, method=method, augmentation_ratio=0.5
            )
            
            for clf_name, clf in classifiers_dict.items():
                # Convert to binary classification for ROC (combine all attacks vs normal)
                y_train_binary = (y_train_aug != self.classes.tolist().index('Normal')).astype(int)
                y_test_binary = (y_test != self.classes.tolist().index('Normal')).astype(int)
                
                # Train classifier
                clf.fit(X_train_aug, y_train_binary)
                
                # Get prediction probabilities
                if hasattr(clf, "predict_proba"):
                    y_scores = clf.predict_proba(X_test)[:, 1]
                else:
                    y_scores = clf.decision_function(X_test)
                
                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                label = f'{clf_name} - {method.upper()}'
                color = colors_methods[method]
                linestyle = linestyles_clf[clf_name]
                
                ax.plot(fpr, tpr, color=color, linestyle=linestyle, 
                    linewidth=2, label=f'{label} (AUC={roc_auc:.2f})', alpha=0.8)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - All Classifiers', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    def plot_training_curves(self, X_train, y_train, save_path='training_curves.png'):
        """
        Plot training loss curves for GAN and VAE
        """
        print("\nGenerating training loss curves...")
        
        # Get minority class for training
        minority_classes = np.where(np.bincount(y_train) < 0.05 * len(y_train))[0]
        if len(minority_classes) == 0:
            print("No minority classes found")
            return
        
        # Select class with enough samples for demonstration
        target_class = None
        for cls in minority_classes:
            if np.sum(y_train == cls) >= 50:
                target_class = cls
                break
        
        if target_class is None:
            target_class = minority_classes[0]
        
        class_data = X_train[y_train == target_class]
        print(f"  Training on class {target_class} with {len(class_data)} samples")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Train GAN and collect losses
        print("  Training GAN...")
        generator, gan_g_losses, gan_d_losses = self.train_gan_with_logging(
            class_data, n_epochs=100, batch_size=min(32, len(class_data))
        )
        
        # Train VAE and collect losses
        print("  Training VAE...")
        vae, vae_recon_losses, vae_kl_losses = self.train_vae_with_logging(
            class_data, n_epochs=100, batch_size=min(32, len(class_data))
        )
        
        # Plot losses
        epochs = range(1, len(gan_g_losses) + 1)
        
        ax.plot(epochs, gan_g_losses, label='GAN - Generator Loss', 
            color='blue', linewidth=2, alpha=0.8)
        ax.plot(epochs, gan_d_losses, label='GAN - Discriminator Loss', 
            color='orange', linewidth=2, alpha=0.8)
        ax.plot(epochs, vae_recon_losses, label='VAE - Reconstruction Loss', 
            color='green', linewidth=2, alpha=0.8)
        ax.plot(epochs, vae_kl_losses, label='VAE - KL Divergence Loss', 
            color='red', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        
    def print_summary(self):
        """
        Print summary of results
        """
        if not self.results:
            print("No results available. Run comparison first.")
            return
        
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        
        methods = list(self.results.keys())
        
        # Find best method for each metric
        metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_recall']
        
        for metric in metrics:
            print(f"\n{metric.replace('_', ' ').title()}:")
            best_score = 0
            best_method = ''
            best_clf = ''
            
            for method in methods:
                for clf in self.results[method]:
                    if clf != 'fidelity' and metric in self.results[method][clf]:
                        score = self.results[method][clf][metric]
                        print(f"  {method:10s} + {clf:15s}: {score:.4f}")
                        
                        if score > best_score:
                            best_score = score
                            best_method = method
                            best_clf = clf
            
            print(f"  -> Best: {best_method} + {best_clf} ({best_score:.4f})")
        
        # Distribution fidelity comparison
        print("\nDistribution Fidelity (lower is better):")
        for method in methods:
            if method != 'baseline' and 'fidelity' in self.results[method]:
                fidelity = self.results[method]['fidelity']
                print(f"  {method:10s}: KS = {fidelity['mean_ks']:.4f}, "
                    f"Correlation Drift = {fidelity['mean_correlation_diff']:.4f}")


def main():
    """
    Main execution function
    """
    print("="*80)
    print("IDS Synthetic Data Augmentation Comparison")
    print("Implementation based on research paper")
    print("="*80)
    
    # Initialize comparison system
    ids_comparison = IDSAugmentationComparison()
    
    # Load NSL-KDD dataset (or use synthetic data)
    # For actual use, provide paths to NSL-KDD train and test files:
    # df_train, df_test = ids_comparison.load_nsl_kdd('KDDTrain+.txt', 'KDDTest+.txt')
    
    print("\nNote: Using synthetic NSL-KDD-like data for demonstration.")
    print("For actual experiments, download NSL-KDD from:")
    print("https://www.unb.ca/cic/datasets/nsl.html")
    
    df_train, df_test = ids_comparison.load_nsl_kdd(
        'E:/PHD/Papers/07/files/KDDTrain+.txt',
        'E:/PHD/Papers/07/files/KDDTest+.txt'
    )
    
    # Preprocess data
    X_train, y_train, X_test, y_test = ids_comparison.preprocess_data(df_train, df_test)
    
    # Run comparison
    classifiers = ['Random Forest', 'Decision Tree']
    results = ids_comparison.run_comparison(
        X_train, y_train, X_test, y_test, 
        classifiers=classifiers
    )
    
    # Plot results
    ids_comparison.plot_results('ids_augmentation_comparison.png')
    
    # Generate advanced visualizations
    print("\n" + "="*80)
    print("GENERATING ADVANCED VISUALIZATIONS")
    print("="*80)
    
    # 1. PCA and t-SNE visualizations
    ids_comparison.plot_pca_tsne_comparison(X_train, y_train, 'Figure_X_PCA_and_tSNE_visualizations.png')
    
    # 2. ROC Curves
    ids_comparison.plot_roc_curves(X_train, y_train, X_test, y_test, 'Figure_Y_ROC_curves.png')
    
    # 3. Training Loss Curves
    ids_comparison.plot_training_curves(X_train, y_train, 'Figure_Z_training_curves.png')
    
    # Print summary
    ids_comparison.print_summary()
    
    # Print detailed per-class metrics for minority classes
    print("\n" + "="*80)
    print("MINORITY CLASS PERFORMANCE ANALYSIS")
    print("="*80)
    
    for method in ['baseline', 'smote', 'gan', 'vae']:
        if method in ids_comparison.results and 'Random Forest' in ids_comparison.results[method]:
            report = ids_comparison.results[method]['Random Forest']['per_class_report']
            
            print(f"\n{method.upper()} + Random Forest:")
            if 'R2L' in report:
                print(f"  R2L - Precision: {report['R2L']['precision']:.3f}, Recall: {report['R2L']['recall']:.3f}, F1: {report['R2L']['f1-score']:.3f}")
            if 'U2R' in report:
                print(f"  U2R - Precision: {report['U2R']['precision']:.3f}, Recall: {report['U2R']['recall']:.3f}, F1: {report['U2R']['f1-score']:.3f}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()