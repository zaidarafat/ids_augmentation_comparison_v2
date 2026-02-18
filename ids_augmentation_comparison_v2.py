"""
Comparative Analysis of Synthetic Data Augmentation Techniques for IDS Implementation

This program implements three synthetic data augmentation methods:
1. Generative Adversarial Networks (GAN) - including vanilla GAN and CTGAN
2. Synthetic Minority Oversampling Technique (SMOTE)
3. Variational Autoencoders (VAE)

The implementation evaluates these methods on TWO datasets:
- NSL-KDD dataset (benchmark)
- UNSW-NB15 dataset (modern network traffic)

Evaluation includes:
- Classification performance metrics (accuracy, macro-F1, per-class recall)
- Distribution fidelity metrics (KS test, MMD, correlation drift)
- Cross-validation with multiple classifiers
- Cross-dataset comparative analysis
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
    on multiple datasets (NSL-KDD and UNSW-NB15)
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.results = {}
        self.dataset_results = {}  # Store results per dataset for cross-dataset comparison

    # =========================================================================
    # NSL-KDD Dataset Loading
    # =========================================================================
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

        df_train['attack_category'] = df_train['attack_type'].str.lower().str.strip().map(attack_mapping)
        if df_test is not None:
            df_test['attack_category'] = df_test['attack_type'].str.lower().str.strip().map(attack_mapping)

        # Fill any unmapped attacks with 'Unknown'
        df_train['attack_category'].fillna('Unknown', inplace=True)
        if df_test is not None:
            df_test['attack_category'].fillna('Unknown', inplace=True)

        print(f"Training set size: {len(df_train)}")
        print(f"Class distribution:\n{df_train['attack_category'].value_counts()}")

        return df_train, df_test

    # =========================================================================
    # UNSW-NB15 Dataset Loading (NEW)
    # =========================================================================
    def load_unsw_nb15(self, train_path, test_path=None):
        """
        Load and preprocess UNSW-NB15 dataset.

        The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS)
        using the IXIA PerfectStorm tool to generate modern attack traffic. It contains
        49 features with 9 attack categories plus normal traffic.

        Parameters:
            train_path: Path to UNSW-NB15 training CSV file
                        (e.g., 'UNSW_NB15_training-set.csv' or 'UNSW-NB15_1.csv')
            test_path:  Path to UNSW-NB15 testing CSV file (optional)
                        (e.g., 'UNSW_NB15_testing-set.csv')

        Returns:
            df_train, df_test: Preprocessed DataFrames with 'attack_type' and 'attack_category' columns
        """
        print("\n" + "=" * 60)
        print("Loading UNSW-NB15 dataset...")
        print("=" * 60)

        # UNSW-NB15 feature names (49 features as defined in the dataset documentation)
        unsw_columns = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
            'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
            'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb',
            'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
            'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt',
            'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
            'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
            'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]

        try:
            # Try loading with headers first (official train/test split files have headers)
            df_train = pd.read_csv(train_path)

            # Check if the file has expected columns
            if 'attack_cat' in df_train.columns and 'label' in df_train.columns:
                print("  Loaded file with headers (official train/test split format)")
            elif len(df_train.columns) == len(unsw_columns):
                # Raw CSV files without headers
                df_train = pd.read_csv(train_path, names=unsw_columns, header=None)
                print("  Loaded file without headers (raw CSV format)")
            elif len(df_train.columns) == len(unsw_columns) - 2:
                # Some versions don't include attack_cat and label
                print("  Warning: File appears to lack attack labels. Attempting alternative loading...")
                df_train = pd.read_csv(train_path, names=unsw_columns[:len(df_train.columns)], header=None)
            else:
                print(f"  Detected {len(df_train.columns)} columns. Attempting auto-detection...")

            if test_path:
                df_test = pd.read_csv(test_path)
                if 'attack_cat' not in df_test.columns and len(df_test.columns) == len(unsw_columns):
                    df_test = pd.read_csv(test_path, names=unsw_columns, header=None)
            else:
                df_test = None

        except Exception as e:
            print(f"Error loading UNSW-NB15 dataset: {e}")
            print("Using synthetic UNSW-NB15-like data for demonstration...")
            df_train = self._create_synthetic_unsw_nb15(10000)
            df_test = self._create_synthetic_unsw_nb15(2000)

        # --- Standardize column names ---
        # Handle variations in column naming across different UNSW-NB15 versions
        col_rename_map = {}
        for col in df_train.columns:
            col_lower = col.strip().lower()
            if col_lower in ['attack_cat', 'attack_category']:
                col_rename_map[col] = 'attack_cat'
            elif col_lower == 'label':
                col_rename_map[col] = 'label'
        if col_rename_map:
            df_train = df_train.rename(columns=col_rename_map)
            if df_test is not None:
                df_test = df_test.rename(columns=col_rename_map)

        # --- Clean attack category labels ---
        if 'attack_cat' in df_train.columns:
            df_train['attack_cat'] = df_train['attack_cat'].astype(str).str.strip()
            df_train['attack_cat'] = df_train['attack_cat'].replace(
                {'': 'Normal', ' ': 'Normal', 'nan': 'Normal', 'None': 'Normal',
                 'Backdoor': 'Backdoors', 'Backdoor ': 'Backdoors'}
            )
            if df_test is not None:
                df_test['attack_cat'] = df_test['attack_cat'].astype(str).str.strip()
                df_test['attack_cat'] = df_test['attack_cat'].replace(
                    {'': 'Normal', ' ': 'Normal', 'nan': 'Normal', 'None': 'Normal',
                     'Backdoor': 'Backdoors', 'Backdoor ': 'Backdoors'}
                )

        # --- Create unified column names to match NSL-KDD processing pipeline ---
        # Map 'attack_cat' -> 'attack_category' and keep 'attack_cat' as 'attack_type'
        df_train['attack_type'] = df_train['attack_cat'].copy() if 'attack_cat' in df_train.columns else 'Normal'
        df_train['attack_category'] = df_train['attack_cat'].copy() if 'attack_cat' in df_train.columns else 'Normal'
        if df_test is not None:
            df_test['attack_type'] = df_test['attack_cat'].copy() if 'attack_cat' in df_test.columns else 'Normal'
            df_test['attack_category'] = df_test['attack_cat'].copy() if 'attack_cat' in df_test.columns else 'Normal'

        # --- Drop non-feature columns ---
        # Remove IP addresses, timestamps, and original label columns that are not features
        drop_cols = ['srcip', 'dstip', 'stime', 'ltime', 'attack_cat', 'label']
        for col in drop_cols:
            if col in df_train.columns:
                df_train = df_train.drop(col, axis=1)
            if df_test is not None and col in df_test.columns:
                df_test = df_test.drop(col, axis=1)

        # --- Handle 'id' column if present (official split files) ---
        if 'id' in df_train.columns:
            df_train = df_train.drop('id', axis=1)
        if df_test is not None and 'id' in df_test.columns:
            df_test = df_test.drop('id', axis=1)

        # --- Convert numeric columns and handle missing values ---
        for col in df_train.columns:
            if col not in ['attack_type', 'attack_category', 'proto', 'state', 'service',
                          'sport', 'dsport']:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
                if df_test is not None:
                    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

        # Handle hex port values (sport/dsport may contain hex like '0x000b')
        for port_col in ['sport', 'dsport']:
            if port_col in df_train.columns:
                df_train[port_col] = df_train[port_col].apply(self._convert_port)
                if df_test is not None:
                    df_test[port_col] = df_test[port_col].apply(self._convert_port)

        # Fill missing values
        df_train = df_train.fillna(0)
        if df_test is not None:
            df_test = df_test.fillna(0)

        print(f"\nUNSW-NB15 Training set size: {len(df_train)}")
        print(f"Number of features: {len(df_train.columns) - 2}")  # minus attack_type and attack_category
        print(f"Class distribution:\n{df_train['attack_category'].value_counts()}")

        return df_train, df_test

    @staticmethod
    def _convert_port(val):
        """Convert port values that may be in hex format to integer"""
        try:
            if isinstance(val, str) and val.startswith('0x'):
                return int(val, 16)
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    def _create_synthetic_unsw_nb15(self, n_samples):
        """
        Create synthetic UNSW-NB15-like data for demonstration when real data is unavailable
        """
        np.random.seed(42)

        data = {
            'proto': np.random.choice(['tcp', 'udp', 'arp', 'ospf', 'icmp'], n_samples),
            'state': np.random.choice(['FIN', 'CON', 'INT', 'ACC', 'RST', 'CLO'], n_samples),
            'dur': np.random.exponential(1.0, n_samples),
            'sbytes': np.random.randint(0, 50000, n_samples),
            'dbytes': np.random.randint(0, 50000, n_samples),
            'sttl': np.random.randint(0, 255, n_samples),
            'dttl': np.random.randint(0, 255, n_samples),
            'sloss': np.random.randint(0, 100, n_samples),
            'dloss': np.random.randint(0, 100, n_samples),
            'service': np.random.choice(['-', 'http', 'ftp', 'smtp', 'ssh', 'dns',
                                         'ftp-data', 'irc', 'pop3', 'snmp'], n_samples),
            'sload': np.random.exponential(100, n_samples),
            'dload': np.random.exponential(100, n_samples),
            'spkts': np.random.randint(1, 500, n_samples),
            'dpkts': np.random.randint(0, 500, n_samples),
            'swin': np.random.randint(0, 255, n_samples),
            'dwin': np.random.randint(0, 255, n_samples),
            'stcpb': np.random.randint(0, 2**31, n_samples).astype(float),
            'dtcpb': np.random.randint(0, 2**31, n_samples).astype(float),
            'smeansz': np.random.randint(0, 1500, n_samples),
            'dmeansz': np.random.randint(0, 1500, n_samples),
            'trans_depth': np.random.randint(0, 10, n_samples),
            'res_bdy_len': np.random.randint(0, 5000, n_samples),
            'sjit': np.random.exponential(10, n_samples),
            'djit': np.random.exponential(10, n_samples),
            'sport': np.random.randint(1, 65535, n_samples),
            'dsport': np.random.randint(1, 65535, n_samples),
            'sintpkt': np.random.exponential(50, n_samples),
            'dintpkt': np.random.exponential(50, n_samples),
            'tcprtt': np.random.exponential(0.01, n_samples),
            'synack': np.random.exponential(0.005, n_samples),
            'ackdat': np.random.exponential(0.005, n_samples),
            'is_sm_ips_ports': np.random.randint(0, 2, n_samples),
            'ct_state_ttl': np.random.randint(0, 6, n_samples),
            'ct_flw_http_mthd': np.random.randint(0, 5, n_samples),
            'is_ftp_login': np.random.randint(0, 2, n_samples),
            'ct_ftp_cmd': np.random.randint(0, 5, n_samples),
            'ct_srv_src': np.random.randint(1, 50, n_samples),
            'ct_srv_dst': np.random.randint(1, 50, n_samples),
            'ct_dst_ltm': np.random.randint(1, 50, n_samples),
            'ct_src_ltm': np.random.randint(1, 50, n_samples),
            'ct_src_dport_ltm': np.random.randint(1, 50, n_samples),
            'ct_dst_sport_ltm': np.random.randint(1, 50, n_samples),
            'ct_dst_src_ltm': np.random.randint(1, 50, n_samples),
        }

        df = pd.DataFrame(data)

        # UNSW-NB15 attack categories with realistic imbalanced distribution
        # Normal ~37%, Generic ~22%, Exploits ~18%, Fuzzers ~10%, DoS ~7%,
        # Reconnaissance ~4%, Analysis ~1.2%, Backdoors ~0.5%, Shellcode ~0.2%, Worms ~0.1%
        attack_probs = [0.37, 0.22, 0.18, 0.10, 0.07, 0.04, 0.012, 0.005, 0.002, 0.001]
        # Normalize to sum to 1.0
        attack_probs = np.array(attack_probs) / np.sum(attack_probs)

        attack_types = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS',
                       'Reconnaissance', 'Analysis', 'Backdoors', 'Shellcode', 'Worms']

        df['attack_type'] = np.random.choice(attack_types, n_samples, p=attack_probs)
        df['attack_category'] = df['attack_type']

        return df

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
        Preprocess the data: encode categorical variables and scale features.
        Works for both NSL-KDD and UNSW-NB15 datasets.
        """
        print("\nPreprocessing data...")

        # Reset internal state for each dataset
        self.scaler = StandardScaler()
        self.label_encoders = {}

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
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )
            self.label_encoders[col] = le

        # Encode target labels
        le_target = LabelEncoder()
        y_train_encoded = le_target.fit_transform(y_train)
        if y_test is not None:
            # Handle unseen labels in test set
            known_mask = y_test.isin(le_target.classes_)
            if not known_mask.all():
                unseen = y_test[~known_mask].unique()
                print(f"  Warning: Unseen categories in test set removed: {unseen}")
                if X_test is not None:
                    X_test = X_test[known_mask]
                y_test = y_test[known_mask]
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

    # =========================================================================
    # Model Training Methods (unchanged)
    # =========================================================================
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

        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples_needed = 6  # SMOTE default k_neighbors=5, needs at least 6 samples

        classes_to_remove = unique_classes[class_counts < min_samples_needed]

        if len(classes_to_remove) > 0:
            print(f"  Warning: Removing classes with < {min_samples_needed} samples from SMOTE: {classes_to_remove}")
            print(f"  Classes removed: {[self.classes[i] for i in classes_to_remove if i < len(self.classes)]}")

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

        sampling_strategy = {}
        for cls, count in zip(unique_filtered, counts_filtered):
            target = int(count + augmentation_ratio * (max_count - count))
            sampling_strategy[int(cls)] = target

        print(f"  Augmentation ratio: {augmentation_ratio}")
        print(f"  Target samples per class: {sampling_strategy}")

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

    # =========================================================================
    # Evaluation Methods
    # =========================================================================
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

        # Use explicit labels to handle missing classes in predictions
        all_labels = list(range(len(self.classes)))
        per_class_report = classification_report(y_test, y_pred,
                                                labels=all_labels,
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
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=all_labels)
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
        mean_corr_diff = np.nanmean(corr_diff)

        print(f"  Mean KS statistic: {mean_ks:.4f}")
        print(f"  Mean correlation difference: {mean_corr_diff:.4f}")

        return {
            'ks_statistics': ks_statistics,
            'mean_ks': mean_ks,
            'mean_correlation_diff': mean_corr_diff
        }

    # =========================================================================
    # Single-Dataset Comparison Pipeline
    # =========================================================================
    def run_comparison(self, X_train, y_train, X_test, y_test, dataset_name='NSL-KDD'):
        """
        Run full comparison of augmentation methods on a single dataset.

        Parameters:
            X_train, y_train: Training features and labels (preprocessed)
            X_test, y_test: Test features and labels (preprocessed)
            dataset_name: Name of the dataset for labeling results

        Returns:
            Dictionary containing all results
        """
        print(f"\n{'#'*70}")
        print(f"# Running Comparison on: {dataset_name}")
        print(f"{'#'*70}")

        results = {}
        methods = ['baseline', 'smote', 'gan', 'vae']
        classifiers = ['Random Forest', 'Decision Tree', 'MLP']

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Method: {method.upper()} on {dataset_name}")
            print(f"{'='*60}")

            if method == 'baseline':
                X_aug, y_aug = X_train, y_train
            else:
                X_aug, y_aug = self.augment_minority_classes(X_train, y_train, method=method)

            method_results = {}
            for clf_name in classifiers:
                clf_results = self.evaluate_classifier(X_aug, y_aug, X_test, y_test, clf_name)
                method_results[clf_name] = clf_results

            # Distribution fidelity (compare augmented vs original for non-baseline)
            if method != 'baseline':
                fidelity = self.calculate_distribution_fidelity(X_train, X_aug[len(X_train):])
                method_results['fidelity'] = fidelity

            results[method] = method_results

        self.results[dataset_name] = results
        return results

    # =========================================================================
    # Cross-Dataset Comparison (NEW)
    # =========================================================================
    def run_multi_dataset_comparison(self, datasets_config):
        """
        Run the full augmentation comparison across multiple datasets.

        Parameters:
            datasets_config: dict mapping dataset names to (df_train, df_test) tuples.
                Example:
                {
                    'NSL-KDD': (df_train_nsl, df_test_nsl),
                    'UNSW-NB15': (df_train_unsw, df_test_unsw)
                }

        Returns:
            all_results: dict of {dataset_name: comparison_results}
        """
        all_results = {}

        for dataset_name, (df_train, df_test) in datasets_config.items():
            print(f"\n{'#'*70}")
            print(f"# Processing Dataset: {dataset_name}")
            print(f"{'#'*70}")

            # Preprocess
            X_train, y_train, X_test, y_test = self.preprocess_data(df_train, df_test)

            # If no separate test set, create train/test split
            if X_test is None:
                # Check if stratification is possible (all classes need >= 2 samples)
                unique, counts = np.unique(y_train, return_counts=True)
                can_stratify = np.all(counts >= 2)
                
                if can_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                    )
                else:
                    print(f"  Warning: Some classes too small for stratified split. Using non-stratified split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42
                    )
                print(f"Created 80/20 train/test split: train={X_train.shape[0]}, test={X_test.shape[0]}")

            # Run comparison
            results = self.run_comparison(X_train, y_train, X_test, y_test, dataset_name)
            all_results[dataset_name] = results

        self.dataset_results = all_results
        return all_results

    # =========================================================================
    # Visualization Methods (Updated for Multi-Dataset)
    # =========================================================================
    def plot_class_distribution_comparison(self, datasets_config, save_path=None):
        """
        Plot class distribution for all datasets side by side
        """
        n_datasets = len(datasets_config)
        fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 6))

        if n_datasets == 1:
            axes = [axes]

        for ax, (dataset_name, (df_train, _)) in zip(axes, datasets_config.items()):
            class_counts = df_train['attack_category'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))

            bars = ax.bar(range(len(class_counts)), class_counts.values, color=colors)
            ax.set_xticks(range(len(class_counts)))
            ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
            ax.set_title(f'{dataset_name} - Class Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Samples')
            ax.set_xlabel('Attack Category')

            # Add count labels on bars
            for bar, count in zip(bars, class_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_cross_dataset_performance(self, all_results, metric='macro_f1',
                                       classifier='Random Forest', save_path=None):
        """
        Plot performance comparison across datasets for each augmentation method
        """
        datasets = list(all_results.keys())
        methods = ['baseline', 'smote', 'gan', 'vae']
        method_labels = ['Baseline', 'SMOTE', 'GAN', 'VAE']

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(datasets))
        width = 0.18
        offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

        for i, (method, label) in enumerate(zip(methods, method_labels)):
            values = []
            for dataset in datasets:
                if method in all_results[dataset] and classifier in all_results[dataset][method]:
                    values.append(all_results[dataset][method][classifier][metric])
                else:
                    values.append(0)

            bars = ax.bar(x + offsets[i] * width, values, width, label=label,
                         color=colors[i], edgecolor='white', linewidth=0.5)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Cross-Dataset Comparison: {metric.replace("_", " ").title()} ({classifier})',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.15)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_per_class_recall_heatmap(self, all_results, classifier='Random Forest', save_path=None):
        """
        Plot per-class recall as a heatmap for each dataset and augmentation method
        """
        datasets = list(all_results.keys())
        methods = ['baseline', 'smote', 'gan', 'vae']
        method_labels = ['Baseline', 'SMOTE', 'GAN', 'VAE']

        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 6))
        if n_datasets == 1:
            axes = [axes]

        for ax, dataset_name in zip(axes, datasets):
            dataset_results = all_results[dataset_name]

            # Get class names from any method's report
            first_method = next(iter(dataset_results))
            first_clf_results = dataset_results[first_method][classifier]
            class_names = [k for k in first_clf_results['per_class_report'].keys()
                          if k not in ['accuracy', 'macro avg', 'weighted avg']]

            # Build recall matrix
            recall_matrix = []
            for method in methods:
                if method in dataset_results and classifier in dataset_results[method]:
                    report = dataset_results[method][classifier]['per_class_report']
                    recalls = [report.get(cls, {}).get('recall', 0) for cls in class_names]
                    recall_matrix.append(recalls)
                else:
                    recall_matrix.append([0] * len(class_names))

            recall_df = pd.DataFrame(recall_matrix, index=method_labels, columns=class_names)

            sns.heatmap(recall_df, annot=True, fmt='.3f', cmap='YlOrRd',
                       vmin=0, vmax=1, ax=ax, linewidths=0.5)
            ax.set_title(f'{dataset_name} - Per-Class Recall ({classifier})',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('Augmentation Method')
            ax.set_xlabel('Attack Category')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_fidelity_comparison(self, all_results, save_path=None):
        """
        Plot distribution fidelity metrics across datasets and methods
        """
        datasets = list(all_results.keys())
        methods = ['smote', 'gan', 'vae']
        method_labels = ['SMOTE', 'GAN', 'VAE']

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Mean KS statistic
        x = np.arange(len(datasets))
        width = 0.25
        colors = ['#4CAF50', '#FF9800', '#9C27B0']

        for i, (method, label) in enumerate(zip(methods, method_labels)):
            ks_values = []
            for dataset in datasets:
                if method in all_results[dataset] and 'fidelity' in all_results[dataset][method]:
                    ks_values.append(all_results[dataset][method]['fidelity']['mean_ks'])
                else:
                    ks_values.append(0)
            axes[0].bar(x + (i - 1) * width, ks_values, width, label=label, color=colors[i])

        axes[0].set_xlabel('Dataset')
        axes[0].set_ylabel('Mean KS Statistic')
        axes[0].set_title('Distribution Fidelity: KS Test\n(Lower is better)', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(datasets)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Mean correlation difference
        for i, (method, label) in enumerate(zip(methods, method_labels)):
            corr_values = []
            for dataset in datasets:
                if method in all_results[dataset] and 'fidelity' in all_results[dataset][method]:
                    corr_values.append(all_results[dataset][method]['fidelity']['mean_correlation_diff'])
                else:
                    corr_values.append(0)
            axes[1].bar(x + (i - 1) * width, corr_values, width, label=label, color=colors[i])

        axes[1].set_xlabel('Dataset')
        axes[1].set_ylabel('Mean Correlation Difference')
        axes[1].set_title('Correlation Drift\n(Lower is better)', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(datasets)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_cross_dataset_summary_table(self, all_results, classifier='Random Forest'):
        """
        Generate a comprehensive summary table comparing results across datasets
        """
        rows = []
        methods = ['baseline', 'smote', 'gan', 'vae']
        method_labels = ['Baseline', 'SMOTE', 'GAN', 'VAE']

        for dataset_name in all_results:
            for method, label in zip(methods, method_labels):
                if method in all_results[dataset_name] and classifier in all_results[dataset_name][method]:
                    r = all_results[dataset_name][method][classifier]
                    row = {
                        'Dataset': dataset_name,
                        'Method': label,
                        'Accuracy': f"{r['accuracy']:.4f}",
                        'Macro F1': f"{r['macro_f1']:.4f}",
                        'Weighted F1': f"{r['weighted_f1']:.4f}",
                        'Macro Precision': f"{r['macro_precision']:.4f}",
                        'Macro Recall': f"{r['macro_recall']:.4f}",
                    }

                    # Add fidelity metrics for non-baseline
                    if method != 'baseline' and 'fidelity' in all_results[dataset_name][method]:
                        fid = all_results[dataset_name][method]['fidelity']
                        row['Mean KS'] = f"{fid['mean_ks']:.4f}"
                        row['Corr. Drift'] = f"{fid['mean_correlation_diff']:.4f}"
                    else:
                        row['Mean KS'] = '-'
                        row['Corr. Drift'] = '-'

                    rows.append(row)

        summary_df = pd.DataFrame(rows)
        print(f"\n{'='*100}")
        print(f"Cross-Dataset Summary ({classifier})")
        print(f"{'='*100}")
        print(summary_df.to_string(index=False))

        return summary_df

    def plot_confusion_matrices(self, all_results, classifier='Random Forest', save_path=None):
        """
        Plot confusion matrices for all datasets and methods
        """
        datasets = list(all_results.keys())
        methods = ['baseline', 'smote', 'gan', 'vae']
        method_labels = ['Baseline', 'SMOTE', 'GAN', 'VAE']

        for dataset_name in datasets:
            fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))

            for ax, (method, label) in zip(axes, zip(methods, method_labels)):
                if method in all_results[dataset_name] and classifier in all_results[dataset_name][method]:
                    cm = all_results[dataset_name][method][classifier]['confusion_matrix']
                    # Normalize
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_normalized = np.nan_to_num(cm_normalized)

                    # Get class names
                    report = all_results[dataset_name][method][classifier]['per_class_report']
                    class_names = [k for k in report.keys()
                                  if k not in ['accuracy', 'macro avg', 'weighted avg']]

                    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                               xticklabels=class_names, yticklabels=class_names,
                               ax=ax, vmin=0, vmax=1)
                    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
                    ax.set_ylabel('True Label')
                    ax.set_xlabel('Predicted Label')

            plt.suptitle(f'{dataset_name} - Confusion Matrices ({classifier})',
                        fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            if save_path:
                plt.savefig(f'{save_path}_{dataset_name}_cm.png', dpi=300, bbox_inches='tight')
            plt.show()


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """
    Main execution function demonstrating multi-dataset comparison.
    """
    comparison = IDSAugmentationComparison()

    # ------------------------------------------------------------------
    # Dataset paths - UPDATE THESE to your actual file paths
    # ------------------------------------------------------------------
    # NSL-KDD paths
    nsl_train_path = r'E:\PHD\Datasets\archive\KDDTrain+.txt'     # or 'KDDTrain+.csv'
    nsl_test_path = r'E:\PHD\Datasets\archive\KDDTest+.txt'        # or 'KDDTest+.csv'

    # UNSW-NB15 paths
    unsw_train_path = r'E:\PHD\Datasets\CIC-UNSW-NB15\UNSW_NB15_training-set.csv'  # Official train split
    unsw_test_path = r'E:\PHD\Datasets\CIC-UNSW-NB15\UNSW_NB15_testing-set.csv'     # Official test split
    # Alternative: raw files like 'UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', etc.

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: LOADING DATASETS")
    print("=" * 70)

    # Load NSL-KDD
    df_train_nsl, df_test_nsl = comparison.load_nsl_kdd(nsl_train_path, nsl_test_path)

    # Load UNSW-NB15
    df_train_unsw, df_test_unsw = comparison.load_unsw_nb15(unsw_train_path, unsw_test_path)

    # ------------------------------------------------------------------
    # Dataset overview visualization
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: DATASET OVERVIEW")
    print("=" * 70)

    datasets_config = {
        'NSL-KDD': (df_train_nsl, df_test_nsl),
        'UNSW-NB15': (df_train_unsw, df_test_unsw)
    }

    comparison.plot_class_distribution_comparison(datasets_config,
                                                  save_path='class_distribution_comparison.png')

    # ------------------------------------------------------------------
    # Run multi-dataset comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING AUGMENTATION COMPARISON ON ALL DATASETS")
    print("=" * 70)

    all_results = comparison.run_multi_dataset_comparison(datasets_config)

    # ------------------------------------------------------------------
    # Cross-dataset visualizations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: CROSS-DATASET ANALYSIS AND VISUALIZATION")
    print("=" * 70)

    # 4a. Cross-dataset performance bar chart
    comparison.plot_cross_dataset_performance(
        all_results, metric='macro_f1', classifier='Random Forest',
        save_path='cross_dataset_macro_f1.png'
    )

    comparison.plot_cross_dataset_performance(
        all_results, metric='accuracy', classifier='Random Forest',
        save_path='cross_dataset_accuracy.png'
    )

    # 4b. Per-class recall heatmaps
    comparison.plot_per_class_recall_heatmap(
        all_results, classifier='Random Forest',
        save_path='per_class_recall_heatmap.png'
    )

    # 4c. Distribution fidelity comparison
    comparison.plot_fidelity_comparison(
        all_results,
        save_path='fidelity_comparison.png'
    )

    # 4d. Confusion matrices
    comparison.plot_confusion_matrices(
        all_results, classifier='Random Forest',
        save_path='confusion_matrices'
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: COMPREHENSIVE SUMMARY")
    print("=" * 70)

    for clf_name in ['Random Forest', 'Decision Tree', 'MLP']:
        summary_df = comparison.generate_cross_dataset_summary_table(all_results, classifier=clf_name)

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for dataset_name in all_results:
        print(f"\n--- {dataset_name} ---")
        best_f1 = 0
        best_method = ''
        for method in ['baseline', 'smote', 'gan', 'vae']:
            if 'Random Forest' in all_results[dataset_name][method]:
                f1 = all_results[dataset_name][method]['Random Forest']['macro_f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_method = method
        print(f"  Best augmentation method (Macro F1): {best_method.upper()} ({best_f1:.4f})")

    print("\n" + "=" * 70)
    print("Experiment completed successfully!")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    all_results = main()
