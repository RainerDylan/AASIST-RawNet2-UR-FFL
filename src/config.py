import os

# Base Paths
REPO_DIR = r"D:\SAMPOERNA\Semester 8\Capstone\AASIST-RawNet2-UR-FFL"
DATASET_TRAIN_FLAC = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_train\flac"

# Exact path to the ASVspoof training protocol file
PROTOCOL_FILE_TRAIN = r"D:\SAMPOERNA\Semester 8\Capstone\Dataset\2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

# Data Directories
RAW_DATA_DIR = os.path.join(REPO_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(REPO_DIR, "data", "processed")

# Audio Processing Parameters
SAMPLE_RATE = 16000
MAX_FRAMES = 64600