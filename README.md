# Retail Footfall Counting

## 🔧 Installation :: Clone the repository:
```bash
git clone -b dev-v2.0 https://github.com/your-repo/retail_footfall_counting.git
cd retail_footfall_counting
```

## 💻 Usage

### Basic Command

```bash
python main.py --video_name "config" --heatmap_start "True" --footfall_save_start "True" --debug_screen "True" --video_write "True" --selected_store_cam "config"
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--selected_store_cam` | Camera config file name (e.g., config, config2) | Required |
| `--video_name` | Absolute path to video file or store name | "/tmp/test.mp4" |
| `--footfall_save_start` | Enable/disable footfall data saving | True |
| `--heatmap_start` | Enable/disable heatmap processing | False |
| `--debug_screen` | Show/hide debug visualization | False |
| `--video_write` | Enable/disable processed video saving | False |
| `--help` | For a full list of available options | None |

## ⚙️ Development

The base branch for development is `dev-v1.1`.


## 🔄 Recent Updates

| Feature/Update | Description |
|----------------|-------------|
| Code Optimization | - Automated FPS calculation (5-7 FPS)<br>- JSON writing in separate thread & schedule<br>- Removed unused variables<br>- adding patches<br>- code running check through log file <br>- remove video write and imshow when running through service file<br>- Heatmap calculation and json saving<br>- added OD/Motion on debug screen|
| Grab Scripts | added capturing scripts in a seperate folder |
| Server Port Configuration | Updated footfall / heatmap server port |
| Data Enrichment | Added device serial ID to all JSON files |

